import 'dotenv/config';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import cors from 'cors';
import express from 'express';
import OpenAI, { toFile } from 'openai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const audioDir = path.join(__dirname, 'public', 'audio');

const app = express();

if (!process.env.OPENAI_API_KEY) {
  console.error('OPENAI_API_KEY is missing. Copy .env.example to .env and set your key.');
  process.exit(1);
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  maxRetries: 3,
  timeout: 90000
});

const PORT = Number(process.env.PORT || 3124);
const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || `http://127.0.0.1:${PORT}`;
const AI_NPC_TOKEN = process.env.AI_NPC_TOKEN || 'change-moi';
const TEXT_MODEL = process.env.TEXT_MODEL || 'gpt-5.2';
const STT_MODEL = process.env.STT_MODEL || 'gpt-4o-transcribe';
const TTS_MODEL = process.env.TTS_MODEL || 'gpt-4o-mini-tts';

app.use(cors());
app.use(express.json({ limit: '14mb' }));
app.use('/audio', express.static(audioDir, {
  maxAge: '15m',
  setHeaders: (res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
  }
}));

function requireToken(req, res, next) {
  if (req.header('x-ai-npc-token') !== AI_NPC_TOKEN) {
    res.status(401).json({ error: 'Invalid token' });
    return;
  }
  next();
}

function cleanReply(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .replace(/["“”]/g, '')
    .trim()
    .slice(0, 420);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getPublicError(error) {
  const code = error?.code || error?.cause?.code;
  const status = error?.status;

  if (code === 'ECONNRESET' || error?.name === 'APIConnectionError') {
    return 'Connexion OpenAI interrompue. Reessaie dans quelques secondes.';
  }

  if (status === 401) {
    return 'Cle OpenAI invalide ou absente.';
  }

  if (status === 429) {
    return 'Quota ou limite OpenAI atteint.';
  }

  if (status === 400) {
    return 'Audio refuse par OpenAI. Essaie une phrase plus courte.';
  }

  return 'Traitement IA impossible pour le moment.';
}

async function withOpenAIRetry(label, fn) {
  let lastError;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const code = error?.code || error?.cause?.code || error?.status || error?.name;
      console.warn(`${label} failed, attempt ${attempt}/3:`, code);

      if (error?.status && error.status < 500 && error.status !== 429) {
        throw error;
      }

      if (attempt < 3) {
        await sleep(900 * attempt);
      }
    }
  }

  throw lastError;
}

async function transcribeAudio({ audioBase64, mimeType }) {
  const audioBuffer = Buffer.from(audioBase64, 'base64');
  const extension = mimeType.includes('wav') ? 'wav' : 'webm';
  const file = await toFile(audioBuffer, `player.${extension}`, { type: mimeType });

  const transcription = await withOpenAIRetry('transcription', () => {
    return openai.audio.transcriptions.create({
      model: STT_MODEL,
      file,
      language: 'fr'
    });
  });

  return transcription.text?.trim() || '';
}

async function createNpcReply({ npcName, personality, playerName, transcript }) {
  const response = await withOpenAIRetry('reply', () => {
    return openai.responses.create({
      model: TEXT_MODEL,
      instructions: [
        personality,
        'Tu es dans une scene roleplay FiveM a Los Santos.',
        'Reponds uniquement avec la replique du PNJ.',
        'Maximum deux phrases courtes.',
        'Ne mentionne pas OpenAI, IA, API, backend, script ou HRP.',
        'Si la demande est dangereuse, impossible ou hors RP, refuse naturellement dans le role.'
      ].join('\n'),
      input: `${playerName} dit a ${npcName}: ${transcript}`
    });
  });

  return cleanReply(response.output_text);
}

async function synthesizeSpeech({ reply, voice, voiceInstructions }) {
  await fs.mkdir(audioDir, { recursive: true });

  const fileName = `${Date.now()}-${crypto.randomUUID()}.mp3`;
  const filePath = path.join(audioDir, fileName);

  const speech = await withOpenAIRetry('speech', () => {
    return openai.audio.speech.create({
      model: TTS_MODEL,
      voice: voice || 'coral',
      input: reply,
      instructions: voiceInstructions || 'Voix francaise naturelle, conversationnelle.',
      response_format: 'mp3'
    });
  });

  const buffer = Buffer.from(await speech.arrayBuffer());
  await fs.writeFile(filePath, buffer);

  return `${PUBLIC_BASE_URL}/audio/${fileName}`;
}

app.get('/health', (_, res) => {
  res.json({ ok: true });
});

app.post('/conversation', requireToken, async (req, res) => {
  try {
    const {
      npcName,
      personality,
      playerName,
      voice,
      voiceInstructions,
      audioBase64,
      mimeType = 'audio/webm'
    } = req.body || {};

    if (!audioBase64 || !npcName || !personality) {
      res.status(400).json({ error: 'Missing required fields' });
      return;
    }

    const transcript = await transcribeAudio({ audioBase64, mimeType });
    if (!transcript) {
      res.status(400).json({ error: 'Empty transcript' });
      return;
    }

    const reply = await createNpcReply({
      npcName,
      personality,
      playerName: playerName || 'Un citoyen',
      transcript
    });

    const audioUrl = await synthesizeSpeech({ reply, voice, voiceInstructions });

    res.json({
      transcript,
      reply,
      audioUrl
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: 'AI processing failed',
      message: getPublicError(error)
    });
  }
});

setInterval(async () => {
  try {
    await fs.mkdir(audioDir, { recursive: true });
    const files = await fs.readdir(audioDir);
    const maxAgeMs = 30 * 60 * 1000;
    const now = Date.now();

    await Promise.all(files.map(async (file) => {
      const filePath = path.join(audioDir, file);
      const stat = await fs.stat(filePath);
      if (now - stat.mtimeMs > maxAgeMs) {
        await fs.unlink(filePath);
      }
    }));
  } catch {
    // Cleanup best effort.
  }
}, 10 * 60 * 1000);

app.listen(PORT, () => {
  console.log(`AI NPC voice backend listening on ${PUBLIC_BASE_URL}`);
});
