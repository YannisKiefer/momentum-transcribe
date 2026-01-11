# Momentum Transcription Service

A lightweight FastAPI service for voice-to-text transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Features

- 🎤 Transcribe audio files (webm, wav, mp3, ogg)
- ⚡ 4x faster than OpenAI Whisper with same accuracy
- 🔒 MIT licensed, no API costs
- 🚀 Ready for Railway deployment

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Server runs at `http://localhost:8000`

### Test Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@your-audio-file.webm"
```

## Deploy to Railway

1. Push this folder to a new GitHub repo
2. Go to [railway.app](https://railway.app)
3. Create new project → Deploy from GitHub
4. Select your repo
5. Railway auto-builds and deploys
6. Go to Settings → Networking → Generate Domain
7. Copy your URL (e.g., `https://your-app.up.railway.app`)

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "base.en"
}
```

### `POST /transcribe`
Transcribe audio file to text.

**Request:**
- `audio`: Audio file (multipart/form-data)

**Response:**
```json
{
  "text": "Hello, this is a test recording.",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "Hello, this is a test recording."}
  ],
  "duration": 2.5,
  "language": "en"
}
```

## Configuration

The service uses the `base.en` model by default, which provides a good balance of speed and accuracy for English transcription.

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | 75MB | Fastest | Good |
| base.en | 150MB | Fast | Better |
| small.en | 500MB | Medium | Great |
| medium.en | 1.5GB | Slow | Excellent |

To change the model, edit `main.py` line 35.

## Cost Estimate (Railway)

| Usage | Monthly Cost |
|-------|-------------|
| Hobby/Dev | ~$5 |
| 100 users | ~$10 |
| 1,000 users | ~$25-50 |

## License

MIT
