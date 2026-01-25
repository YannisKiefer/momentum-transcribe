"""
Momentum Transcription Service
FastAPI + faster-whisper for voice-to-text transcription
Deploy on Railway

v2.0: Added API key auth for external services (Clawdbot)
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(
    title="Momentum Transcription Service",
    description="Voice-to-text transcription using faster-whisper",
    version="2.0.0"
)

# CORS - Allow Momentum domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://dailymomentum.io",
        "https://www.dailymomentum.io",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# API Key for external services (Clawdbot, etc.)
# Set via TRANSCRIBE_API_KEY environment variable on Railway
API_KEY = os.environ.get("TRANSCRIBE_API_KEY")

# Load model on startup (cached in memory)
# Using 'base' (multilingual) instead of 'base.en' (English-only)
print("🎤 Loading Whisper model (base - multilingual)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Model loaded and ready!")


def verify_auth(authorization: str | None, origin: str | None) -> bool:
    """
    Verify request authentication.
    
    Allows:
    1. API Key auth (Bearer <api_key>) - for Clawdbot
    2. Requests from allowed CORS origins (browser requests from Momentum)
    3. No auth if API_KEY not configured (backwards compat)
    """
    # If no API key configured, allow all (backwards compat)
    if not API_KEY:
        return True
    
    # Check API key auth
    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            if token == API_KEY:
                return True
    
    # Allow browser requests from Momentum domains (CORS handles this)
    if origin and any(allowed in origin for allowed in [
        "localhost",
        "dailymomentum.io",
        "vercel.app"
    ]):
        return True
    
    return False


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Momentum Transcription",
        "status": "healthy",
        "model": "base",
        "version": "2.0.0"
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "auth_required": bool(API_KEY)
    }


@app.post("/transcribe")
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    authorization: str | None = Header(None),
):
    """
    Transcribe audio file to text.
    
    Accepts: audio/webm, audio/wav, audio/mp3, audio/ogg, audio/mpeg
    
    Headers:
        - Authorization: Bearer <api_key> (required for non-browser requests)
    
    Returns:
        {
            "text": "Full transcription text",
            "segments": [{"start": 0.0, "end": 2.5, "text": "..."}],
            "duration": 10.5,
            "language": "en"
        }
    """
    # Check authentication
    origin = request.headers.get("origin")
    
    if not verify_auth(authorization, origin):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide API key via Authorization: Bearer <key>"
        )
    
    # Validate content type
    content_type = audio.content_type or ""
    if not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid content type: {content_type}. Expected audio file."
        )
    
    # Determine file extension from content type
    ext_map = {
        "audio/webm": ".webm",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "application/octet-stream": ".webm",  # Default for unknown
    }
    suffix = ext_map.get(content_type, ".webm")
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe with faster-whisper
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect results
        text_parts = []
        segment_list = []
        
        for segment in segments:
            text_parts.append(segment.text)
            segment_list.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
        
        full_text = " ".join(text_parts).strip()
        
        # Handle empty transcription
        if not full_text:
            return {
                "text": "",
                "segments": [],
                "duration": info.duration if info else 0,
                "language": info.language if info else "unknown",
                "message": "No speech detected in audio"
            }
        
        return {
            "text": full_text,
            "segments": segment_list,
            "duration": round(info.duration, 2) if info else 0,
            "language": info.language if info else "unknown"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
