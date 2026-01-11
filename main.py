"""
Momentum Transcription Service
FastAPI + faster-whisper for voice-to-text transcription
Deploy on Railway
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(
    title="Momentum Transcription Service",
    description="Voice-to-text transcription using faster-whisper",
    version="1.0.0"
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

# Load model on startup (cached in memory)
# Using 'base' (multilingual) instead of 'base.en' (English-only)
print("🎤 Loading Whisper model (base - multilingual)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Model loaded and ready!")


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Momentum Transcription",
        "status": "healthy",
        "model": "base"
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model": "base",
        "device": "cpu",
        "compute_type": "int8"
    }


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    
    Accepts: audio/webm, audio/wav, audio/mp3, audio/ogg, audio/mpeg
    
    Returns:
        {
            "text": "Full transcription text",
            "segments": [{"start": 0.0, "end": 2.5, "text": "..."}],
            "duration": 10.5,
            "language": "en"
        }
    """
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
                "language": "en",
                "message": "No speech detected in audio"
            }
        
        return {
            "text": full_text,
            "segments": segment_list,
            "duration": round(info.duration, 2) if info else 0,
            "language": info.language if info else "en"
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
