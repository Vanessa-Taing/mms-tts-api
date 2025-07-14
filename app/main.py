# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from app.tts_engine import TTSEngine
from app.config import config
import tempfile
import logging
import json
from scipy.io.wavfile import write

app = FastAPI(
    title="MMS TTS Service",
    description="Massively Multilingual Speech Text-to-Speech API",
    version="1.0"
)

tts_engine = TTSEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize TTS engine and pre-download models"""
    logging.basicConfig(level=logging.INFO)
    await tts_engine.initialize()
    logging.info("Service initialized and ready")

@app.get("/languages")
async def list_languages():
    """List available languages"""
    return {
        "available": sorted(tts_engine.available_languages),
        "default": config.DEFAULT_LANGUAGE
    }
    
@app.post("/synthesize")
async def synthesize(request: Request):
    try:
        # Handle both JSON and form data
        try:
            data = await request.json()
            text = data.get('text', '')
            lang = data.get('lang', config.DEFAULT_LANGUAGE)
        except:
            form_data = await request.form()
            text = form_data.get('text', '')
            lang = form_data.get('lang', config.DEFAULT_LANGUAGE)
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        audio, sr = await tts_engine.synthesize(text, lang)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, sr, audio)
            return FileResponse(f.name, media_type="audio/wav")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Service health check"""
    return {"status": "healthy", "languages_loaded": len(tts_engine.available_languages)}