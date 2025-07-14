# app/main.py
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import FileResponse
from app.tts_engine import TTSEngine
from app.config import config
import tempfile
import logging
import json
from scipy.io.wavfile import write
import unicodedata
from typing import Optional

app = FastAPI(
    title="MMS TTS Service",
    description="Massively Multilingual Speech Text-to-Speech API",
    version="1.0"
)

tts_engine = TTSEngine()

def normalize_unicode_text(text: str) -> str:
    """Normalize Unicode text to handle different encodings"""
    if not text:
        return ""
    
    # Normalize Unicode to NFC form (canonical decomposition, then canonical composition)
    normalized = unicodedata.normalize('NFC', text)
    
    # Log the actual characters for debugging
    logging.info(f"Original text bytes: {text.encode('utf-8')}")
    logging.info(f"Normalized text: {normalized}")
    logging.info(f"Character codes: {[ord(c) for c in normalized[:20]]}")  # First 20 chars
    
    return normalized

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
async def synthesize(
    text: str = Form(...),
    lang: str = Form(default=config.DEFAULT_LANGUAGE)
):
    """
    Synthesize speech from text using form data
    This ensures proper handling of Unicode characters
    """
    try:
        # Normalize and validate input
        text = normalize_unicode_text(text)
        
        if not text.strip():
            raise HTTPException(400, "No valid text provided after normalization")
        
        # Validate language
        if lang not in tts_engine.available_languages:
            raise HTTPException(400, f"Language '{lang}' not supported. Available: {sorted(tts_engine.available_languages)}")
        
        logging.info(f"Synthesizing text: '{text}' in language: {lang}")
        
        audio, sr = await tts_engine.synthesize(text, lang)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, sr, audio)
            return FileResponse(
                f.name,
                media_type="audio/wav",
                filename=f"tts_{lang}.wav"
            )
            
    except Exception as e:
        logging.error(f"Synthesis error: {str(e)}")
        raise HTTPException(500, f"Synthesis failed: {str(e)}")

@app.post("/synthesize_json")
async def synthesize_json(request: Request):
    """
    Alternative JSON endpoint for synthesis
    """
    try:
        data = await request.json()
        text = data.get('text', '')
        lang = data.get('lang', config.DEFAULT_LANGUAGE)
        
        # Normalize Unicode text
        text = normalize_unicode_text(text)
        
        if not text.strip():
            raise HTTPException(400, "No valid text provided")
        
        if lang not in tts_engine.available_languages:
            raise HTTPException(400, f"Language '{lang}' not supported")
        
        audio, sr = await tts_engine.synthesize(text, lang)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, sr, audio)
            return FileResponse(
                f.name,
                media_type="audio/wav",
                filename=f"tts_{lang}.wav"
            )
            
    except Exception as e:
        logging.error(f"JSON synthesis error: {str(e)}")
        raise HTTPException(500, f"Synthesis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Service health check"""
    return {"status": "healthy", "languages_loaded": len(tts_engine.available_languages)}

@app.get("/debug/text")
async def debug_text(text: str):
    """Debug endpoint to inspect text processing"""
    try:
        # Get raw bytes representation
        byte_repr = text.encode('utf-8').hex()
        
        # Get character codes
        char_codes = [ord(c) for c in text]
        
        return {
            "original": text,
            "normalized": text.lower(),
            "length": len(text),
            "char_codes": char_codes,
            "bytes": byte_repr
        }
    except Exception as e:
        raise HTTPException(400, detail=str(e))