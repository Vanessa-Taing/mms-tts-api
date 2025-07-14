# app/tts_engine.py (fixed version)
import os
import re
import tempfile
import torch
import commons
import utils
from pathlib import Path
from models import SynthesizerTrn
from scipy.io.wavfile import write
from typing import Optional
from app.config import config
import logging
import aiofiles
import asyncio
import subprocess

logger = logging.getLogger(__name__)

def preprocess_char(text, lang=None):
    """
    Special treatment of characters in certain languages
    """
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text

class TextMapper:
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        """Convert text to sequence of IDs - now with proper error handling"""
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            if symbol in self._symbol_to_id:
                symbol_id = self._symbol_to_id[symbol]
                sequence += [symbol_id]
            else:
                logger.warning(f"Symbol '{symbol}' not in vocabulary")
                # Skip the symbol instead of crashing
                continue
        return sequence

    def filter_oov(self, text):
        """Filter out-of-vocabulary characters - THIS WAS MISSING"""
        val_chars = self._symbol_to_id
        logger.info(f"Input text for OOV filtering: '{text}'")
        logger.info(f"Available vocabulary chars: {list(val_chars.keys())[:20]}...")  # Show first 20
        
        # Debug: show which characters are being filtered
        filtered_chars = []
        removed_chars = []
        for char in text:
            if char in val_chars:
                filtered_chars.append(char)
            else:
                removed_chars.append(char)
        
        txt_filt = "".join(filtered_chars)
        logger.info(f"Removed characters: {removed_chars}")
        logger.info(f"text after filtering OOV: '{txt_filt}'")
        return txt_filt

    async def uromanize(self, text, lang):
        """Async uromanization using perl script"""
        uroman_pl = Path(os.environ.get("UROMAN_DIR", "/app/uroman")) / "bin" / "uroman.pl"
        
        if not uroman_pl.exists():
            raise FileNotFoundError(f"uroman.pl not found at {uroman_pl}")
        # Ensure text is properly encoded
        text = text.encode('utf-8').decode('utf-8')
        # Use synchronous approach similar to the working example
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf_in, \
             tempfile.NamedTemporaryFile(mode='r', delete=False) as tf_out:
            
            try:
                # Write input text
                with open(tf_in.name, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Run uromanization command
                cmd = f"perl {uroman_pl} -l xxx < {tf_in.name} > {tf_out.name}"
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()
                
                if proc.returncode != 0:
                    raise RuntimeError(f"Uromanization failed: {stderr.decode()}")
                
                # Read output
                with open(tf_out.name, "r", encoding="utf-8") as f:
                    outtext = f.read().strip()
                    outtext = re.sub(r"\s+", " ", outtext)
                
                return outtext
                
            finally:
                # Clean up temporary files
                os.unlink(tf_in.name)
                os.unlink(tf_out.name)

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

class TTSEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.available_languages = set()
        
    async def initialize(self):
        """Pre-download and cache models"""
        for lang in config.PREDOWNLOAD_LANGUAGES:
            try:
                await self._load_model(lang)
                logger.info(f"Loaded model for {lang}")
                self.available_languages.add(lang)
            except Exception as e:
                logger.error(f"Failed to load {lang}: {str(e)}")

    async def _download_model(self, lang):
        """Download model if not exists"""
        lang_dir = config.MODEL_DIR / lang
        if lang_dir.exists():
            return lang_dir
        
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        lang_fn = config.MODEL_DIR / f"{lang}.tar.gz"
        
        # Download model
        download_cmd = [
            "wget", "-q", 
            f"https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz",
            "-O", str(lang_fn)
        ]
        proc = await asyncio.create_subprocess_exec(*download_cmd)
        await proc.wait()
        
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to download {lang} model")
        
        # Extract model
        extract_cmd = ["tar", "zxf", str(lang_fn), "-C", str(config.MODEL_DIR)]
        proc = await asyncio.create_subprocess_exec(*extract_cmd)
        await proc.wait()
        
        # Clean up
        lang_fn.unlink(missing_ok=True)
        return lang_dir

    async def _load_model(self, lang):
        """Load or download model for a language"""
        if lang in self.models:
            return self.models[lang]
        
        lang_dir = await self._download_model(lang)
        vocab_file = lang_dir / "vocab.txt"
        config_file = lang_dir / "config.json"
        
        hps = utils.get_hparams_from_file(config_file)
        text_mapper = TextMapper(vocab_file)
        
        net_g = SynthesizerTrn(
            len(text_mapper.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).to(self.device)
        
        net_g.eval()
        g_pth = lang_dir / "G_100000.pth"
        utils.load_checkpoint(g_pth, net_g, None)
        
        model_data = {
            "net_g": net_g,
            "text_mapper": text_mapper,
            "hps": hps
        }
        self.models[lang] = model_data
        return model_data

    async def preprocess_text(self, text, lang):
        """Handle text preprocessing with uromanization - FIXED VERSION"""
        model = await self._load_model(lang)
        text_mapper = model["text_mapper"]
        hps = model["hps"]
        
        logger.info(f"Original text: '{text}'")
        
        # Character preprocessing
        text = preprocess_char(text, lang=lang)
        logger.info(f"After preprocess_char: '{text}'")
        
        # Check if uromanization is needed
        is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
        logger.info(f"Is uroman needed? {is_uroman} (training_files: {hps.data.training_files})")
        
        if is_uroman:
            logger.info(f"Uromanizing text for {lang}")
            text = await text_mapper.uromanize(text, lang)
            logger.info(f"Uromanized text: '{text}'")
        
        # Convert to lowercase
        text = text.lower()
        logger.info(f"After lowercase: '{text}'")
        
        # Filter out-of-vocabulary characters - THIS WAS THE MISSING PIECE
        text = text_mapper.filter_oov(text)
        
        if not text.strip():
            raise ValueError("No valid characters found after preprocessing")
        
        return text_mapper.get_text(text, hps)

    async def synthesize(self, text: str, lang: str = config.DEFAULT_LANGUAGE):
        """Synthesize speech with proper text preprocessing"""
        try:
            model = await self._load_model(lang)
            stn_tst = await self.preprocess_text(text, lang)
            
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(self.device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
                audio = model["net_g"].infer(
                    x_tst, x_tst_lengths, 
                    noise_scale=.667,
                    noise_scale_w=0.8, 
                    length_scale=1.0
                )[0][0,0].cpu().float().numpy()
            
            return audio, model["hps"].data.sampling_rate
        
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            raise