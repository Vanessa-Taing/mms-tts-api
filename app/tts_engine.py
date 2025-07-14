# app/tts_engine.py
import asyncio
from asyncio import subprocess
import os
import torch
import commons
import utils
from pathlib import Path
from models import SynthesizerTrn
from scipy.io.wavfile import write
from typing import Optional
from app.config import config
import logging

logger = logging.getLogger(__name__)

class TextMapper:
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            if symbol in self._symbol_to_id:
                symbol_id = self._symbol_to_id[symbol]
                sequence += [symbol_id]
            else:
                logger.warning(f"Symbol '{symbol}' not in vocabulary")
        return sequence

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

class TTSEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}  # Loaded models cache
        self.available_languages = set()
        
    async def initialize(self):
        """Pre-download and cache models during startup"""
        for lang in config.PREDOWNLOAD_LANGUAGES:
            try:
                await self._load_model(lang)
                logger.info(f"Successfully loaded model for language: {lang}")
                self.available_languages.add(lang)
            except Exception as e:
                logger.error(f"Failed to load model for {lang}: {str(e)}")

    async def _download_model(self, lang):
        """Download model if not exists"""
        lang_dir = config.MODEL_DIR / lang
        if lang_dir.exists():
            return lang_dir
        
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        lang_fn = config.MODEL_DIR / f"{lang}.tar.gz"
        
        # Use asyncio.create_subprocess_exec instead of subprocess.run
        download_process = await asyncio.create_subprocess_exec(
            "wget", f"https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz",
            "-O", str(lang_fn),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await download_process.wait()
        
        extract_process = await asyncio.create_subprocess_exec(
            "tar", "zxf", str(lang_fn), "-C", str(config.MODEL_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await extract_process.wait()
        
        lang_fn.unlink()
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

    async def synthesize(self, text: str, lang: str = config.DEFAULT_LANGUAGE):
        """Synthesize speech from text"""
        if lang not in self.available_languages:
            await self._load_model(lang)
            self.available_languages.add(lang)
            
        model = self.models[lang]
        text = text.lower()
        stn_tst = model["text_mapper"].get_text(text, model["hps"])
        
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