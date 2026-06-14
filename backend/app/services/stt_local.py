import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import librosa
import io
import logging
import time

logger = logging.getLogger(__name__)

class QwenSTT:
    def __init__(self, model_id="Qwen/Qwen3-ASR-0.6B", device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Loading {model_id} on {self.device}...")
        start_time = time.time()
        
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            logger.info(f"Qwen3-ASR loaded in {time.time() - start_time:.2f}s")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to load Qwen3-ASR: {e}")
            self.available = False

    def transcribe(self, audio_bytes):
        if not self.available:
            return {"success": False, "error": "Model not loaded"}
            
        try:
            # Load audio using librosa (handles various formats and resamples to 16kHz)
            audio_file = io.BytesIO(audio_bytes)
            audio, _ = librosa.load(audio_file, sr=16000)
            
            result = self.pipe(audio)
            text = result.get("text", "").strip()
            
            return {
                "success": True, 
                "text": text,
                "language": "auto" # Qwen3-ASR handles language auto-detection
            }
        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            return {"success": False, "error": str(e)}

# Singleton instance
_qwen_stt = None

def get_qwen_stt():
    global _qwen_stt
    if _qwen_stt is None:
        _qwen_stt = QwenSTT()
    return _qwen_stt
