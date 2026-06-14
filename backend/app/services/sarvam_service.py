"""
Sarvam.ai Service for BCREC Voice Agent

Provides Text-to-Speech (TTS) and Speech-to-Text (STT) using Sarvam AI APIs.
- TTS: Bulbul v3 model with 30+ Indian voices
- STT: Saaras v3 model for 22 Indian languages
"""
import base64
import logging
import io
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from sarvamai import SarvamAI
    SARVAM_AVAILABLE = True
except ImportError:
    SARVAM_AVAILABLE = False
    logger.warning("sarvamai not installed. Sarvam service will be unavailable.")


class SarvamService:
    """
    Sarvam AI service for TTS and STT operations.
    """
    
    # Supported languages
    SUPPORTED_LANGUAGES = {
        "en-IN": "English (India)",
        "hi-IN": "Hindi",
        "hi": "Hindi",
        "bn-IN": "Bengali",
        "bn": "Bengali",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "mr-IN": "Marathi",
        "gu-IN": "Gujarati",
        "pa-IN": "Punjabi",
        "od-IN": "Odia",
    }
    
    # Available TTS voices (Bulbul v3)
    VOICES = {
        "aditya": "Aditya (Male)",
        "ritu": "Ritu (Female)",
        "ashutosh": "Ashutosh",
        "priya": "Priya",
        "neha": "Neha",
        "rahul": "Rahul",
        "pooja": "Pooja",
        "rohan": "Rohan",
        "simran": "Simran",
        "kavya": "Kavya",
        "amit": "Amit",
        "dev": "Dev",
        "ishita": "Ishita",
        "shreya": "Shreya",
        "ratan": "Ratan",
        "varun": "Varun",
        "manan": "Manan",
        "sumit": "Sumit",
        "roopa": "Roopa",
        "kabir": "Kabir",
        "aayan": "Aayan",
        "shubh": "Shubh",
        "advait": "Advait",
        "anand": "Anand",
        "tanya": "Tanya",
        "tarun": "Tarun",
        "sunny": "Sunny",
        "mani": "Mani",
        "gokul": "Gokul",
        "vijay": "Vijay",
        "shruti": "Shruti",
        "suhani": "Suhani",
        "mohit": "Mohit",
        "kavitha": "Kavitha",
        "rehan": "Rehan",
        "soham": "Soham",
        "rupali": "Rupali",
        "niharika": "Niharika"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        self.api_key = api_key
        
        if SARVAM_AVAILABLE and api_key:
            try:
                self.client = SarvamAI(api_subscription_key=api_key)
                logger.info("Sarvam client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Sarvam client: {e}")
                self.client = None
        else:
            if not api_key:
                logger.warning("Sarvam API key not provided")
            if not SARVAM_AVAILABLE:
                logger.warning("Sarvam library not installed")
    
    def is_available(self) -> bool:
        """Check if Sarvam service is available"""
        return self.client is not None
    
    def _normalize_text_for_tts(self, text: str) -> str:
        """Normalize text to prevent spelling out and awkward pauses in TTS"""
        import re
        
        # Replace abbreviations and initialisms
        replacements = {
            r"\bRs\b\.?": "Rupees",
            r"\bRs\b": "Rupees",
            r"\bDr\b\.?": "Doctor",
            r"\bProf\b\.?": "Professor",
            r"\bB\.?Tech\b\.?": "B Tech",
            r"\bM\.?Tech\b\.?": "M Tech",
            r"\bMBA\b": "M B A",
            r"\bMCA\b": "M C A",
            r"\bCSE\b": "C S E",
            r"\bIT\b": "I T",
            r"\bECE\b": "E C E",
            r"\bEE\b": "E E",
            r"\bME\b": "M E",
            r"\bCE\b": "C E",
            r"\bMAKAUT\b": "M A K A U T",
            r"\bNAAC\b": "N A A C",
            r"\bNBA\b": "N B A",
            r"\bWBJEE\b": "W B J E E",
            r"\bJEE\b": "J E E",
            r"\bLPA\b": "Lakhs per annum",
            r"\bB\.?C\.?\b": "B C",
            r"\bTCS\b": "T C S",
            r"\bTPO\b": "T P O",
            r"\bHOD\b": "H O D",
        }
        
        normalized = text
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            
        # Replace slashes in units
        normalized = re.sub(r"/month\b", " per month", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"/year\b", " per year", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"/sem\b", " per semester", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"/", " or ", normalized)
        
        # Format phone numbers (remove hyphens between digits)
        normalized = re.sub(r"(\d+)-(\d+)", r"\1 \2", normalized)
        
        # Remove period after single letters (like middle initials: Sanjay S. Pawar -> Sanjay S Pawar)
        normalized = re.sub(r"\b([A-Z])\.", r"\1", normalized)
        
        # Clean double spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized

    async def text_to_speech(
        self,
        text: str,
        language: str = "en-IN",
        speaker: str = "shubh",
        pace: float = 1.0,
        model: str = "bulbul:v3"
    ) -> Dict[str, Any]:
        """
        Convert text to speech using Sarvam TTS API.
        """
        if not self.client:
            return {
                "success": False,
                "error": "Sarvam client not initialized"
            }
        
        try:
            # Normalize text for perfect pronunciation
            normalized_text = self._normalize_text_for_tts(text)
            logger.info(f"Normalized TTS text: '{normalized_text}'")
            
            response = self.client.text_to_speech.convert(
                text=normalized_text,
                target_language_code=language,
                speaker=speaker,
                model=model,
                pace=pace,
                speech_sample_rate=24000
            )
            
            # Combine all audio chunks
            audio_base64 = "".join(response.audios)
            audio_bytes = base64.b64decode(audio_base64)
            
            logger.info(f"TTS generated: {len(audio_bytes)} bytes, lang={language}, speaker={speaker}")
            
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "format": "wav",
                "language": language,
                "speaker": speaker
            }
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def speech_to_text(
        self,
        audio_bytes: bytes,
        language: str = "en-IN",
        model: str = "saaras:v3"
    ) -> Dict[str, Any]:
        """
        Convert speech to text using Sarvam STT API.
        
        Args:
            audio_bytes: Audio data (wav/mp3 format)
            language: Language code (default: en-IN, use 'auto' for detection)
            model: STT model (default: saaras:v3)
        
        Returns:
            Dict with 'text', 'language', and 'success'
        """
        if not self.client:
            return {
                "success": False,
                "error": "Sarvam client not initialized"
            }
        
        try:
            # Create file-like object for upload
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            lang_param = None if language == "auto" else language
            
            logger.info(f"STT request: audio_size={len(audio_bytes)} bytes, model={model}, language={lang_param}")
            
            response = self.client.speech_to_text.transcribe(
                file=audio_file,
                model=model,
                language_code=lang_param
            )
            
            logger.info(f"Sarvam raw response type: {type(response)}")
            logger.info(f"Sarvam raw response: {response}")
            
            transcript = ""
            detected_language = language
            
            if hasattr(response, 'transcript'):
                transcript = response.transcript or ""
            elif hasattr(response, 'text'):
                transcript = response.text or ""
            elif isinstance(response, dict):
                transcript = response.get('transcript', '') or response.get('text', '') or ""
            else:
                transcript = str(response) or ""
            
            if hasattr(response, 'language_code') and response.language_code:
                detected_language = response.language_code
            
            logger.info(f"Extracted transcript: '{transcript[:100] if transcript else 'EMPTY'}'")
            
            logger.info(f"STT result: '{transcript[:100]}...' detected_lang={detected_language}")
            
            return {
                "success": True,
                "text": transcript,
                "language": detected_language
            }
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get list of available TTS voices"""
        return self.VOICES.copy()
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()


# Global instance
_sarvam_service: Optional[SarvamService] = None


def get_sarvam_service(api_key: Optional[str] = None) -> SarvamService:
    """Get or create the Sarvam service singleton"""
    global _sarvam_service
    
    if _sarvam_service is None:
        _sarvam_service = SarvamService(api_key=api_key)
    return _sarvam_service


def init_sarvam_service(api_key: str) -> SarvamService:
    """Initialize Sarvam service with API key"""
    global _sarvam_service
    _sarvam_service = SarvamService(api_key=api_key)
    return _sarvam_service
