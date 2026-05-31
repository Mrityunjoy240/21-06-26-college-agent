import logging
import time
from typing import Any

from app.config import settings
from app.services.llm.groq_service import get_groq_service
from app.services.sarvam_service import get_sarvam_service
from app.services.stt_deepgram import get_deepgram_service

logger = logging.getLogger(__name__)


class VoiceSessionManager:
    """
    Coordinates the real-time voice pipeline:
    Deepgram (STT) -> Groq (LLM/RAG) -> Sarvam (TTS)
    """

    def __init__(self):
        self.stt = get_deepgram_service()
        self.llm = get_groq_service()
        self.tts = get_sarvam_service(settings.sarvam_api_key)

    async def process_audio_chunk(
        self, audio_bytes: bytes, conversation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Processes a single audio chunk and returns the full pipeline result.
        """
        start_total = time.time()

        # 1. Transcribe (STT)
        stt_start = time.time()
        stt_result = await self.stt.transcribe(audio_bytes)
        stt_time = time.time() - stt_start

        if not stt_result.get("success") or not stt_result.get("text"):
            return {
                "success": False,
                "error": stt_result.get("error", "No transcript generated"),
                "stage": "stt",
            }

        user_text = stt_result["text"]
        logger.info(f"Pipeline [STT]: {user_text} ({stt_time:.2f}s)")

        # 2. Generate Answer (LLM/RAG)
        llm_start = time.time()
        # Note: GroqService already handles the RAG knowledge injection
        llm_result = await self.llm.generate_response(user_text, conversation_history=[])
        llm_time = time.time() - llm_start

        answer_text = llm_result.get("answer", "")
        logger.info(f"Pipeline [LLM]: {answer_text[:50]}... ({llm_time:.2f}s)")

        # 3. Convert to Speech (TTS)
        tts_start = time.time()
        tts_result = await self.tts.text_to_speech(answer_text)
        tts_time = time.time() - tts_start

        if not tts_result.get("success"):
            return {
                "success": False,
                "error": tts_result.get("error", "TTS failed"),
                "stage": "tts",
                "transcript": user_text,
                "answer": answer_text,
            }

        total_time = time.time() - start_total
        logger.info(f"Pipeline Complete: Total Time {total_time:.2f}s")

        return {
            "success": True,
            "transcript": user_text,
            "answer": answer_text,
            "audio_bytes": tts_result["audio_bytes"],
            "metrics": {
                "stt_time": stt_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "total_time": total_time,
            },
        }


# Singleton
_session_manager = None


def get_voice_session_manager() -> VoiceSessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = VoiceSessionManager()
    return _session_manager
