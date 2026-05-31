import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DeepgramSTTService:
    """
    STT Service using Deepgram API.
    Provides high-performance transcription for real-time voice.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.url = "https://api.deepgram.com/v1/listen"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def transcribe(
        self, audio_bytes: bytes, content_type: str = "audio/wav"
    ) -> dict[str, Any]:
        """
        Transcribe audio using Deepgram REST API.
        For production telephony, we would typically use the WebSocket API.
        """
        if not self.api_key:
            return {"success": False, "error": "Deepgram API key not provided"}

        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": content_type}

        params = {
            "model": "nova-2",
            "smart_format": "true",
            "language": "en",  # default to english
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url, headers=headers, params=params, content=audio_bytes, timeout=30.0
                )

                if response.status_code != 200:
                    logger.error(f"Deepgram API error: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Deepgram returned {response.status_code}"}

                data = response.json()
                transcript = (
                    data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )

                return {
                    "success": True,
                    "text": transcript,
                    "confidence": data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("confidence", 0.0),
                }
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            return {"success": False, "error": str(e)}


# Global instance
_deepgram_service = None


def get_deepgram_service(api_key: str | None = None) -> DeepgramSTTService:
    global _deepgram_service
    if _deepgram_service is None:
        from app.config import settings

        _deepgram_service = DeepgramSTTService(api_key=api_key or settings.deepgram_api_key)
    return _deepgram_service
