from abc import ABC, abstractmethod
from typing import Any


class TelephonyAdapter(ABC):
    """
    Abstract base class for telephony providers (Exotel, Twilio, etc.)
    """

    @abstractmethod
    async def handle_incoming_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Process the initial incoming call webhook.
        """
        pass

    @abstractmethod
    async def disconnect_call(self, call_id: str) -> bool:
        """
        Terminate an active call.
        """
        pass
