import logging
from typing import Any

import httpx

from .base_adapter import TelephonyAdapter

logger = logging.getLogger(__name__)


class ExotelAdapter(TelephonyAdapter):
    """
    Exotel Telephony Adapter.
    Currently a placeholder for future integration.
    """

    def __init__(self, api_key: str, api_token: str, sid: str):
        self.api_key = api_key
        self.api_token = api_token
        self.sid = sid
        self.base_url = f"https://api.exotel.com/v1/Accounts/{self.sid}"

    async def handle_incoming_call(self, payload: dict[str, Any]) -> str:
        """
        Processes an incoming Exotel webhook and returns the XML response.
        """
        call_sid = payload.get("CallSid")
        caller_number = payload.get("From", "Unknown")

        logger.info(f"☎️ Incoming Exotel call | Sid: {call_sid} | From: {caller_number}")

        # In a production scenario, we would trigger a LiveKit session here
        # For the demo, we return an XML that greets the user and passes the call to our bridge

        # The URL where Exotel should send the audio/receive instructions
        # This would typically be a SIP URI or an HTTP endpoint for Audio Streaming
        bridge_url = f"https://api.bcrec-ai.edu.in/voice/bridge/{call_sid}"

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Welcome to Dr. B.C. Roy Engineering College. Your call is being connected to our AI admissions assistant.</Say>
    <Passthru>{bridge_url}</Passthru>
</Response>"""
        return xml

    async def disconnect_call(self, call_sid: str) -> bool:
        """
        Uses Exotel API to hang up an active call.
        """
        if not self.api_key or not self.api_token:
            logger.error("Exotel credentials missing")
            return False

        try:
            async with httpx.AsyncClient() as client:
                auth = (self.api_key, self.api_token)
                url = f"{self.base_url}/Calls/{call_sid}"
                response = await client.post(url, auth=auth, data={"Status": "completed"})

                if response.status_code == 200:
                    logger.info(f"Successfully disconnected call: {call_sid}")
                    return True
                else:
                    logger.error(f"Failed to disconnect call: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Exotel disconnect error: {e}")
            return False

    def generate_passthru_xml(self, target_url: str) -> str:
        """
        Generates generic Passthru XML for Exotel.
        """
        return f'<?xml version="1.0" encoding="UTF-8"?><Response><Passthru>{target_url}</Passthru></Response>'
