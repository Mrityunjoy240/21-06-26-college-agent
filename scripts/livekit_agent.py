import asyncio
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from dotenv import load_dotenv
from livekit import agents, api, rtc
from livekit.agents import (
    WorkerOptions,
    cli,
    stt,
    tts,
    utils,
)
from livekit.plugins import openai, silero

from app.database import get_db, init_db
from app.services.llm.groq_service import get_groq_service
from app.services.sarvam_service import get_sarvam_service

load_dotenv("backend/.env", override=True)

# Initialize DB
init_db()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("livekit-agent")


def detect_language(text: str) -> str:
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn-IN"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"
    return "en-IN"


# DATABASE MEMORY HELPERS
def load_history(phone_number: str):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM conversations WHERE phone_number = ? ORDER BY updated_at DESC LIMIT 1",
            (phone_number,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None, []
        conv_id = row["id"]
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        )
        messages = [{"role": m["role"], "content": m["content"]} for m in cursor.fetchall()]
        conn.close()
        return conv_id, messages
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return None, []


def save_msg(conv_id: str, phone_number: str, role: str, content: str):
    try:
        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        if not conv_id:
            conv_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO conversations (id, phone_number, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, phone_number, now, now),
            )
        else:
            cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        cursor.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conv_id, role, content, now),
        )
        conn.commit()
        conn.close()
        return conv_id
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return conv_id


class SarvamSTT(stt.STT):
    def __init__(self, api_key: str):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False, offline_recognize=True
            )
        )
        self.service = get_sarvam_service(api_key)

    async def _recognize_impl(
        self, buffer: rtc.AudioFrame | utils.AudioBuffer, *, language: str | None = None, **kwargs
    ) -> stt.SpeechEvent:
        try:
            frame = buffer if isinstance(buffer, rtc.AudioFrame) else utils.merge_frames(buffer)
            audio_data = frame.to_wav_bytes()
            result = await self.service.speech_to_text(
                audio_data, language="auto", model="saaras:v3"
            )
            if result.get("success"):
                text = result.get("text", "")
                logger.info(f"STT: {text}")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=text, confidence=0.95, language=result.get("language", "en-IN")
                        )
                    ],
                )
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])


try:
    import audioop
except ImportError:
    import audioop_lts as audioop


class SarvamTTS(tts.TTS):
    def __init__(self, api_key: str):
        # 8kHz for Telephony Compatibility
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=8000, num_channels=1
        )
        self.service = get_sarvam_service(api_key)

    def synthesize(
        self, text: str, *, conn_options: agents.utils.http_context.APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        tts_self = self

        class Stream(tts.ChunkedStream):
            def __init__(self):
                super().__init__(
                    tts=tts_self,
                    input_text=text,
                    conn_options=conn_options
                    or agents.utils.http_context.DEFAULT_API_CONNECT_OPTIONS,
                )

            async def _run(self, emitter: tts.AudioEmitter):
                lang = detect_language(text)
                # Request audio from Sarvam (which returns 24kHz)
                res = await tts_self.service.text_to_speech(
                    text, speaker="ritu" if lang == "bn-IN" else "shubh", language=lang
                )
                if res.get("success"):
                    emitter.initialize(
                        request_id=utils.shortuuid(),
                        sample_rate=8000,
                        num_channels=1,
                        mime_type="audio/pcm",
                    )
                    data = res["audio_bytes"]
                    if data.startswith(b"RIFF"):
                        data = data[44:] # Skip WAV header
                    
                    # Reduce volume (gain) to prevent clipping noise
                    # 0.7 multiplier reduces volume by 30% for a cleaner sound
                    try:
                        data = audioop.mul(data, 2, 0.7) 
                    except Exception as e:
                        logger.error(f"Failed to adjust audio gain: {e}")

                    # Resample 24000 Hz to 8000 Hz
                    try:
                        data, _ = audioop.ratecv(data, 2, 1, 24000, 8000, None)
                        
                        # CHUNKING: Break into 20ms packets (320 bytes for 8kHz 16-bit mono)
                        # This prevents jitter and makes the audio smooth for telephony.
                        chunk_size = 320 
                        for i in range(0, len(data), chunk_size):
                            chunk = data[i : i + chunk_size]
                            if len(chunk) < chunk_size:
                                chunk = chunk.ljust(chunk_size, b"\x00") # Padding
                            emitter.push(chunk)
                            await asyncio.sleep(0.015) # Steady stream to avoid bursting
                            
                    except Exception as e:
                        logger.error(f"Failed to resample or chunk audio: {e}")
                    
                    emitter.flush()

        return Stream()


async def run_directly(room_name: str):
    logger.info(f"Starting persistent direct agent for room: {room_name}")
    
    # Load components once outside the loop to conserve resources and avoid reloading
    logger.info("Loading VAD, STT, and TTS components...")
    vad = silero.VAD.load(min_speech_duration=0.25, min_silence_duration=0.8)
    stt_comp = SarvamSTT(os.getenv("SARVAM_API_KEY"))
    tts_comp = SarvamTTS(os.getenv("SARVAM_API_KEY"))
    
    # KB
    groq_internal = get_groq_service()
    logger.info(f"Loaded SARVAM_API_KEY: {os.getenv('SARVAM_API_KEY')[:8]}...")
    
    while True:
        logger.info("Ready for next call. Waiting to connect...")
        token = (
            api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
            .with_identity("bcrec-agent-direct")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        room = rtc.Room()
        session = agents.AgentSession(stt=stt_comp, tts=tts_comp, vad=vad)

        current_phone = "unknown"
        current_conv_id = None
        history = []

        @session.on("user_input_transcribed")
        def on_transcript(ev):
            async def process():
                nonlocal current_conv_id, history
                try:
                    user_text = ev.transcript.strip()
                    if not user_text:
                        return

                    logger.info(f"User ({current_phone}): {user_text}")
                    current_conv_id = save_msg(current_conv_id, current_phone, "user", user_text)
                    history.append({"role": "user", "content": user_text})

                    # Direct Brain Logic
                    resp = await groq_internal.generate_response(user_text, conversation_history=history)
                    answer = resp.get("answer", "I'm sorry, I missed that.")
                    
                    logger.info(f"Assistant: {answer}")
                    session.say(answer, allow_interruptions=False)  # Set to False to prevent line static/noise from canceling speech
                    
                    current_conv_id = save_msg(current_conv_id, current_phone, "assistant", answer)
                    history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    logger.exception(f"Error in processing agent logic: {e}")

            asyncio.create_task(process())

        from livekit.agents import voice
        dummy_agent = voice.Agent(instructions="Welcome to Dr. B.C. Roy Engineering College.", stt=stt_comp, tts=tts_comp, vad=vad)

        try:
            await room.connect(os.getenv("LIVEKIT_URL"), token)
            await session.start(agent=dummy_agent, room=room)

            # Wait for participant metadata
            try:
                timeout = 2.0  # Reduced from 5.0 for faster startup
                start_time = asyncio.get_event_loop().time()
                while len(room.remote_participants) == 0 and (asyncio.get_event_loop().time() - start_time) < timeout:
                    await asyncio.sleep(0.05)

                if len(room.remote_participants) > 0:
                    participant = list(room.remote_participants.values())[0]
                    meta = json.loads(participant.metadata or "{}")
                    current_phone = meta.get("phone_number", "unknown")
                    logger.info(f"Identified Caller: {current_phone}")
                else:
                    current_phone = "unknown"
            except Exception:
                current_phone = "unknown"

            # LOAD MEMORY
            current_conv_id, history = load_history(current_phone)
            greeting = "Hello! This is the B C Roy Engineering College AI. How can I help you today?"

            logger.info(f"Starting greeting: {greeting}")
            await asyncio.sleep(0.2)  # Reduced delay for snappy feel
            session.say(greeting, allow_interruptions=False)
            current_conv_id = save_msg(current_conv_id, current_phone, "assistant", greeting)

            # Keep connection alive until participant disconnects
            while room.isconnected():
                await asyncio.sleep(1)
        except Exception as loop_err:
            logger.error(f"Error in active call session: {loop_err}")
        finally:
            logger.info("Call disconnected or session ended. Cleaning up room connection...")
            if room.isconnected():
                await room.disconnect()
            await asyncio.sleep(2)  # Wait briefly before listening for the next call



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "dev", "direct"])
    parser.add_argument("--room", default="bcrec-voice-call")
    args = parser.parse_args()

    if args.command == "direct":
        asyncio.run(run_directly(args.room))
    else:
        # Standard worker for non-telephony
        cli.run_app(WorkerOptions(entrypoint_fnc=None, agent_name="bcrec-agent"))
