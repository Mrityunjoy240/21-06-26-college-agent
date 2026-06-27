import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from typing import AsyncIterable, List, Dict, Any, Literal

# Force HuggingFace to load from local disk cache — skips all network HEAD/GET
# requests that were causing prewarm to timeout. Model was already downloaded.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Fix Windows cp1252 crash: the rupee symbol (₹) and Bengali chars in LLM
# responses crash the Rich console logger. Force UTF-8 everywhere so child
# processes spawned by LiveKit IPC inherit the correct encoding.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from dotenv import load_dotenv

_dotenv_path = os.path.join(os.path.dirname(__file__), "..", "backend", ".env")
load_dotenv(_dotenv_path, override=True)

from livekit import agents, api, rtc
from livekit.agents import (
    WorkerOptions,
    cli,
    stt,
    tts,
    llm,
    utils,
    voice,
    vad,
)
from livekit.plugins import silero

from app.database import get_db, init_db
from app.services.llm.groq_service import get_groq_service, SYSTEM_PROMPT
from app.services.sarvam_service import get_sarvam_service

# Initialize DB
init_db()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("livekit-agent")

# ---------------------------------------------------------------------------
# PHONETIC LEXICON & NUMBER CONVERSION
# ---------------------------------------------------------------------------
LEXICON = {
    "BCREC": "BCREC",
    "MAKAUT": "Ma-Kaut",
    "AIML": "A. I. M. L.",
    "AML": "A. I. M. L.",
    "CSE": "CSE",
    "ECE": "E. C. E.",
    "IT": "Information Technology",
    "B.Tech": "B. Tech",
    "M.Tech": "M. Tech",
    "WBJEE": "W. B. J. E. E.",
    "JEE": "J. E. E.",
}

BN_NUMS = {
    "0": "শূন্য",
    "1": "এক",
    "2": "দুই",
    "3": "তিন",
    "4": "চার",
    "5": "পাঁচ",
    "6": "ছয়",
    "7": "সাত",
    "8": "আট",
    "9": "নয়",
}


def convert_phone_numbers(text: str) -> str:
    """Only converts phone-number-like digit sequences to Bengali words."""
    processed = text

    # Identify phone numbers (7+ digits) and convert them digit-by-digit
    def digit_replacer(match):
        digits = match.group()
        return " ".join([BN_NUMS.get(d, d) for d in digits])

    # Convert sequences of 7 to 11 digits (phone numbers)
    processed = re.sub(r"\d{7,11}", digit_replacer, processed)

    # Also handle specific college numbers with dashes/spaces
    processed = re.sub(r"\d{4}[-\s]\d{7}", digit_replacer, processed)

    return processed


def apply_lexicon(text: str, lang: str) -> str:
    """Apply permanent pronunciation rules."""
    processed = text

    # 1. Expand technical acronyms based on LEXICON
    for word, phonetic in LEXICON.items():
        processed = re.sub(rf"\b{re.escape(word)}\b", phonetic, processed)

    # 2. Normalize AML to AIML for consistency
    processed = re.sub(r"\bAML\b", "A. I. M. L.", processed, flags=re.IGNORECASE)

    # 3. Handle numbers for Bengali
    if lang == "bn-IN":
        # Only convert phone numbers digit-by-digit
        # Fees/Currency are now handled by the LLM in words
        processed = convert_phone_numbers(processed)

    return processed


# ---------------------------------------------------------------------------
# NATIVE LLM WRAPPER (Molding Groq for LiveKit v1.5.x)
# ---------------------------------------------------------------------------
class BCRECGroqLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self._service = get_groq_service()

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: List[llm.Tool] | None = None,
        conn_options: llm.APIConnectOptions = agents.DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: agents.NotGivenOr[bool] = agents.NOT_GIVEN,
        tool_choice: agents.NotGivenOr[llm.ToolChoice] = agents.NOT_GIVEN,
        extra_kwargs: agents.NotGivenOr[Dict[str, Any]] = agents.NOT_GIVEN,
    ) -> llm.LLMStream:
        return BCRECGroqStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            service=self._service,
        )


class BCRECGroqStream(llm.LLMStream):
    def __init__(self, llm_inst, *, chat_ctx, tools, conn_options, service):
        super().__init__(llm=llm_inst, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._service = service
        self._id = utils.shortuuid()

    async def _run(self):
        t0 = time.time()
        query = ""

        # Collect user + assistant messages only (skip system — _build_messages adds SYSTEM_PROMPT)
        history = []
        for msg in self.chat_ctx.messages():
            if msg.role == "system":
                continue
            content_text = ""
            if isinstance(msg.content, str):
                content_text = msg.content
            elif isinstance(msg.content, list):
                parts = []
                for c in msg.content:
                    if isinstance(c, str):
                        parts.append(c)
                    elif hasattr(c, "text"):
                        parts.append(c.text)
                content_text = " ".join(parts)

            if msg.role == "user":
                query = content_text

            if content_text:
                history.append({"role": msg.role, "content": content_text})

        # Cap at last 12 turns (6 user + 6 assistant) to prevent unbounded prompt growth
        if len(history) > 12:
            history = history[-12:]

        logger.info(f"LLM Query: '{query[:100]}...' History: {len(history)} turns")

        first_chunk = True
        async for chunk in self._service.stream_response(
            query, session_id="livekit", conversation_history=history[:-1]
        ):
            if first_chunk:
                ttft = round((time.time() - t0) * 1000)
                logger.info(f"TURN TTFT={ttft}ms (user speech → LLM first token)")
                first_chunk = False
            self._event_ch.send_nowait(
                llm.ChatChunk(id=self._id, delta=llm.ChoiceDelta(role="assistant", content=chunk))
            )

        turn_total = round((time.time() - t0) * 1000)
        logger.info(f"TURN COMPLETE total={turn_total}ms (user speech → LLM done)")
        self._event_ch.close()


# ---------------------------------------------------------------------------
# SARVAM COMPONENTS (Molding for v1.5.x)
# ---------------------------------------------------------------------------
# BCREC acronyms Sarvam STT frequently mis-transcribes
_STT_ACRONYM_FIXES = [
    (r"\bcse[\s-]?aml\b", "CSE-AIML"),
    (r"\bcs e[\s-]?aml\b", "CSE-AIML"),
    (r"\bcciml\b", "AIML"),
    (r"\ba[\s-]?i[\s-]?ml\b", "AIML"),
    (r"\bcsd\b", "CSD"),
    (r"\bdata sci\b", "Data Science"),
    (r"\bcyber sec\b", "Cyber Security"),
    (r"\binfo tech\b", "Information Technology"),
    (r"\belec[ -]?comm\b", "ECE"),
    (r"\bh[\s-]?o[\s-]?d\b", "HOD"),
    (r"\bprincipal\b", "Principal"),
]


def _fix_stt_acronyms(text: str) -> str:
    """Normalize Sarvam STT output before passing to LLM."""
    result = text
    for pattern, replacement in _STT_ACRONYM_FIXES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    if result != text:
        logger.info(f"STT corrected: '{text}' -> '{result}'")
    return result


class SarvamSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False, diarization=False, offline_recognize=True
            )
        )
        self.service = get_sarvam_service(os.getenv("SARVAM_API_KEY"))

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: llm.APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            frame = buffer if isinstance(buffer, rtc.AudioFrame) else utils.merge_frames(buffer)
            audio_data = frame.to_wav_bytes()
            result = await self.service.speech_to_text(
                audio_data, language="auto", model="saaras:v3"
            )
            if result.get("success"):
                text = result.get("text", "")
                text = _fix_stt_acronyms(text)
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
            logger.error(f"Sarvam STT Error: {e}")
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])


class SarvamTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.service = get_sarvam_service(os.getenv("SARVAM_API_KEY"))

    def synthesize(
        self, text: str, *, conn_options: llm.APIConnectOptions = agents.DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        logger.info(f"SarvamTTS.synthesize called for: {text[:50]}...")
        return SarvamChunkedStream(
            tts=self, input_text=text, conn_options=conn_options, service=self.service
        )


class SarvamChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts, input_text, conn_options, service):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self.service = service

    async def _run(self, emitter: tts.AudioEmitter):
        text = self._input_text
        if not text or not text.strip():
            emitter.end_input()
            return

        # Identify language
        lang = (
            "bn-IN"
            if re.search(r"[\u0980-\u09FF]", text)
            else "hi-IN"
            if re.search(r"[\u0900-\u097F]", text)
            else "en-IN"
        )

        # Pick speaker from shared map
        from app.utils.voice_utils import LANG_SPEAKER_MAP

        speaker = LANG_SPEAKER_MAP.get(lang, "shubh")

        logger.info(f"Synthesizing: {text[:60]}... (lang={lang}, speaker={speaker})")
        res = await self.service.text_to_speech(text.strip(), speaker=speaker, language=lang)

        if res.get("success"):
            data = res["audio_bytes"]

            # Find the actual audio data chunk in WAV (don't hardcode 44-byte header)
            if data.startswith(b"RIFF"):
                # Skip "RIFF" + size + "WAVE" = 12 bytes, then iterate chunks
                i = 12
                while i < len(data) - 8:
                    chunk_id = data[i : i + 4]
                    chunk_size = int.from_bytes(data[i + 4 : i + 8], "little")
                    if chunk_id == b"data":
                        data = data[i + 8 : i + 8 + chunk_size]
                        break
                    i += 8 + chunk_size
                else:
                    # Fallback: strip first 44 bytes (standard PCM header)
                    data = data[44:]

            emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=24000,
                num_channels=1,
                mime_type="audio/pcm",
            )

            # Push in 100ms chunks (4800 bytes @ 24000Hz 16-bit mono)
            chunk_size = 4800
            for j in range(0, len(data), chunk_size):
                chunk = data[j : j + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = chunk.ljust(chunk_size, b"\x00")
                emitter.push(chunk)

            logger.info("Finished pushing audio chunk")
        else:
            logger.error(f"Sarvam TTS failed: {res.get('error')}")

        emitter.end_input()


from livekit.agents.tts import StreamAdapter
from livekit.agents.tokenize.basic import SentenceTokenizer


# ---------------------------------------------------------------------------
# PREWARM — runs ONCE per worker process, not in every subprocess fork.
# This is the correct LiveKit pattern to avoid re-loading BGE-M3 repeatedly.
# ---------------------------------------------------------------------------
def prewarm(proc: agents.JobProcess):
    logger.info("Prewarming agent components (VAD, STT, LLM, TTS)...")
    proc.userdata["vad"] = silero.VAD.load(min_speech_duration=0.3, min_silence_duration=1.0)
    proc.userdata["stt"] = SarvamSTT()
    proc.userdata["llm"] = BCRECGroqLLM()
    proc.userdata["tts"] = StreamAdapter(tts=SarvamTTS(), sentence_tokenizer=SentenceTokenizer())
    logger.info("Prewarm complete — agent is ready to accept jobs.")


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT (The Worker Model)
# ---------------------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Starting agent job {ctx.job.id}")

    # Pull pre-warmed components from process userdata (loaded once by prewarm)
    vad_inst = ctx.proc.userdata["vad"]
    stt_comp = ctx.proc.userdata["stt"]
    llm_comp = ctx.proc.userdata["llm"]
    tts_comp = ctx.proc.userdata["tts"]

    # Single source of truth: SYSTEM_PROMPT from groq_service.py
    # Voice-specific additions layered on top for telephony optimizations.
    INSTRUCTIONS = (
        SYSTEM_PROMPT
        + """

VOICE TELEPHONY RULES (ADDITIONAL):
- Be concise for voice. 2-4 short sentences is fine.
- Use common English loanwords in Bengali (ডিপার্টমেন্ট, এডমিশন, ফিস).
- Phone numbers stay as digits (0343-2501353) for digit-by-digit TTS."""
    )

    agent = voice.Agent(
        instructions=INSTRUCTIONS,
        stt=stt_comp,
        tts=tts_comp,
        llm=llm_comp,
        vad=vad_inst,
        turn_handling={
            "interruption": {"enabled": True, "mode": "vad", "min_words": 5},
            "endpointing": {"min_delay": 0.5, "max_delay": 4.0},
        },
    )

    session = voice.AgentSession(
        stt=stt_comp,
        tts=tts_comp,
        llm=llm_comp,
        vad=vad_inst,
    )

    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")

    await session.start(agent, room=ctx.room)
    logger.info("Agent session started")

    await asyncio.sleep(0.5)
    logger.info("Sending greeting...")
    session.say(
        "Hello! BCREC AI assistant here. How can I help you today?", allow_interruptions=False
    )

    while ctx.room.isconnected():
        await asyncio.sleep(1)

    get_groq_service().clear_session("livekit")
    logger.info("Room disconnected, session cleared, exiting entrypoint")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="bcrec-agent",
            initialize_process_timeout=120.0,  # BGE-M3 needs ~30s to load from cache
        )
    )
