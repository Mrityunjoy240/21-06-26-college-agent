import asyncio
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from typing import AsyncIterable, List, Dict, Any, Literal

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from dotenv import load_dotenv
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
    "JEE": "J. E. E."
}

BN_NUMS = {
    "0": "শূন্য", "1": "এক", "2": "দুই", "3": "তিন", "4": "চার", 
    "5": "পাঁচ", "6": "ছয়", "7": "সাত", "8": "আট", "9": "নয়"
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
            service=self._service
        )

class BCRECGroqStream(llm.LLMStream):
    def __init__(self, llm_inst, *, chat_ctx, tools, conn_options, service):
        super().__init__(llm=llm_inst, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._service = service
        self._id = utils.shortuuid()

    async def _run(self):
        history = []
        query = ""
        
        for msg in self.chat_ctx.messages():
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
        
        logger.info(f"LLM Query: '{query[:100]}...' History: {len(history)} turns")
        
        async for chunk in self._service.stream_response(query, conversation_history=history[:-1]):
            self._event_ch.send_nowait(llm.ChatChunk(
                id=self._id,
                delta=llm.ChoiceDelta(role="assistant", content=chunk)
            ))
        
        self._event_ch.close()

# ---------------------------------------------------------------------------
# SARVAM COMPONENTS (Molding for v1.5.x)
# ---------------------------------------------------------------------------
class SarvamSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False, diarization=False, offline_recognize=True
            )
        )
        self.service = get_sarvam_service(os.getenv("SARVAM_API_KEY"))

    async def _recognize_impl(
        self, buffer: utils.AudioBuffer, *, language: str | None = None, conn_options: llm.APIConnectOptions
    ) -> stt.SpeechEvent:
        try:
            frame = buffer if isinstance(buffer, rtc.AudioFrame) else utils.merge_frames(buffer)
            audio_data = frame.to_wav_bytes()
            result = await self.service.speech_to_text(audio_data, language="auto", model="saaras:v3")
            if result.get("success"):
                text = result.get("text", "")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text=text, confidence=0.95, language=result.get("language", "en-IN"))],
                )
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])
        except Exception as e:
            logger.error(f"Sarvam STT Error: {e}")
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

class SarvamTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=8000,
            num_channels=1
        )
        self.service = get_sarvam_service(os.getenv("SARVAM_API_KEY"))

    def synthesize(self, text: str, *, conn_options: llm.APIConnectOptions = agents.DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        logger.info(f"SarvamTTS.synthesize called for: {text[:50]}...")
        return SarvamChunkedStream(tts=self, input_text=text, conn_options=conn_options, service=self.service)

class SarvamChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts, input_text, conn_options, service):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self.service = service

    async def _run(self, emitter: tts.AudioEmitter):
        # Identify language first
        lang = "bn-IN" if re.search(r"[\u0980-\u09FF]", self._input_text) else "hi-IN" if re.search(r"[\u0900-\u097F]", self._input_text) else "en-IN"
        
        # Apply Lexicon and Number Processing
        clean_text = apply_lexicon(self._input_text, lang).strip()
        
        if not clean_text:
            emitter.end_input()
            return

        logger.info(f"Synthesizing stream chunk: {clean_text[:50]}... (lang: {lang})")
        res = await self.service.text_to_speech(clean_text, speaker="ritu" if lang == "bn-IN" else "shubh", language=lang)
        
        if res.get("success"):
            data = res["audio_bytes"]
            if data.startswith(b"RIFF"): data = data[44:]
            
            data = audioop.mul(data, 2, 1.0)
            data, _ = audioop.ratecv(data, 2, 1, 24000, 8000, None)
            
            emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=8000,
                num_channels=1,
                mime_type="audio/pcm"
            )
            
            chunk_size = 320 
            for j in range(0, len(data), chunk_size):
                chunk = data[j : j + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = chunk.ljust(chunk_size, b"\x00")
                emitter.push(chunk)
                
            logger.info("Finished pushing audio chunk")
        else:
            logger.error(f"Sarvam TTS failed: {res.get('error')}")
        
        emitter.end_input()

# ---------------------------------------------------------------------------
# GLOBAL PRE-WARMED SERVICES
# ---------------------------------------------------------------------------
# Initialize these globally so they are shared across jobs and pre-loaded
# when the worker process starts, NOT when the call arrives.
vad_inst = silero.VAD.load(min_speech_duration=0.3, min_silence_duration=1.0)
stt_comp = SarvamSTT()
llm_comp = BCRECGroqLLM()

from livekit.agents.tts import StreamAdapter
from livekit.agents.tokenize.basic import SentenceTokenizer
tts_comp = StreamAdapter(tts=SarvamTTS(), sentence_tokenizer=SentenceTokenizer())

# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT (The Worker Model)
# ---------------------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Starting agent job {ctx.job.id}")
    
    agent = voice.Agent(
        instructions="""You are a VOICE-FIRST AI assistant for Dr. B.C. Roy Engineering College (BCREC). 
        
        CRITICAL RULES FOR TELEPHONY:
        1. BE EXTREMELY CONCISE. Never speak more than 2-3 short sentences.
        2. NO TABLES OR LISTS. 
        3. ASK FOLLOW-UP QUESTIONS.
        4. USE PUNCTUATION.
        5. Never apologize for being brief.

        LANGUAGE & LINGUISTIC RULES:
        - USE ENGLISH LOANWORDS: In Bengali and Hindi, always use common English terms instead of formal translations.
        - EXAMPLES: Use 'ডিপার্টমেন্ট' (Department), 'এডমিশন' (Admission), 'প্রিন্সিপাল' (Principal), 'ফিস' (Fees).
        - NUMBERS & FEES: For all fees, amounts, and currency, write them out entirely in WORDS in the target language.
          * Example (Bengali): Instead of '3.5 lakh', write 'তিন লাখ পঞ্চাশ হাজার'. Instead of '35,000', write 'পঁয়ত্রিশ হাজার'.
          * Example (Hindi): Instead of '4.5 lakh', write 'साढ़े चार lakh'.
        - PHONE NUMBERS: Always write phone numbers as digits (e.g., 0343-2501353) so the system can read them digit-by-digit.
        - TONE: Sound like a friendly college staff member using everyday language.
        - Keep facts identical across languages. Answer only from the provided context.""",
        stt=stt_comp,
        tts=tts_comp,
        llm=llm_comp,
        vad=vad_inst,
        turn_handling={
            "interruption": {"enabled": True, "mode": "vad", "min_words": 2},
            "endpointing": {"min_delay": 0.5, "max_delay": 4.0}
        }
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
    session.say("Hello, BCREC AI assistant here. How can I help you?", allow_interruptions=False)

    while ctx.room.isconnected():
        await asyncio.sleep(1)
    
    logger.info("Room disconnected, exiting entrypoint")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="bcrec-agent"))
