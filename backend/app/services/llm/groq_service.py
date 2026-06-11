"""
Groq LLM Service - Clean RAG Architecture
==========================================
Flow: User Query -> Vector Search (top 4 chunks) -> LLM with context -> Answer

No more full KB injection. No conflicting rules. Just simple, clean RAG.
"""
import logging
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from app.config import settings
from app.services.vector_store import get_vector_store
from app.utils.language_detect import detect_language


# ---------------------------------------------------------------------------
# System Prompt — Voice-First Telephony Optimization
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI admission assistant for Dr. B.C. Roy Engineering College (BCREC), Durgapur.

CRITICAL ROLE CLARITY:
- THE PRINCIPAL (Head of College) is Dr. Sanjay S. Pawar.
- THE VICE PRINCIPAL is Prof. (Dr.) K. M. Hossain.
- If asked "Who is the principal?", always answer: Dr. Sanjay S. Pawar.

CRITICAL RULES FOR VOICE INTERACTION (SPEED IS PARAMOUNT):
- BE EXTREMELY CONCISE. Your response must be 1 to 2 short sentences MAX.
- NEVER PROVIDE TABLES, BULLET POINTS, OR NUMBERED LISTS. Use conversational paragraphs only.
- NEVER use filler phrases like "Based on the context", "I have been given information", or "The answer is". Just state the facts immediately.
- End sentences with a period (.) to allow the voice to start speaking immediately.

LANGUAGE & TRANSLATION RULES (FACTUAL CONSISTENCY IS MANDATORY):
- The facts you provide MUST remain exactly the same regardless of the language you are speaking in. Do not invent different facts for Hindi vs English.
- If the user writes in Bengali script → translate the facts to Bengali. Use common English loanwords phonetically. Keep it very short.
- If the user writes in Hindi script → translate the facts to Hindi. Keep it very short.
- If the user writes in English or Romanized (Banglish/Hinglish) → reply in the SAME style they used.
- NEVER switch languages mid-response.

PRONUNCIATION & CURRENCY RULES (TTS OPTIMIZATION):
- NEVER use digits (0-9) for fees or amounts.
- ABSOLUTELY FORBIDDEN: "5.98", "6.1", "₹4.5", "Rs. 5,000".
- YOU MUST write the entire number out in words.
- HINDI: Use words like "पाँच लाख" (panch lakh), "छह लाख" (chheh lakh), "चार हजार" (chaar hazaar).
- BENGALI: Use words like "পাঁচ লাখ" (panch lakh), "ছয় লাখ" (chhoy lakh), "দশ হাজার" (dosh hazaar).
- Example Hindi: Instead of "₹4.49 लाख", write "लगभग साढ़े चार लाख रुपये".
- Example Bengali: Instead of "Rs. 5,98,300", write "প্রায় ছয় লাখ টাকা".
- Example English: Instead of "Rs. 5,98,300", write "approximately six lakh rupees".
- This is the ONLY way the TTS will pronounce it correctly.

CONTENT RULES:
- Answer ONLY from the CONTEXT provided below. Do not make up facts.
- If asked about "Departments" or "Admission", list ONLY the main degree categories (B.Tech, MCA, MBA, M.Tech) or the top 3 B.Tech branches. NEVER list subjects like Math, Physics, or Biology as departments.
- If the context does not contain the answer, say so politely and give the college phone: 0343-2501353."""


class GroqService:
    """
    Clean Hybrid RAG service: JSON (Precision) + Vector Store (Context).
    """

    def __init__(self):
        self.model = "llama-3.1-8b-instant"
        self.max_tokens = 1024
        self.client = settings.groq_client
        self.async_client = getattr(settings, 'async_groq_client', None)
        self.vector_store = get_vector_store()
        
        # Load high-precision core knowledge once into memory
        try:
            # backend/app/services/llm/groq_service.py -> .parent x4 -> backend/
            kb_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge_base" / "combined_kb.json"
            with open(kb_path, "r", encoding="utf-8") as f:
                self.core_kb = json.load(f)
            logger.info(f"Core Knowledge (JSON) loaded successfully from {kb_path}")
        except Exception as e:
            logger.error(f"Failed to load combined_kb.json: {e}")
            self.core_kb = {}

        if self.client:
            logger.info("GroqService ready.")
        else:
            logger.warning("GroqService: Groq client not found. Check GROQ_API_KEY in .env")

    def _get_precision_context(self, query: str) -> str:
        """Extract relevant parts of the JSON based on keywords (Fast & Accurate)."""
        q = query.lower()
        context_parts = []
        
        # Topic Mapping (Keywords -> JSON Keys)
        mapping = {
            "fees": ["fees_summary", "courses", "scholarships"],
            "hostel": ["hostel"],
            "admission": ["admission", "admission_documents", "branch_change"],
            "placement": ["placements"],
            "contact": ["college", "departments"],
            "principal": ["principal", "vice_principal"],
            "infrastructure": ["infrastructure", "student_life"],
            "courses": ["courses"]
        }
        
        found_keys = set()
        for topic, keys in mapping.items():
            if topic in q:
                found_keys.update(keys)
        
        # If no specific topic found, give general college info
        if not found_keys:
            found_keys.update(["college", "quick_answers"])
            
        for key in found_keys:
            if key in self.core_kb:
                context_parts.append(f"## {key.upper()}\n{json.dumps(self.core_kb[key], indent=1)}")
                
        return "\n\n".join(context_parts)

    def _retrieve_context(self, query: str) -> str:
        """Combine Precision JSON with Vector Search."""
        # 1. Get high-precision facts from JSON
        precision_ctx = self._get_precision_context(query)
        
        # 2. Get fuzzy context from Vector Store (fallback/extra)
        try:
            results = self.vector_store.search(query, k=2) # Reduced k since we have JSON
            vector_ctx = "\n\n---\n\n".join(doc.page_content for doc in results)
        except Exception:
            vector_ctx = ""
            
        return f"{precision_ctx}\n\n--- ADDITIONAL CONTEXT ---\n\n{vector_ctx}"

    def _build_messages(
        self,
        query: str,
        context: str,
        history: List[Dict],
        lang: str
    ) -> List[Dict]:
        """Build the messages list for Groq API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add last 6 conversation turns (3 user + 3 assistant) for memory
        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Final user message with retrieved context
        lang_hint = {
            "bn": "Reply in Bengali script (বাংলা).",
            "hi": "Reply in Hindi script (हिन्दी).",
            "en": "Reply in English.",
        }.get(lang, "Reply in the same language as the user.")

        user_message = f"""CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

{lang_hint} Answer based only on the context above."""

        messages.append({"role": "user", "content": user_message})
        return messages

    def _is_lang_switch(self, query: str) -> tuple[bool, str | None]:
        """
        Detect if user is just asking to switch language.
        Returns (is_switch, forced_lang_code).
        e.g. "in bengali" → (True, "bn")
             "in hindi" → (True, "hi")
             "in english" → (True, "en")
             "বাংলায় বলো" → (True, "bn")
        """
        q = query.strip().lower()
        patterns = {
            "bn": ["in bengali", "bangla te", "banglay", "বাংলায়", "বাংলা তে", "bengali te", "bengali তে", "bangla y"],
            "hi": ["in hindi", "hindi me", "hindi mein", "हिंदी में", "हिंदी मे"],
            "en": ["in english", "english e", "ইংরেজিতে", "english me"],
        }
        for lang, triggers in patterns.items():
            for t in triggers:
                if t in q or q == t:
                    return True, lang
        return False, None

    def _get_last_user_question(self, history: List[Dict]) -> str | None:
        """Get the most recent user question from conversation history."""
        for turn in reversed(history):
            if turn.get("role") == "user":
                return turn.get("content", "").strip()
        return None

    async def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Main entry point. Returns dict with 'answer', 'voice_text', 'source'."""
        if not self.client:
            return {"answer": "Service unavailable. Please call 0343-2501353.", "voice_text": "", "source": "error"}

        start = time.time()
        try:
            history = conversation_history or []

            # 1. Check if user is just switching language (e.g. "in bengali")
            is_switch, forced_lang = self._is_lang_switch(query)
            if is_switch and forced_lang and history:
                # Re-run the PREVIOUS user question in the new language
                prev_question = self._get_last_user_question(history[:-1] if history else [])
                if prev_question:
                    query = prev_question
                    logger.info(f"Language switch detected → re-running '{query[:50]}' in lang={forced_lang}")
                lang = forced_lang
            else:
                # 2. Detect language normally
                lang = detect_language(query)

            # 3. Retrieve relevant context via vector search
            context = self._retrieve_context(query)

            # 4. Build messages with history + context
            messages = self._build_messages(
                query=query,
                context=context,
                history=history,
                lang=lang
            )

            # 5. Call Groq
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=self.max_tokens,
            )

            answer = completion.choices[0].message.content.strip()

            # 6. Currency normalization for Bengali
            if lang == "bn":
                answer = answer.replace("রুপি", "টাকা").replace("টাকা.", "টাকা।")

            latency_ms = round((time.time() - start) * 1000)
            logger.info(f"Groq response ({lang}, {latency_ms}ms): {answer[:80]}...")

            return {
                "answer": answer,
                "voice_text": answer,  # same text — voice_utils will clean it
                "source": "groq_rag",
                "model": self.model,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {
                "answer": "Sorry, something went wrong. Please call the college at 0343-2501353.",
                "voice_text": "Sorry, something went wrong.",
                "source": "error",
            }

    async def stream_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ):
        """Streaming version for low-latency web/voice UI."""
        if not self.async_client:
            yield "Service unavailable. Please call 0343-2501353."
            return

        try:
            history = conversation_history or []
            lang = detect_language(query)
            context = self._retrieve_context(query)
            messages = self._build_messages(query, context, history, lang)

            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=self.max_tokens,
                stream=True
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            yield "Sorry, something went wrong."

    def is_available(self) -> bool:
        return self.client is not None


# Singleton
_groq_service = None

def get_groq_service() -> GroqService:
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service
