"""
Groq LLM Service - Clean RAG Architecture
==========================================
Flow: User Query -> Vector Search (top 4 chunks) -> LLM with context -> Answer

No more full KB injection. No conflicting rules. Just simple, clean RAG.

Phase 0 (Hallucination Guardrail):
- After LLM response, extract numbers/entities and verify they exist in context.
- If a critical entity (fee amount, principal name, etc.) is NOT found in the
  retrieved context, replace the answer with a polite fallback.
"""

import logging
import os
import re
import time
import json
import hashlib
import random
import asyncio
import collections
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0 — Hallucination guard configuration
# ---------------------------------------------------------------------------
# Toggle validation on/off via env var. Defaults to ON.
HALLUCINATION_GUARD_ENABLED = os.getenv("HALLUCINATION_GUARD_ENABLED", "true").lower() == "true"

# Topics where we MUST validate entities against context.
# For other topics (greetings, general chat) we skip validation.
VALIDATED_TOPICS = {
    "fees",
    "fee",
    "hostel",
    "admission",
    "principal",
    "vice_principal",
    "contact",
    "phone",
    "email",
    "address",
    "placement",
    "cutoff",
    "scholarship",
    "documents",
    "eligibility",
}

GREETING_PATTERNS = (
    r"^hi+$",
    r"^hello+$",
    r"^hey+$",
    r"^hii+$",
    r"^hiii+$",
    r"^hello there$",
    r"^good (morning|afternoon|evening)$",
)

FALLBACK_ANSWER_EN = (
    "I'm sorry, I don't have that specific information. "
    "Please call the college at 0343-2501353 for accurate details."
)
FALLBACK_ANSWER_HI = (
    "क्षमा करें, मेरे पास यह विशिष्ट जानकारी नहीं है। "
    "कृपया सटीक जानकारी के लिए कॉलेज को 0343-2501353 पर कॉल करें।"
)
FALLBACK_ANSWER_BN = (
    "দুঃখিত, আমার কাছে এই নির্দিষ্ট তথ্য নেই। সঠিক তথ্যের জন্য অনুগ্রহ করে কলেজে 0343-2501353 নম্বরে কল করুন।"
)

# ---------------------------------------------------------------------------
# Phase 4 — Query cache configuration
# ---------------------------------------------------------------------------
# Cache TTL (seconds) and max entries. Tunable via env vars.
try:
    import cachetools  # type: ignore

    CACHE_AVAILABLE = True
except ImportError:
    cachetools = None  # type: ignore
    CACHE_AVAILABLE = False
    logger.warning("cachetools not installed — query cache will be disabled")

CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL", "600"))  # 10 min default
CACHE_MAX_ENTRIES = int(os.getenv("QUERY_CACHE_MAX", "256"))

# Knowledge gaps log path — admin checks this to know what queries need coverage.
_KNOWLEDGE_GAPS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge_gaps.json"
)

# Combined KB path — used to auto-invalidate cache when the file changes.
_KB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "knowledge_base"
    / "combined_kb.json"
)

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

LANGUAGE & TRANSLATION RULES:
- Bengali script input → reply in Bengali
- Hindi script input → reply in Hindi
- English/Banglish/Hinglish input → reply in the SAME style they used
- NEVER switch languages mid-response

TTS-FIRST WRITING RULES (Critical for voice — your output goes directly to text-to-speech):
- Spell out ALL numbers as words. NEVER use digits (0-9).
- Replace symbols: "percent" not "%", "per year" not "/year", "rupees" not "₹" or "Rs."
- Phone numbers as space-separated digits: "zero three four three two five zero one three five three"
- NO markdown, no bullet lists, no tables, no emoji. Plain sentences only.
- Use ONLY department abbreviations: CSE, IT, ECE, EE, ME, CE, CSD, AIML
- NEVER repeat both abbreviation and full name. Say "CSE" not "CSE (Computer Science and Engineering)"
- End every sentence with a period.

LANGUAGE-SPECIFIC NUMBER FORMATING:
- ENGLISH: "six lakh four thousand seven hundred rupees", "ninety-one percent", "approximately five lakh rupees"
- HINDI: "पाँच लाख" (panch lakh), "इक्यानबे प्रतिशत" (ikyanwe pratishat)
- BENGALI: "পাঁচ লাখ" (pañch lakh), "একানব্বই শতাংশ" (ekanabboi śatansh)

CRITICAL: Numbers must be spelled out in the script of the response language.
Example — fee 6,04,700:
  ENGLISH: "six lakh four thousand seven hundred rupees"  (NOT "₹6,04,700")
  HINDI: "छह लाख चार हज़ार सात सौ रुपये"  (NOT "6,04,700 रुपये")
  BENGALI: "ছয় লাখ চার হাজার সাতশো টাকা"  (NOT "6,04,700 টাকা")

NATURAL VOICE RULES:
- Be conversational and natural, like a helpful campus counselor.
- Keep responses concise for voice. Brief paragraphs, not lists.
- NEVER use filler phrases like "Based on the context provided" or "According to the knowledge base". Speak naturally.

CONTENT RULES:
- Answer ONLY from the CONTEXT below. Do not make up facts.
- NEVER invent company names. Only list companies mentioned in the context.
- If user gives rank/marks, use cutoff data to tell them eligible departments. Otherwise ask for rank and marks first.
- Physics, Chemistry, Math, Biology are FIRST-YEAR SUBJECTS, NOT admission departments. B.Tech departments: CSE, IT, ECE, EE, ME, CE, CSD, AIML, Data Science, Cyber Security.

OUT-OF-KB DEFLECTION:
- If context is empty or doesn't contain the answer, respond politely with phone:
  ENGLISH: "I don't have information about this. Please call the college: zero three four three two five zero one three five three."
  HINDI: "मेरे पास इस बारे में जानकारी नहीं है। कृपया कॉलेज को कॉल करें: शून्य तीन चार तीन दो पाँच शून्य एक तीन पाँच तीन।"
  BENGALI: "আমার কাছে এই বিষয়ে তথ্য নেই। অনুগ্রহ করে কলেজে কল করুন: শূন্য তিন চার তিন দুই পাঁচ শূন্য এক তিন পাঁচ তিন।"
- Do NOT make up data. Do NOT invent names, fees, or numbers."""


# ---------------------------------------------------------------------------
# Phase 5 — Rate limiter + circuit breaker for Groq API (429 prevention)
# ---------------------------------------------------------------------------
# Sliding window: max R requests in W seconds before self-throttling
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("GROQ_RATE_LIMIT_MAX", "25"))  # 25 req/min free tier
RATE_LIMIT_WINDOW_SEC = int(os.getenv("GROQ_RATE_LIMIT_WINDOW", "60"))

# Circuit breaker: after N consecutive 429s, pause for B seconds
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("GROQ_CIRCUIT_THRESHOLD", "3"))
CIRCUIT_BREAKER_BACKOFF = int(os.getenv("GROQ_CIRCUIT_BACKOFF", "30"))

# Max backoff cap (single call) to prevent unbounded wait
MAX_BACKOFF_SEC = 10.0

# Model fallback list — tried in order on rate limit
# Primary: llama-3.1-8b-instant (fast, good quality)
# Fallback: groq/compound-mini (lighter, different backend = separate rate limit pool)
# Fallback: qwen/qwen3-32b (good multilingual, may have separate rate limits)
FALLBACK_MODELS = ["llama-3.1-8b-instant", "groq/compound-mini", "qwen/qwen3-32b"]


@dataclass
class RateLimiter:
    """Simple sliding-window rate limiter + circuit breaker."""

    max_requests: int = RATE_LIMIT_MAX_REQUESTS
    window_sec: int = RATE_LIMIT_WINDOW_SEC
    circuit_threshold: int = CIRCUIT_BREAKER_THRESHOLD
    circuit_backoff: float = float(CIRCUIT_BREAKER_BACKOFF)

    _timestamps: "collections.deque[float]" = field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )
    _consecutive_429s: int = 0
    _circuit_open_until: float = 0.0

    def _prune(self, now: float) -> None:
        """Remove timestamps outside the window."""
        cutoff = now - self.window_sec
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def acquire(self, now: float | None = None) -> float:
        """Try to acquire a slot. Returns wait time in seconds (0 = go now)."""
        now = now or time.time()
        self._prune(now)

        # Circuit breaker check
        if now < self._circuit_open_until:
            remaining = self._circuit_open_until - now
            logger.warning(
                f"Circuit breaker OPEN — waiting {remaining:.1f}s "
                f"(consecutive 429s={self._consecutive_429s})"
            )
            return min(remaining, MAX_BACKOFF_SEC)

        # Rate limit check
        if len(self._timestamps) >= self.max_requests:
            oldest = self._timestamps[0]
            wait = oldest + self.window_sec - now
            if wait > 0:
                return min(wait, MAX_BACKOFF_SEC)

        self._timestamps.append(now)
        return 0.0

    def record_429(self) -> None:
        """Record a 429 response. May open the circuit breaker."""
        self._consecutive_429s += 1
        if self._consecutive_429s >= self.circuit_threshold:
            self._circuit_open_until = time.time() + self.circuit_backoff
            logger.warning(
                f"Circuit breaker TRIPPED after {self._consecutive_429s} consecutive 429s — "
                f"pausing {self.circuit_backoff}s"
            )

    def record_success(self) -> None:
        """Reset consecutive 429 counter on success."""
        self._consecutive_429s = 0

    @property
    def is_circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until


class GroqService:
    """
    Clean Hybrid RAG service: JSON (Precision) + Vector Store (Context).
    """

    def __init__(self):
        self.model = "llama-3.1-8b-instant"
        self.max_tokens = 384
        self.client = settings.groq_client
        self.async_client = getattr(settings, "async_groq_client", None)
        self.vector_store = get_vector_store()

        # Session memory — in-memory dict, cleared on server restart
        self._sessions: Dict[str, List[Dict]] = {}

        # Core knowledge is read from disk on EVERY request (no stale cache across processes).
        self.core_kb = {}

        # Phase 4 — Query cache (in-memory, TTL-based)
        # Caches the FINAL response by (query, lang, context_hash, kb_mtime).
        # Auto-invalidates if the KB file changes on disk.
        self._cache = None
        self._cache_stats = {"hits": 0, "misses": 0}
        self._kb_mtime = None
        try:
            if _KB_PATH.exists():
                self._kb_mtime = _KB_PATH.stat().st_mtime
        except Exception:
            pass
        if CACHE_AVAILABLE and cachetools is not None:
            self._cache = cachetools.TTLCache(maxsize=CACHE_MAX_ENTRIES, ttl=CACHE_TTL_SECONDS)
            logger.info(f"Query cache ENABLED (ttl={CACHE_TTL_SECONDS}s, max={CACHE_MAX_ENTRIES})")
        else:
            logger.info("Query cache DISABLED (cachetools not installed)")

        # Phase 5 — Rate limiter + circuit breaker
        self._rate_limiter = RateLimiter()

        if self.client:
            logger.info("GroqService ready.")
        else:
            logger.warning("GroqService: Groq client not found. Check GROQ_API_KEY in .env")

    # -----------------------------------------------------------------------
    # Phase 4 — Cache helpers
    # -----------------------------------------------------------------------
    def _read_kb(self) -> dict:
        """Read combined_kb.json from disk. Cached in memory with mtime check."""
        try:
            kb_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "knowledge_base"
                / "combined_kb.json"
            )
            mtime = kb_path.stat().st_mtime
            if (
                self._kb_mtime is not None
                and mtime == self._kb_mtime
                and hasattr(self, "_kb_cache")
            ):
                return self._kb_cache
            with open(kb_path, "r", encoding="utf-8") as f:
                self._kb_cache = json.load(f)
            self._kb_mtime = mtime
            return self._kb_cache
        except Exception as e:
            logger.error(f"Failed to read combined_kb.json: {e}")
            return {}

    def _cache_key(self, query: str, lang: str, context: str) -> str:
        """Build a stable cache key from query + lang + context + KB mtime."""
        ctx_hash = hashlib.md5(context.encode("utf-8")).hexdigest()[:12] if context else "0"
        return f"{lang}|{query.strip().lower()}|{ctx_hash}|{self._kb_mtime or 0}"

    def _check_kb_changed(self) -> None:
        """Invalidate cache if combined_kb.json mtime changed on disk."""
        try:
            if not _KB_PATH.exists():
                return
            mtime = _KB_PATH.stat().st_mtime
            if self._kb_mtime is None or mtime != self._kb_mtime:
                if self._cache is not None:
                    logger.info("KB file changed on disk — invalidating query cache")
                    self._cache.clear()
                self._kb_mtime = mtime
        except Exception as e:
            logger.debug(f"KB mtime check failed: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Expose hit/miss counts for /metrics later."""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        ratio = round(self._cache_stats["hits"] / total, 3) if total else 0.0
        return {
            **self._cache_stats,
            "total": total,
            "hit_ratio": ratio,
            "size": len(self._cache) if self._cache is not None else 0,
            "max_size": CACHE_MAX_ENTRIES,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "enabled": self._cache is not None,
        }

    def invalidate_cache(self) -> int:
        """Manually clear the cache. Returns # of entries cleared."""
        if self._cache is None:
            return 0
        cleared = len(self._cache)
        self._cache.clear()
        logger.info(f"Query cache manually invalidated ({cleared} entries)")
        return cleared

    def reload_kb(self) -> int:
        """Invalidate cache. KB is read from disk on every request now,
        so no reload needed — just clear stale cache entries.
        Returns the number of voice_ready_answers entries (0 if none)."""
        try:
            self._kb_mtime = _KB_PATH.stat().st_mtime if _KB_PATH.exists() else None
            if self._cache is not None:
                self._cache.clear()
            kb = self._read_kb()
            count = len(kb.get("voice_ready_answers", {}))
            logger.info(f"Cache invalidated. KB has {count} voice_ready_answers entries")
            return count
        except Exception as e:
            logger.error(f"Failed to reload KB: {e}")
            return 0

    def _normalize_query(self, query: str) -> str:
        """Fix common STT mis-transcriptions of BCREC department names before RAG.
        Also transliterates Roman-script Bengali college terms to Bengali script
        so that vector search matches the native-script KB entries."""
        q = query
        q = re.sub(r"\bcseaml\b", "CSE-AIML", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcs e[\s-]?aml\b", "CSE-AIML", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcciml\b", "AIML", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcsd\b", "CSD", q, flags=re.IGNORECASE)
        q = re.sub(r"\bdata sci\b", "Data Science", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcyber sec\b", "Cyber Security", q, flags=re.IGNORECASE)
        q = re.sub(r"\binfo tech\b", "Information Technology", q, flags=re.IGNORECASE)
        q = re.sub(r"\belec[ -]?comm\b", "ECE", q, flags=re.IGNORECASE)
        q = re.sub(r"\belectrical\b", "EE", q, flags=re.IGNORECASE)
        q = re.sub(r"\bmechanical\b", "ME", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcivil\b", "CE", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcomputer\b", "CSE", q, flags=re.IGNORECASE)
        q = re.sub(r"\bai[\s-]?ml\b", "AIML", q, flags=re.IGNORECASE)
        # Roman-script Bengali college terms → Bengali script for vector match
        q = re.sub(r"\bupo[- ]?pradhan\b", "উপ-প্রধান", q, flags=re.IGNORECASE)
        if q != query:
            logger.info(f"Query normalized: '{query}' -> '{q}'")
        return q

    def _retrieve_context(self, query: str) -> str:
        """Enhanced retriever: vector search + semantic re-rank + section-aware filtering.
        Keeps top docs across sources, prioritizes by semantic anchor + language match."""
        normalized = self._normalize_query(query)
        try:
            language = detect_language(query)
            results = self.vector_store.search(normalized, k=10)
            if not results:
                return ""

            query_lower = normalized.lower()
            query_words = set(query_lower.split())

            def semantic_score(doc):
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                score = 0
                anchor = meta.get("semantic_anchor", "").lower()
                anchor_words = anchor.split()
                score += sum(1 for w in anchor_words if w in query_words)
                section = meta.get("section", "")
                if section and section.lower() in query_lower:
                    score += 2
                if meta.get("language") == language:
                    score += 1
                if meta.get("language") == "en" and language != "en":
                    score -= 0.5
                return score

            ranked = sorted(results, key=semantic_score, reverse=True)

            context_chunks = []
            seen_sections = set()
            for doc in ranked:
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                section = meta.get("section", "")
                sub = meta.get("subsection", "")
                dedup_key = f"{section}:{sub}"
                if dedup_key in seen_sections:
                    continue
                seen_sections.add(dedup_key)
                text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                if "[" in text and "]" in text:
                    text = text.split("]", 1)[-1].strip()
                context_chunks.append(text)
                if len(context_chunks) >= 5:
                    break

            return "\n\n---\n\n".join(context_chunks)
        except Exception:
            logger.warning("Vector search failed, returning empty context")
            return ""

    def _build_messages(
        self, query: str, context: str, history: List[Dict], lang: str
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
            "bn": [
                "in bengali",
                "bangla te",
                "banglay",
                "বাংলায়",
                "বাংলা তে",
                "bengali te",
                "bengali তে",
                "bangla y",
            ],
            "hi": ["in hindi", "hindi me", "hindi mein", "हिंदी में", "हिंदी मे"],
            "en": ["in english", "english e", "ইংরেজিতে", "english me"],
        }
        for lang, triggers in patterns.items():
            for t in triggers:
                if t in q or q == t:
                    return True, lang
        return False, None

    def _is_greeting(self, query: str) -> bool:
        """Handle short greetings without calling the LLM."""
        q = query.strip().lower()
        if not q:
            return False
        return any(re.match(pattern, q) for pattern in GREETING_PATTERNS)

    def _get_last_user_question(self, history: List[Dict]) -> str | None:
        """Get the most recent user question from conversation history."""
        for turn in reversed(history):
            if turn.get("role") == "user":
                return turn.get("content", "").strip()
        return None

    # -----------------------------------------------------------------------
    # Session memory (in-memory, clears on restart)
    # -----------------------------------------------------------------------
    def _get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session. Returns empty list if new session."""
        return self._sessions.get(session_id, [])

    def _append_session_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Store a user+assistant turn in the session's in-memory history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": "user", "content": user_msg})
        self._sessions[session_id].append({"role": "assistant", "content": assistant_msg})
        # Keep last 12 turns to prevent unbounded growth
        if len(self._sessions[session_id]) > 12:
            self._sessions[session_id] = self._sessions[session_id][-12:]

    def clear_session(self, session_id: str) -> None:
        """Clear a session's memory. Called when session ends."""
        self._sessions.pop(session_id, None)

    # -----------------------------------------------------------------------
    # Phase 0 — Hallucination guard
    # -----------------------------------------------------------------------
    def _should_validate(self, query: str) -> bool:
        """Only validate on high-stakes topics (fees, names, contact, etc.)."""
        if not HALLUCINATION_GUARD_ENABLED:
            return False
        q = query.lower()
        return any(topic in q for topic in VALIDATED_TOPICS)

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract verifiable entities from the LLM answer.
        Returns: {"numbers": [...], "names": [...], "phone_like": [...]}
        """
        entities: Dict[str, List[str]] = {
            "numbers": [],
            "names": [],
            "phone_like": [],
        }

        # 1) Numbers (≥3 digit) — likely fee amounts, intakes, cutoffs
        #    E.g. "598300", "5,98,300", "180" — but NOT "93.6" (decimal)
        for m in re.finditer(r"\b\d[\d,]{2,}\b", text):
            raw = m.group(0)
            # Skip if this number is part of a percentage (e.g., 80.62%)
            end_pos = m.end()
            if end_pos < len(text) and text[end_pos] == "%":
                continue
            num = raw.replace(",", "")
            if num.isdigit() and len(num) >= 3:
                entities["numbers"].append(num)

        # 2) Percentages like "90%", "85%+"
        for m in re.finditer(r"\b\d{1,3}\s*%(?:\+)?", text):
            entities["numbers"].append(m.group(0).strip().replace("+", ""))

        # 3) Phone-like numbers (10+ digits possibly with hyphens/spaces)
        for m in re.finditer(r"\b\d{4}[\s\-]?\d{3,7}\b", text):
            entities["phone_like"].append(re.sub(r"[\s\-]", "", m.group(0)))

        # 4) Proper nouns / titles: "Dr.", "Prof.", words starting with capital
        #    We only check KNOWN key names below — generic names create noise.
        #    So we leave this empty; number/phone checks are stronger signals.
        return entities

    def _context_contains(self, context: str, entity: str) -> bool:
        """Check if entity appears in context (normalize Unicode digits to ASCII)."""
        if not entity:
            return True

        def normalize_digits(text: str) -> str:
            """Convert all Unicode digit scripts to ASCII."""
            result = text
            bengali_digits = "০১২৩৪৫৬৭৮৯"
            devanagari_digits = "०१२३४५६७८९"
            for i in range(10):
                result = result.replace(bengali_digits[i], str(i))
                result = result.replace(devanagari_digits[i], str(i))
            return result

        ctx_norm = normalize_digits(context).replace(",", "").replace(" ", "").lower()
        ent_norm = normalize_digits(entity).replace(",", "").replace(" ", "").lower()
        return ent_norm in ctx_norm

    def _validate_answer(self, answer: str, context: str, query: str) -> tuple[bool, str]:
        """
        Validate that critical entities in `answer` appear in the context.
        Skips numbers that appear in the user's query (they provided them).
        Skips low-risk queries like placement rate (college-wide stat, never hallucinated).
        Returns (is_valid, reason_if_invalid).
        """
        if not self._should_validate(query):
            return True, ""

        # Exempt only general placement queries without numbers — ones asking about rate/companies
        query_lower = query.lower()
        has_number = bool(re.search(r"\d", query_lower))
        if not has_number and any(
            word in query_lower for word in ["placement", "प्लेसमेंट", "প্লেসমেন্ট"]
        ):
            return True, ""

        entities = self._extract_entities(answer)

        # Only check numbers and phone-like strings — these are the high-signal
        # hallucination markers. Names are too noisy to check blindly.
        critical = entities["numbers"] + entities["phone_like"]
        if not critical:
            return True, ""

        # Extract user-provided numbers from the query — skip those
        query_entities = self._extract_entities(query)
        query_numbers = set(query_entities["numbers"] + query_entities["phone_like"])

        missing = [
            e for e in critical if e not in query_numbers and not self._context_contains(context, e)
        ]
        if missing:
            return False, f"entities not found in context: {missing[:5]}"
        return True, ""

    def _prepare_for_tts(self, text: str, lang: str) -> str:
        """Catch remaining digits the LLM missed — delegate to voice_utils."""
        try:
            from app.utils.voice_utils import clean_for_voice

            return clean_for_voice(text)
        except Exception:
            return text

    def _safe_fallback(self, lang: str) -> str:
        """Return a polite, language-appropriate fallback message."""
        if lang == "hi":
            return FALLBACK_ANSWER_HI
        if lang == "bn":
            return FALLBACK_ANSWER_BN
        return FALLBACK_ANSWER_EN

    def _log_gap(self, query: str, lang: str, reason: str) -> None:
        """Log an unanswered query to knowledge_gaps.json so admins know what to add."""
        import datetime

        try:
            path = _KNOWLEDGE_GAPS_PATH
            gaps = []
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    gaps = json.load(f)
            gaps.append(
                {
                    "query": query,
                    "lang": lang,
                    "reason": reason,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "count": 1,
                }
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gaps, f, indent=2, ensure_ascii=False)
            logger.info(f"Knowledge gap logged: '{query[:60]}' ({reason})")
        except Exception as e:
            logger.debug(f"Failed to log knowledge gap: {e}")

    async def generate_response(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Main entry point. Uses in-memory session memory for conversation context.
        Returns dict with 'answer', 'voice_text', 'source'."""
        if not self.client:
            return {
                "answer": "Service unavailable. Please call 0343-2501353.",
                "voice_text": "",
                "source": "error",
            }

        start = time.time()
        try:
            history = self._get_session_history(session_id)

            # 1. Check if user is just switching language (e.g. "in bengali")
            is_switch, forced_lang = self._is_lang_switch(query)
            if is_switch and forced_lang and history:
                prev_question = self._get_last_user_question(history[:-1] if history else [])
                if prev_question:
                    query = prev_question
                    logger.info(
                        f"Language switch detected → re-running '{query[:50]}' in lang={forced_lang}"
                    )
                lang = forced_lang
            else:
                lang = detect_language(query)

            # 2. Deterministic greeting
            if self._is_greeting(query):
                greeting_answer = {
                    "en": "Hello. How can I help you with admissions, fees, courses, or hostel details?",
                    "hi": "नमस्ते। मैं admissions, fees, courses, और hostel details में मदद कर सकता हूँ।",
                    "bn": "হ্যালো। আমি admissions, fees, courses, আর hostel details নিয়ে সাহায্য করতে পারি।",
                }.get(lang, "Hello. How can I help you?")

                latency_ms = round((time.time() - start) * 1000)
                logger.info(f"GREETING HIT (lang={lang}, {latency_ms}ms): '{query[:60]}'")
                self._append_session_turn(session_id, query, greeting_answer)
                return {
                    "answer": greeting_answer,
                    "voice_text": greeting_answer,
                    "source": "greeting_deterministic",
                    "model": "none",
                    "latency_ms": latency_ms,
                    "hallucination_validated": True,
                    "tokens": {"prompt": 0, "completion": 0},
                    "cache_hit": False,
                }

            # 3. Retrieve context
            context = self._retrieve_context(query)

            # 4. Cache lookup (skip for sessions with history)
            if self._cache is not None and not history:
                self._check_kb_changed()
                cache_key = self._cache_key(query, lang, context)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache_stats["hits"] += 1
                    latency_ms = round((time.time() - start) * 1000)
                    logger.info(f"CACHE HIT (lang={lang}, {latency_ms}ms): '{query[:60]}'")
                    cached_out = dict(cached)
                    cached_out["latency_ms"] = latency_ms
                    cached_out["cache_hit"] = True
                    self._append_session_turn(session_id, query, cached_out["answer"])
                    return cached_out
                self._cache_stats["misses"] += 1

            # 5. Build messages with session history + context
            messages = self._build_messages(
                query=query, context=context, history=history, lang=lang
            )

            # 6. Call Groq with rate limiter + model fallback + circuit breaker
            completion = None
            used_model = self.model
            models_to_try = list(dict.fromkeys([self.model] + FALLBACK_MODELS))
            success = False

            for current_model in models_to_try:
                for attempt in range(3):
                    wait = self._rate_limiter.acquire()
                    if wait > 0:
                        logger.info(
                            f"Rate limiter: waiting {wait:.1f}s before calling {current_model}"
                        )
                        await asyncio.sleep(wait)

                    try:
                        completion = self.client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=self.max_tokens,
                        )
                        self._rate_limiter.record_success()
                        used_model = current_model
                        success = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        is_rate_limit = (
                            "429" in error_str
                            or "rate" in error_str.lower()
                            or "too many" in error_str.lower()
                        )
                        if is_rate_limit:
                            self._rate_limiter.record_429()
                            if attempt < 2:
                                backoff = min((2**attempt) + random.random(), MAX_BACKOFF_SEC)
                                logger.warning(
                                    f"Groq rate limited on {current_model} "
                                    f"(model {models_to_try.index(current_model) + 1}/{len(models_to_try)}, "
                                    f"attempt {attempt + 1}/3), "
                                    f"retrying in {backoff:.1f}s"
                                )
                                await asyncio.sleep(backoff)
                            else:
                                logger.warning(f"All retries exhausted for {current_model}")
                        else:
                            raise
                if success:
                    break
                if current_model != models_to_try[-1]:
                    logger.warning(
                        f"Switching model {current_model} -> {models_to_try[models_to_try.index(current_model) + 1]}"
                    )

            if not success:
                raise RuntimeError(f"All Groq models ({models_to_try}) rate limited after retries")
            answer = completion.choices[0].message.content.strip()
            # Strip reasoning tags (e.g. <think>...</think>) from inference models
            import re as _re

            answer = _re.sub(r"<think>.*?</think>\s*", "", answer, flags=_re.DOTALL).strip()

            # 7. Hallucination guard
            is_valid, reason = self._validate_answer(answer, context, query)
            if not is_valid:
                logger.warning(
                    f"PHASE 0 GUARD TRIPPED (lang={lang}, query='{query[:60]}'): {reason}. "
                    f"LLM said: '{answer[:80]}'"
                )
                self._log_gap(query, lang, f"hallucination_guard: {reason}")
                answer = self._safe_fallback(lang)

            # 7.5 Out-of-KB detection
            if is_valid and not self._is_greeting(query) and "0343-2501353" not in answer:
                a = answer.lower().strip()
                unknown_signals = (
                    a.startswith("i'm not ")
                    or a.startswith("i am not ")
                    or "no information" in a
                    or "not aware" in a
                    or "don't have" in a
                    or "does not contain" in a
                    or "context does not contain" in a
                    or "not found in" in a
                    or a.startswith("i'm afraid")
                    or a.startswith("i'm sorry")
                    or "unfortunately" in a
                    or "sorry, i" in a
                    or "beyond the context" in a
                    or "can't find" in a
                    or "cannot find" in a
                    or "cannot answer" in a
                    or "जानकारी नहीं" in a
                    or "पता नहीं" in a
                    or "জান নেই" in a
                    or "পাওয়া যায়নি" in a
                )
                if unknown_signals:
                    logger.warning(
                        f"OUT-OF-KB DETECTED (lang={lang}, query='{query[:60]}'): "
                        f"LLM said '{answer[:80]}' without phone → replacing with fallback"
                    )
                    self._log_gap(query, lang, "out_of_kb_deflection")
                    answer = self._safe_fallback(lang)

            # 8. Bengali normalization
            if lang == "bn":
                answer = answer.replace("রুপি", "টাকা").replace("টাকা.", "টাকা।")

            # 8.5 Prepare for TTS — catch remaining digits the LLM missed
            answer = self._prepare_for_tts(answer, lang)

            # 9. Token tracking
            prompt_tokens = 0
            completion_tokens = 0
            try:
                usage = getattr(completion, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            except Exception:
                pass
            latency_ms = round((time.time() - start) * 1000)
            logger.info(
                f"Groq response ({lang}, {latency_ms}ms, validated={is_valid}, "
                f"tokens=in:{prompt_tokens}/out:{completion_tokens}): {answer[:80]}..."
            )

            response_payload: Dict[str, Any] = {
                "answer": answer,
                "voice_text": answer,
                "source": "groq_rag",
                "model": used_model,
                "latency_ms": latency_ms,
                "hallucination_validated": is_valid,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                },
                "cache_hit": False,
            }

            # 10. Store in session memory
            self._append_session_turn(session_id, query, answer)

            # 11. Cache (only if no history, i.e. first turn)
            if self._cache is not None and is_valid and not history:
                try:
                    cache_key = self._cache_key(query, lang, context)
                    self._cache[cache_key] = response_payload
                except Exception as e:
                    logger.debug(f"Cache store failed: {e}")

            return response_payload

        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {
                "answer": "Sorry, something went wrong. Please call the college at 0343-2501353.",
                "voice_text": "Sorry, something went wrong.",
                "source": "error",
                "cache_hit": False,
            }

    async def stream_response(
        self,
        query: str,
        session_id: str = "default",
        conversation_history: Optional[List[Dict]] = None,
    ):
        """True streaming for low-latency telephony. Yields tokens immediately.
        Hallucination guard runs in background — logs violations but does NOT block.
        """
        t0 = time.time()
        if not self.async_client:
            yield "Service unavailable. Please call 0343-2501353."
            return

        try:
            history = (
                conversation_history
                if conversation_history is not None
                else self._get_session_history(session_id)
            )
            lang = detect_language(query)

            llm_query = self._normalize_query(query)
            context = self._retrieve_context(query)
            t_prep = time.time() - t0

            messages = self._build_messages(llm_query, context, history, lang)

            # Retry/backoff for rate limits (async) — with rate limiter + model fallback
            stream = None
            models_to_try = list(dict.fromkeys([self.model] + FALLBACK_MODELS))
            success = False

            for current_model in models_to_try:
                for attempt in range(3):
                    wait = self._rate_limiter.acquire()
                    if wait > 0:
                        await asyncio.sleep(wait)

                    try:
                        stream = await self.async_client.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=self.max_tokens,
                            stream=True,
                        )
                        self._rate_limiter.record_success()
                        success = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        is_rate_limit = (
                            "429" in error_str
                            or "rate" in error_str.lower()
                            or "too many" in error_str.lower()
                        )
                        if is_rate_limit:
                            self._rate_limiter.record_429()
                            if attempt < 2:
                                backoff = min((2**attempt) + random.random(), MAX_BACKOFF_SEC)
                                logger.warning(
                                    f"Groq rate limited (async) on {current_model} "
                                    f"(model {models_to_try.index(current_model) + 1}/{len(models_to_try)}, "
                                    f"attempt {attempt + 1}/3), "
                                    f"retrying in {backoff:.1f}s"
                                )
                                await asyncio.sleep(backoff)
                            else:
                                logger.warning(f"All retries exhausted for {current_model}")
                        else:
                            raise
                if success:
                    break
                if current_model != models_to_try[-1]:
                    logger.warning(
                        f"Switching model {current_model} -> "
                        f"{models_to_try[models_to_try.index(current_model) + 1]}"
                    )

            if not success:
                raise RuntimeError(f"All Groq models ({models_to_try}) rate limited after retries")
            t_llm_first = time.time() - t0

            # Stream tokens immediately — no upfront buffering
            buffer: List[str] = []
            first_token = True
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    if first_token:
                        ttft = round((time.time() - t0) * 1000)
                        logger.info(
                            f"STREAM TTFT={ttft}ms prep={round(t_prep * 1000)}ms "
                            f"llm_setup={round((t_llm_first - t_prep) * 1000)}ms "
                            f"lang={lang} q='{query[:60]}'"
                        )
                        first_token = False
                    buffer.append(content)
                    yield content

            full_answer = "".join(buffer).strip()
            if not full_answer:
                return

            t_total = round((time.time() - t0) * 1000)
            char_count = len(full_answer)
            logger.info(
                f"STREAM COMPLETE total={t_total}ms chars={char_count} "
                f"~{round(char_count / t_total * 1000, 1)}cps lang={lang}"
            )

            # Non-blocking validation (logs warnings, doesn't replace output)
            is_valid, reason = self._validate_answer(full_answer, context, query)
            if not is_valid:
                logger.warning(
                    f"PHASE 0 GUARD (stream, non-blocking) lang={lang} query='{query[:60]}': {reason}"
                )
                self._log_gap(query, lang, f"hallucination_guard_stream: {reason}")

            already_said_no_info = any(
                phrase in full_answer.lower()
                for phrase in (
                    "no information",
                    "not aware",
                    "don't have",
                    "does not contain",
                    "context does not contain",
                    "not found in",
                    "can't find",
                    "cannot find",
                    "cannot answer",
                    "जानकारी नहीं",
                    "পাতা নেই",
                    "পাওয়া যায়নি",
                    "জান নেই",
                )
            )
            if already_said_no_info and "0343-2501353" not in full_answer:
                logger.warning(
                    f"OUT-OF-KB DETECTED (stream, non-blocking) lang={lang} query='{query[:60]}': "
                    f"LLM said '{full_answer[:80]}' without phone"
                )
                self._log_gap(query, lang, "out_of_kb_deflection_stream")

            if conversation_history is None:
                self._append_session_turn(session_id, query, full_answer)

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
