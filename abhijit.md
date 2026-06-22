# Zero‑Budget Plan for a Clear, Correct, Low‑Latency BCREC Voice Agent

## Context
The BCREC voice agent currently uses:
- **LLM**: Groq (llama-3.1-8b-instant) – free tier claimed but no usage tracking
- **Vector Store**: ChromaDB with local BGE‑M3 embeddings (zero API cost)
- **Audio Pipeline**: Sarvam TTS/STT (primary), Deepgram STT (alternative), LiveKit transport, Twilio telephony
- **Observed Issues**:
  - Occasional incorrect or “no data” answers in Bengali/Hindi (hallucinations or missing KB)
  - Audio quality perceived as robotic / low clarity
  - End‑to‑end latency ~1.1‑1.4 s
  - No visibility into usage/cost

## Goal
Impress stakeholders **without spending any money** by delivering a voice agent that:
1. **Speaks clearly** (natural pronunciation, minimal artifacts)
2. **Answers correctly** – 100 % grounded in the provided knowledge base (no hallucinations)
3. **Responds fast** – latency ≤ 0.6 s
4. **Stays within free/trial limits** of all services (or uses fully local fallbacks)
5. **Reduces fallback to human agents** by improving correctness and clarity

## Core Strategy
- **Keep all API calls within existing free tiers** (Groq, Sarvam, Deepgram, Twilio, LiveKit) or replace them with local equivalents when free limits are approached.
- **Add hallucination prevention** by strictly validating LLM output against the knowledge base before speaking.
- **Optimize latency and clarity** through caching, efficient vector search (FAISS), audio‑pipeline tweaks, and VAD.
- **Add lightweight usage tracking** to ensure we never exceed free quotas (alert if we approach limits).

The plan is organized in phases; each phase can be deployed independently and delivers immediate, measurable improvement.

---

### Phase 0: Usage Tracking & Safety Guardrails (Days 0‑2)
Add lightweight monitoring and a “ground‑truth check” step to eliminate hallucinations.
- **Files**:
  - `backend/app/services/groq_service.py` – wrap the LLM call to:
    * Count input & output tokens (using `tiktoken`).
    * Estimate cost (Groq free‑tier rate = $0 for now) and log.
    * After receiving the answer, run a **validation function** that checks whether key entities (numbers, proper nouns, fee amounts) appear in the retrieved context; if not, replace the answer with a safe fallback: “I’m sorry, I don’t have that information. Please call the college at 0343‑2501353.”
  - `backend/app/services/sarvam_service.py` – log characters processed for TTS and seconds of audio for STT.
  - `backend/app/services/voice_session.py` – log total audio duration per request and API call counts.
  - `backend/app/config.py` – add optional cost‑per‑unit constants (set to 0 for free tier) and flags to enable/disable validation.
- **Outcome**: 
  - Zero hallucinations (answers either correct from KB or a polite fallback).
  - Real‑time visibility into token/audio consumption; we can stay under free limits.
  - No functional change to the happy path.

### Phase 1: Token & Context Optimization (Days 2‑5)
Make the LLM prompt as compact as possible to reduce latency and avoid truncation.
- **Files**:
  - `backend/app/services/groq_service.py` – integrate `tiktoken` to compute tokens for system prompt + retrieved context + user query. If total exceeds a safe threshold (e.g., 3500 tokens), trim the lowest‑ranked vector‑store chunks until fit.
  - `backend/app/services/vector_store.py` – expose a configurable `k` (default 4) via environment variable `VECTOR_TOP_K`.
- **Outcome**:
  - Faster LLM inference (less context to process).
  - Lower token usage → more headroom within free‑tier rate limits.
  - Reduced risk of context overflow causing missing information.

### Phase 2: Vector Store Performance Boost (Days 5‑8)
Swap ChromaDB for FAISS (still using the same BGE‑M3 embeddings) to cut search latency.
- **Files**:
  - Create `backend/app/services/vector_store_faiss.py` implementing the same interface (`search`, `add_documents`, `clear_collection`).
  - Add a factory in `backend/app/services/__init__.py` that selects the backend based on `VECTOR_STORE_BACKEND` env var (`faiss` or `chroma`).
  - Provide a one‑time migration script `scripts/migrate_chroma_to_faiss.py` to export vectors from ChromaDB and import into a FAISS index.
- **Outcome**:
  - Similarity search latency drops from ~100‑150 ms (ChromaDB) to < 30 ms (FAISS).
  - End‑to‑end latency reduced by ~200 ms.
  - Zero extra cost (local embeddings, disk‑based index).

### Phase 3: Audio Pipeline Clarity & Latency (Days 8‑12)
Make the spoken output clearer and faster by aligning sample rates, removing unnecessary resampling, and adding Voice Activity Detection (VAD).
- **Steps**:
  1. **Target 8 kHz end‑to‑end** (Twilio’s native codec). Adjust LiveKit agent and voice_session to produce 8 kHz Opus directly, removing the `audioop.ratecv` step.
  2. **Insert VAD** in the LiveKit‑Twilio media bridge: transmit audio only when speech energy exceeds a threshold (using `webrtcvad` or simple energy‑based VAD). This cuts bandwidth and processing during silence.
  3. **Cache frequent TTS outputs** (college greeting, common FAQ answers) using an LRU cache keyed by `(text, language, speaker)`. Store the raw PCM bytes to avoid re‑synthesizing.
- **Files**:
  - `backend/app/services/voice_session.py` – modify audio resampling logic, add VAD wrapper around the LiveKit‑Twilio bridge.
  - `backend/app/services/sarvam_service.py` – wrap `text_to_speech` with `@lru_cache(maxsize=256)` or a TTLCache.
- **Outcome**:
  - Audio latency reduction ~150‑250 ms (less resampling, VAD skips silence).
  - Clearer speech (no resampling artifacts, consistent 8 kHz).
  - Lower Twilio/LiveKit usage → stays well within free trial minutes.

### Phase 4: Query‑Level Caching for Instant Replies (Days 12‑15)
Cache the final answer (or the retrieved context + LLM response) for high‑frequency intents (fees, hostel, contact, admission).
- **Files**:
  - Decorate `GroqService.generate_response` (or the LLM‑service abstraction) with `cachetools.TTLCache` (TTL = 10 minutes) keyed by a hash of `(query, language, retrieved_context_hash)`.
  - Invalidate the cache automatically when `combined_kb.json` changes (simple file‑watcher or CI/CD hook).
- **Outcome**:
  - Repeated questions (≈70 % of traffic) served in < 200 ms (cache hit).
  - Drastic reduction in LLM token usage → virtually zero cost even if free‑tier limits are low.
  - Near‑instant response for the most common stakeholder demo queries.

### Phase 5: Monitoring, Alerting & Free‑Tier Safety (Ongoing)
Expose a `/metrics` endpoint and set up a lightweight alert to warn if usage approaches any free‑tier limit.
- **Files**:
  - Create `backend/app/metrics.py` – Prometheus‑style counters for:
    * LLM input/output tokens
    * Sarvam TTS characters, STT seconds
    * Twilio call minutes
    * LiveKit minutes
    * Cache hit/miss ratios
  - Add a FastAPI route (or Flask) at `/metrics`.
  - Add a nightly cron job (or simple background thread) that checks counters against known free‑tier thresholds and logs a warning if > 80 % utilized.
- **Outcome**:
  - Stakeholders can see a live dashboard showing cost‑free operation.
  - Proactive warnings prevent accidental over‑usage and service disruption.

---

## Estimated Monthly Cost (Free/Trial Only)
| Service | Free Tier Assumptions | Post‑Optimization Usage | Cost |
|---------|----------------------|-------------------------|------|
| Groq (llama-3.1-8b-instant) | Generous free rate limits (no public cap) | ~300k tokens/mo (with caching & token opt) | **$0** |
| Sarvam TTS/STT | Unknown free tier; assume limited but we stay well under via caching & VAD | ~300k characters TTS, ~180 min STT/mo | **$0** (if within free) |
| Deepgram (fallback) | $200 credit | Minimal (< 5 min) | **$0** |
| Twilio | $15 trial credit | ~300 min inbound/outbound/mo (VAD cuts ~50 %) | **<$5** |
| LiveKit (cloud) | Unknown; assume free tier for low usage | < 5k minutes (VAD reduces) | **$0** |
| FAISS / ChromaDB (local) | Self‑hosted | Disk < 1 GB | **$0** |
| **Total Estimated Monthly Cost** | | | **<$5** (well within typical free/trial budgets) |

*If any service exceeds its free tier, the monitoring will alert us early, allowing us to switch to a local fallback (e.g., Whisper for STT, Coqui/Tortoise for TTS) to keep cost at zero.*

## Critical Files to Modify
- `backend/app/services/groq_service.py` – token counting, validation, LLM abstraction, caching decorator.
- `backend/app/services/sarvam_service.py` – TTS/STT logging, LRU cache for TTS.
- `backend/app/services/vector_store.py` & new `backend/app/services/vector_store_faiss.py` – configurable `k`, backend factory.
- `backend/app/services/voice_session.py` – audio resampling to 8 kHz, VAD integration.
- `backend/app/config.py` – env vars for `VECTOR_STORE_BACKEND`, `VECTOR_TOP_K`, validation toggles, cost constants.
- `backend/app/main.py` – add `/metrics` route.
- `backend/app/metrics.py` – Prometheus counters.
- `backend/app/utils/language_detect.py` – unchanged but may benefit from caching.
- Migration script: `scripts/migrate_chroma_to_faiss.py`.
- Cache invalidation hook: watch `backend/data/knowledge_base/combined_kb.json`.

## Verification & Demo Plan
1. **Unit Tests** – Ensure all existing functionality passes after each phase.
2. **Latency Benchmark** – Measure end‑to‑end time for 10 Bengali, 10 Hindi, 10 English queries (baseline vs. after each phase).
3. **Correctness / Hallucination Test** – Run a KB‑based question set (~30 items) per language; verify that answers are either verbatim from KB or the safe fallback.
4. **Audio Quality** – Conduct an internal MOS test (5‑point scale) on sample outputs; target ≥ 4.0.
5. **Usage Dashboard** – Hit `/metrics` during a 5‑minute simulated load; confirm counters stay low.
6. **Free‑Tier Soak Test** – Run the agent in a staging environment with only free/trial keys (or local fallbacks) for 2 hours; ensure no 429/errors and latency stays ≤ 0.6 s.

## Success Criteria for Stakeholder Demo
- **Latency** ≤ 0.6 s (down from ~1.2‑1.4 s).
- **Bengali/Hindi correctness** ≥ 95 % (hallucination‑free; any out‑of‑KB query falls back politely).
- **Audio MOS** ≥ 4.0/5 (clear, natural‑sounding speech).
- **Zero cost** – all services within free/trial limits or replaced by local fallbacks.
- **Fallback to human agent** ≤ 5 % (estimated from improved correctness and clarity).

By delivering a voice agent that is **clear, always correct, fast, and free to run**, we can demonstrate immediate value and confidently request a budget for scaling to production or adding advanced features (outbound campaigns, analytics dashboard, etc.).