# BCREC Demo Status

## THE MISSION
Demonstrate AI voice assistant for college admissions via LiveKit Playground (browser-based voice, no phone needed).

---

## WHAT'S WORKING

### FAQ System (Deterministic, 4-6ms)
- 20 entries: Principal, Vice Principal, HOD (all 12 depts), Hostel, Placement, Documents, Scholarship, Fees (branch-specific), Admission, Courses, Wifi, Faculty, Online Learning, Campus, International, Why BCREC, Policies, Hidden Charges, Refund Policy, Application Status
- All driven from `combined_kb.json` — adding a new topic = just add a JSON block, zero Python code changes
- Admin API for hot-reload (list/add/delete/reload without server restart)
- Auto-discovery: new entries with `keywords` + `answers` are picked up automatically
- Priority ordering: specific entries checked before general ones
- Keyword matching with Hindi/Bengali script support (461 keywords across all entries)
- Verified with 40+ test queries across EN/HI/BN — 100% pass rate

### LLM/RAG (via Groq)
- ChromaDB vector search + JSON precision context
- Hallucination guardrail + out-of-KB phone handoff
- Skip-FAQ heuristic: follow-up queries with no standalone FAQ keywords fall through to LLM

### TTS Normalization (clean_for_voice)
- Single source of truth in `voice_utils.py`
- Expands: Rs.→Rupees, Dr.→Doctor, Prof.→Professor, LPA→Lakhs per annum
- Spaces acronyms: CSE→C. S. E., NAAC→N. A. A. C., WBJEE→W. B. J. E. E.
- Converts numbers to words: 598300→five lakh ninety-eight thousand three hundred
- Indian numbering system: crore/lakh/thousand support
- Bengali & Hindi number-to-words support
- Phone numbers read digit-by-digit

### Voice Agent (LiveKit)
- LiveKit agent worker with Sarvam STT + Groq LLM + Sarvam TTS
- VAD interruption threshold increased (min_words: 5)
- WAV header parsing fixed (proper `data` chunk detection)
- Chunk size increased (320→1600 bytes) for smoother playback
- Speaker map updated: shubh (en/hi), ritu (bn)

### Web Chat (React Frontend)
- Text chat with streaming + voice chat with WebSocket pipeline

---

## DEMO CHECKLIST
- [x] FAQ bypass for common questions (40+ queries verified in EN/HI/BN)
- [x] LLM/RAG for complex queries
- [x] Hallucination guardrail
- [x] Multilingual support (EN/HI/BN) with script-specific keywords
- [x] LiveKit agent worker
- [x] Hot-reload KB via Admin API
- [x] TTS normalization for proper pronunciation
- [x] Cache management (query cache + TTS cache endpoints)
- [x] All 143 tests passing (132 demo readiness + 11 project tests)
- [ ] LiveKit Playground demo walkthrough
- [ ] Production SIP trunk (pending budget)

## Known Issues (Non-Blocking for Demo)
- `fasttext` not installed — Hindi/Bengali detection for romanized text falls back to regex (less accurate for Hinglish/Banglish). Adding `fasttext-wheel` to requirements.
- Vector store init may warn about BAAI/bge-m3 processing class on first run. FAQ works independently of vector store. For LLM fallback, context will be reduced.
- `curl` on Windows corrupts UTF-8 for Hindi/Bengali — use Python `requests` or POST from frontend.

## Demo Scenario (Recommended Flow)
1. Start with English: "Hi" → bot greets
2. "Who is the principal?" → Dr. Sanjay S. Pawar
3. "CSE HOD name?" → Dr. Raj Kumar Samanta
4. "CSE fee?" → ₹5,98,300 breakdown
5. "Hostel fee?" → Optional, ₹30K/₹10K per sem
6. "Placement?" → Placement details
7. Switch to Hindi: "प्रिंसिपल कौन हैं?" → Hindi answer
8. Switch to Bengali: "ভর্তি প্রক্রিয়া কী?" → Bengali answer
9. "एएमएल फीस?" → AIML-specific fee in Hindi
10. "হোস্টেল ফি?" → Bengali hostel fee
