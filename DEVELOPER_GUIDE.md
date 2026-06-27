# Developer Guide - College Voice Agent

Technical details for developers working on the BCREC Voice Agent.

---

## Architecture Overview

- **Frontend:** React SPA (audio capture/playback, text chat, streaming)
- **Backend:** FastAPI server (LLM orchestration, voice services, admin API)
- **Knowledge Base:** `combined_kb.json` — single source of truth for FAQ + context

---

## FAQ System (Auto-Discovery)

The FAQ system in `_try_faq()` (`groq_service.py:519`) automatically discovers all `voice_ready_answers` entries.

### Adding a New FAQ Topic

Just add a JSON block to `voice_ready_answers` in `combined_kb.json`:

```json
"your_topic": {
  "keywords": ["trigger1", "trigger2", "ट्रिगर"],
  "answers": {
    "en": "English answer...",
    "hi": "Hindi answer...",
    "bn": "Bengali answer..."
  }
}
```

Then reload via Admin API or restart the server. **Zero Python code changes.**

### Priority Ordering

The code checks entries in a specific priority order:
1. `vice_principal` (checked before principal)
2. `principal`
3. `hod` (with sub-answer department routing)
4. `hidden_charges`, `refund_policy` (checked before general fees)
5. `fees` (with branch-specific sub-answers)
6. `international`, `application_status`, `why_bcrec`, `campus`, `policies`
7. All remaining entries auto-discovered

If adding a new broad-category entry (like `admission`), make sure more specific entries are checked first so they don't get shadowed.

### Admin API (Hot-Reload)

All KB mutations are available through the admin API:

```bash
# Get token
curl -X POST http://localhost:8080/token -d "username=admin&password=admin"

# List all FAQs
curl http://localhost:8080/admin/kb/faq -H "Authorization: Bearer <token>"

# Add/update an entry
curl -X POST http://localhost:8080/admin/kb/faq/campus \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"keywords":["campus"],"answers":{"en":"...","hi":"...","bn":"..."}}'

# Delete an entry
curl -X DELETE http://localhost:8080/admin/kb/faq/campus \
  -H "Authorization: Bearer <token>"

# Force reload from disk
curl -X POST http://localhost:8080/admin/kb/reload \
  -H "Authorization: Bearer <token>"
```

---

## API Keys

- `GROQ_API_KEY`: LLM inference (Groq Cloud)
- `SARVAM_API_KEY`: STT and TTS (Sarvam AI)

---

## Code Navigation

### Backend (`backend/`)
- `app/api/qa.py` — QA query endpoint (`POST /qa/query`)
- `app/api/admin.py` — Admin endpoints for KB CRUD
- `app/api/ws_voice.py` — WebSocket voice streaming pipeline
- `app/services/llm/groq_service.py` — LLM, FAQ, caching logic
- `app/services/sarvam_service.py` — Sarvam AI voice API wrapper
- `app/utils/language_detect.py` — Language detection (fasttext or regex fallback)

### Frontend (`frontend/`)
- `src/components/VoiceChat/` — Voice UI with streaming
- `src/hooks/useVoiceWS.ts` — WebSocket hook for streaming pipeline

---

## Testing

### API Testing (UTF-8 Safe)
On Windows, `curl` corrupts Unicode characters. Use Python requests instead:

```python
import requests
r = requests.post("http://localhost:8080/qa/query",
    json={"message": "कैंपस का आकार क्या है"})
print(r.json()["source"])  # "faq_deterministic"
```

Or use `curl` with `--data-binary @file` and a UTF-8 encoded file.

### Running the Server
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## Known Issues

- `fasttext` not installed — romanized Hinglish/Banglish detection is less accurate
- Stale tests in `backend/tests/test_groq_service.py` reference old `_format_knowledge_base`
