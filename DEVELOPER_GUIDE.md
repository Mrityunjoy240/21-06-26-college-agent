# 🛠️ Developer Guide - College Voice Agent

This guide provides technical details for developers working on the BCREC Voice Agent.

---

## 🏗️ Architecture Overview

The system follows a decoupled Frontend-Backend architecture:

-   **Frontend:** React SPA that handles audio capture and playback.
-   **Backend:** FastAPI server that orchestrates LLM and Voice services.
-   **Knowledge Base:** A structured JSON file (`combined_kb.json`) that acts as the source of truth.

---

## 🚀 Environment Setup

### 1. Requirements
-   Python 3.10+
-   Node.js 18+
-   Docker (Optional)

### 2. API Keys Needed
-   `GROQ_API_KEY`: For LLM inference ([Groq Cloud](https://console.groq.com/))
-   `SARVAM_API_KEY`: For STT and TTS ([Sarvam AI](https://www.sarvam.ai/))

### 3. Backend Configuration (`backend/.env`)
```env
GROQ_API_KEY=your_key
SARVAM_API_KEY=your_key
CORS_ORIGINS=["http://localhost:5173"]
```

---

## 📂 Code Navigation

### Backend (`/backend`)
-   `app/api/`: Endpoint definitions (QA, STT, TTS, Admin).
-   `app/services/`: Core logic for AI services.
    -   `llm/groq_service.py`: Handles prompt engineering and LLM calls.
    -   `sarvam_service.py`: Wrapper for Sarvam AI APIs.
-   `data/knowledge_base/`: JSON/Markdown files containing college data.
-   `uploads/`: Directory for dynamic document uploads.

### Frontend (`/frontend`)
-   `src/components/Chatbot/`: Main UI for interaction.
-   `src/components/VoiceChat/`: Specialized voice-only interface.
-   `src/hooks/useVoice.ts`: Custom hook for Web Speech API management.

---

## 🔧 Core Workflows

### Updating the Knowledge Base
1.  Navigate to `backend/data/knowledge_base/`.
2.  Update `combined_kb.json` with new information.
3.  The backend automatically reloads the KB on every request (in the current prototype).

### Adding New Voice Models
1.  Check `backend/app/api/tts.py` and `stt.py`.
2.  Update the `language_speaker_map` to include new voice profiles from Sarvam.

---

## 🧪 Testing

### API Testing
Use the provided `curl` commands or Postman:
```bash
# Test LLM
curl -X POST http://localhost:8000/qa/groq-query -H "Content-Type: application/json" -d '{"message": "What is the CSE fee?"}'
```

### Voice Testing
Use the Admin Dashboard to record and test STT/TTS latency.

---

## 🚢 Deployment

### Using Docker
```bash
docker-compose up --build
```

### Manual Deployment
1.  **Backend:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
2.  **Frontend:** `npm run build` and serve `dist/` via Nginx.
