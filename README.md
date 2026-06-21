# College Voice Agent

AI-powered multilingual chatbot + telephony voice agent for college admissions.

## Architecture

- **Backend**: FastAPI (Python) with RAG pipeline (Groq LLM + ChromaDB + BGE-M3 embeddings)
- **Multilingual**: English, Hindi, Bengali (FastText detection + script-based)
- **TTS/STT**: Sarvam AI (Bulbul v3 TTS, Saaras v3 STT)
- **Telephony**: Dograh (Twilio/Vonage integration)
- **Frontend**: React + TypeScript + Material-UI

## Quick Start

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add API keys
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

- `POST /qa/query` - Ask a question (text)
- `POST /qa/query-stream` - Streaming response (SSE)
- `POST /qa/tts-direct` - Text to speech
- `POST /qa/stt` - Speech to text
- `GET /api/conversations` - Conversation history
- `POST /admin/upload` - Upload KB documents
