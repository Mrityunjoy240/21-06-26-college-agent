#!/bin/bash
set -e

echo "=== College Voice Agent - Codespace Setup ==="

# Copy env file if not exists
if [ ! -f backend/.env ]; then
    cp .env.example backend/.env
    echo "Created backend/.env from .env.example"
    echo "⚠️  Add your GROQ_API_KEY and SARVAM_API_KEY to backend/.env"
fi

# Copy root env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# Install Python deps
echo "Installing Python dependencies..."
cd backend
pip install -r requirements.txt -q
cd ..

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Add API keys to backend/.env (GROQ_API_KEY, SARVAM_API_KEY)"
echo "  2. Generate OSS_JWT_SECRET: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
echo "  3. Start: docker compose up -d"
echo "  4. Open http://localhost:3010 for Dograh"
echo ""
echo "To test the backend alone (no Docker):"
echo "  cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
