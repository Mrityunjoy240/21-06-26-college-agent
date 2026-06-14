import asyncio
import time
import os
import sys

# Add backend to path so we can import services directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.config import settings
from app.services.llm.groq_service import get_groq_service
from app.utils.language_detect import detect_language
import re

# Mock Lexicon for testing TTS input formatting
LEXICON = {
    "BCREC": "B. C. R. E. C.",
    "MAKAUT": "Ma-Kaut",
    "AIML": "A. I. M. L.",
    "AML": "A. I. M. L.",
    "CSE": "C. S. E.",
    "ECE": "E. C. E.",
    "IT": "I. T.",
    "B.Tech": "B. Tech",
    "M.Tech": "M. Tech",
    "WBJEE": "W. B. J. E. E.",
    "JEE": "J. E. E."
}

def apply_lexicon(text: str) -> str:
    processed = text
    processed = re.sub(r"\bAML\b", "AIML", processed, flags=re.IGNORECASE)
    for word, phonetic in LEXICON.items():
        processed = re.sub(rf"\b{re.escape(word)}\b", phonetic, processed)
    return processed

async def run_tests():
    print("--- STARTING AUTOMATED PIPELINE TEST ---")
    groq_service = get_groq_service()
    
    test_questions = [
        ("English - Departments", "What are the departments or admission options available?"),
        ("Hindi - Departments", "कौन कौन से डिपार्टमेंट्स या एडमिशन ऑप्शंस हैं?"),
        ("Bengali - Departments", "কলেজে কি কি ডিপার্টমেন্ট বা অ্যাডমিশন অপশন আছে?"),
        ("English - Entity", "Who is the principal of the college?"),
    ]
    
    for category, question in test_questions:
        print(f"\n[{category}] Question: {question}")
        start_time = time.time()
        
        # 1. Detect Language
        lang = detect_language(question)
        print(f"Detected Lang: {lang}")
        
        # 2. Get LLM Response
        # We simulate the exact call made in LiveKit
        response_data = await groq_service.generate_response(question, [])
        answer = response_data.get("answer", "")
        latency = response_data.get("latency_ms", 0)
        
        # 3. Apply Lexicon (simulate TTS pre-processing)
        tts_ready_text = apply_lexicon(answer)
        
        # 4. Calculate sentence chunks to check if TTS will break
        sentence_ends = r'(?<!\bDr\.)(?<!\bProf\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bRs\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+'
        sentences = [s.strip() for s in re.split(sentence_ends, tts_ready_text) if s.strip()]
        
        print(f"Time Taken: {latency}ms")
        print(f"Raw Output: {answer}")
        print(f"TTS Chunks ({len(sentences)}):")
        for i, s in enumerate(sentences):
            print(f"  [{i+1}] {s}")

if __name__ == "__main__":
    asyncio.run(run_tests())
