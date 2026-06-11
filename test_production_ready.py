import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.services.llm.groq_service import get_groq_service

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("stress-test")

TEST_SCENARIOS = [
    # Topic, Query (EN), Query (HI), Query (BN)
    ("Principal", "Who is the principal?", "प्रिंसिपल कौन है?", "প্রিন্সিপাল কে?"),
    ("Vice Principal", "Who is the vice principal?", "वाइस प्रिंसिपल कौन है?", "ভাইস প্রিন্সিপাল কে?"),
    ("Fees", "What are the fees for B.Tech CSE?", "बीटेक सीएसई की फीस क्या है?", "বিটেক সিএসইর ফিস কত?"),
    ("Fees", "What is the hostel fee?", "हॉस्टल की फीस क्या है?", "হস্টেল ফিস কত?"),
    ("Admission", "How to get admission in B.Tech?", "बीटेक में एडमिशन कैसे लें?", "বিটেকে অ্যাডমিশন কিভাবে নেব?"),
    ("Placement", "What is the highest package?", "सबसे ज्यादा पैकेज क्या है?", "সবচেয়ে বেশি প্যাকেজ কত?"),
    ("Contact", "What is the college phone number?", "कॉलेज का फोन नंबर क्या है?", "কলেজের ফোন নম্বর কি?"),
    ("Location", "Where is the college located?", "कॉलेज कहाँ स्थित है?", "কলেজ কোথায় অবস্থিত?"),
]

async def run_test():
    service = get_groq_service()
    results = []
    
    print(f"{'TOPIC':<15} | {'LANG':<5} | {'STATUS':<8} | {'RESPONSE'}")
    print("-" * 100)
    
    for topic, q_en, q_hi, q_bn in TEST_SCENARIOS:
        for lang, query in [("en", q_en), ("hi", q_hi), ("bn", q_bn)]:
            try:
                # We use generate_response to simulate the full RAG pipeline
                res = await service.generate_response(query)
                answer = res.get("answer", "")
                
                # Basic Verification Logic
                status = "🟢 PASS"
                
                # Multilingual Name Checks
                principal_names = ["Sanjay", "संजय", "সঞ্জয়"]
                vp_names = ["Hossain", "होसैन", "হোসেন"]
                
                if topic == "Principal" and not any(n in answer for n in principal_names):
                    status = f"🔴 FAIL (Wrong Name: {answer[:30]})"
                elif topic == "Vice Principal" and not any(n in answer for n in vp_names):
                    status = f"🔴 FAIL (Wrong Name: {answer[:30]})"
                elif topic == "Fees" and any(x in query.lower() for x in ["cse", "বিটেক"]):
                    # Check for words instead of digits
                    words_en = ["lakh", "rupees"]
                    words_hi = ["लाख", "रुपये"]
                    words_bn = ["লাখ", "টাকা"]
                    
                    if lang == "en" and not all(w in answer.lower() for w in words_en):
                        status = "🟡 WARN (Words missing in EN)"
                    elif lang == "hi" and not all(w in answer for w in words_hi):
                        status = "🟡 WARN (Words missing in HI)"
                    elif lang == "bn" and not all(w in answer for w in words_bn):
                        status = "🟡 WARN (Words missing in BN)"
                    
                    if "5" in answer and "." in answer: # Detect "5.98" style
                        status = "🔴 FAIL (Used digits instead of words)"
                
                print(f"{topic:<15} | {lang:<5} | {status:<8} | {answer[:70]}...")
                results.append({"topic": topic, "lang": lang, "status": status, "answer": answer})
                
            except Exception as e:
                print(f"{topic:<15} | {lang:<5} | 🔴 ERROR | {str(e)}")

    print("-" * 100)
    failed = [r for r in results if "🔴" in r["status"]]
    if not failed:
        print("✅ ALL TESTS PASSED! THE BRAIN IS STABLE.")
    else:
        print(f"❌ {len(failed)} TESTS FAILED. INVESTIGATING...")

if __name__ == "__main__":
    asyncio.run(run_test())
