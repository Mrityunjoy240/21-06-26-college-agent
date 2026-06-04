import asyncio
import os
import sys
import json
import logging

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from dotenv import load_dotenv
load_dotenv("backend/.env", override=True)

from app.services.llm.groq_service import get_groq_service

# Mocking settings if needed, but GroqService uses app.config.settings
# Ensure GROQ_API_KEY is set in environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quality-test")

test_cases = [
    # ENGLISH CASES
    {
        "name": "EN-Basic-Greeting",
        "query": "Hello",
        "lang": "en",
        "expected_checks": ["greeting", "concise"]
    },
    {
        "name": "EN-Fees-BTech",
        "query": "What is the total fee for B.Tech?",
        "lang": "en",
        "expected_checks": ["Rs.", "6.08 Lakhs", "number words", "phonetic"]
    },
    {
        "name": "EN-Placement-Highest",
        "query": "Tell me about the highest placement package.",
        "lang": "en",
        "expected_checks": ["30 LPA", "Cyberwissen", "concise"]
    },
    {
        "name": "EN-Departments-Phonetic",
        "query": "Which departments do you have?",
        "lang": "en",
        "expected_checks": ["C S E", "A I M L", "E C E"]
    },
    {
        "name": "EN-Hostel-Rules",
        "query": "What are the hostel rules for boys?",
        "lang": "en",
        "expected_checks": ["curfew", "concise"]
    },
    
    # HINDI CASES
    {
        "name": "HI-Basic-Greeting",
        "query": "नमस्ते",
        "lang": "hi",
        "expected_checks": ["Devanagari script", "concise"]
    },
    {
        "name": "HI-Fees-BTech",
        "query": "बीटेक की फीस कितनी है?",
        "lang": "hi",
        "expected_checks": ["Devanagari script", "Rs.", "6.08 Lakhs"]
    },
    {
        "name": "HI-Placement",
        "query": "प्लेसमेंट कैसा है?",
        "lang": "hi",
        "expected_checks": ["Devanagari script", "80%"]
    },
    
    # BENGALI CASES
    {
        "name": "BN-Basic-Greeting",
        "query": "হ্যালো",
        "lang": "bn",
        "expected_checks": ["Bengali script", "concise"]
    },
    {
        "name": "BN-Fees-BTech",
        "query": "B.Tech এর ফিস কত?",
        "lang": "bn",
        "expected_checks": ["Bengali script", "টাকা", "সি এস ই"]
    },
    {
        "name": "BN-Placement",
        "query": "প্লেসমেন্ট কেমন হয় এখানে?",
        "lang": "bn",
        "expected_checks": ["Bengali script", "৮০%"]
    },
    
    # MIXED LANGUAGE / EDGE CASES
    {
        "name": "MIX-Hinglish-Admission",
        "query": "Admission process kya hai?",
        "lang": "hi",
        "expected_checks": ["Devanagari script"]
    },
    {
        "name": "MIX-Benglish-Fees",
        "query": "Fees koto lagbe?",
        "lang": "bn",
        "expected_checks": ["Bengali script"]
    },
    {
        "name": "EDGE-Out-Of-Scope",
        "query": "How is the weather in Durgapur today?",
        "lang": "en",
        "expected_checks": ["concise", "polite refusal or general info"]
    },
    {
        "name": "EDGE-Long-Query",
        "query": "I am interested in taking admission in your college for computer science engineering but I want to know about the placement records of the last three years and also if I can get any scholarship based on my WBJEE rank which is 15000.",
        "lang": "en",
        "expected_checks": ["concise", "C S E", "scholarship"]
    }
]

def analyze_response(case, response):
    name = case["name"]
    answer = response["answer"]
    voice_text = response["voice_text"]
    
    issues = []
    
    # Check Brevity (Sentences)
    sentences = [s for s in answer.split('.') if s.strip()]
    if len(sentences) > 2:
        issues.append(f"Too long: {len(sentences)} sentences.")
    
    # Check Script
    if case["lang"] == "hi":
        if not any('\u0900' <= c <= '\u097F' for c in answer):
            issues.append("Expected Hindi (Devanagari) script but got something else.")
    elif case["lang"] == "bn":
        if not any('\u0980' <= c <= '\u09FF' for c in answer):
            issues.append("Expected Bengali script but got something else.")
            
    # Check Phonetics in voice_text
    if case["lang"] == "en":
        for ac in ["CSE", "AIML", "ECE"]:
            if ac in voice_text and " " not in ac: # This is a simple check, LLM might expand it to "C S E"
                # Wait, the LLM should expand it or _clean_for_voice should.
                # Actually, _clean_for_voice converts "CSE" to "C S E"
                if ac in voice_text.replace(" ", ""):
                    # check if it is expanded
                    expanded = " ".join(list(ac))
                    if expanded not in voice_text:
                        issues.append(f"Acronym {ac} not expanded in voice_text.")

    # Check Currency in Bengali
    if case["lang"] == "bn":
        if "রুপি" in answer:
            issues.append("Used 'রুপি' instead of 'টাকা' in Bengali.")

    return issues

async def run_tests():
    groq_service = get_groq_service()
    results = []
    
    print(f"{'Name':<25} | {'Lang':<5} | {'Status':<10} | {'Issues'}")
    print("-" * 80)
    
    for case in test_cases:
        try:
            response = await groq_service.generate_response(case["query"])
            issues = analyze_response(case, response)
            
            status = "PASS" if not issues else "FAIL"
            print(f"{case['name']:<25} | {case['lang']:<5} | {status:<10} | {', '.join(issues)}")
            
            results.append({
                "case": case["name"],
                "query": case["query"],
                "answer": response["answer"],
                "voice_text": response["voice_text"],
                "status": status,
                "issues": issues
            })
        except Exception as e:
            print(f"{case['name']:<25} | {case['lang']:<5} | ERROR      | {str(e)}")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(run_tests())
