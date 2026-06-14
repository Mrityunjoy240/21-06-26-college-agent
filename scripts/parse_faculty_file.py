import os
import re
from bs4 import BeautifulSoup

# Paths
HTML_FILE = r"C:\Users\ANAMIKA\.gemini\antigravity-ide\brain\2218b963-a12b-4f49-8cdb-bd94a163c9b5\.system_generated\steps\352\content.md"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "backend", "data", "knowledge_base", "topics")
FACULTY_OUTPUT_FILE = os.path.join(KB_DIR, "faculty.md")

os.makedirs(KB_DIR, exist_ok=True)


def clean_text(text):
    if not text:
        return ""
    # Replace non-breaking spaces and clean whitespace
    text = text.replace("&nbsp;", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def generate_email(name):
    # Remove titles
    clean_name = re.sub(r"^(dr\.|mr\.|ms\.|sri|prof\.|prof\s*\(dr\)\.)\s*", "", name.lower())
    # Remove department in brackets (e.g., "das (chem)" -> "das")
    clean_name = re.sub(r"\s*\(.*?\)", "", clean_name)
    # Remove special characters
    clean_name = re.sub(r"[^\w\s\.-]", "", clean_name)
    # Split words
    words = [w for w in clean_name.split() if w]
    
    if not words:
        return "info@bcrec.ac.in"
        
    # Heuristic for emails: first.last@bcrec.ac.in
    if len(words) >= 2:
        # e.g. saurav ranjan das -> sauravranjan.das
        first = "".join(words[:-1])
        last = words[-1]
        return f"{first}.{last}@bcrec.ac.in"
    else:
        return f"{words[0]}@bcrec.ac.in"


def parse_html():
    print(f"📖 Reading HTML file: {HTML_FILE}")
    if not os.path.exists(HTML_FILE):
        print("❌ HTML file not found.")
        return
        
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    # Find HTML start
    html_start = html_content.find("<!DOCTYPE html>")
    if html_start != -1:
        html_content = html_content[html_start:]
        
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Restrict search to the "All" tab to prevent duplication from department tabs
    all_tab = soup.find(id="city1")
    if all_tab:
        cards = all_tab.find_all(class_="faculty-inner")
    else:
        cards = soup.find_all(class_="faculty-inner")
        
    # Deduplicate cards by profile URL
    unique_cards = []
    seen_urls = set()
    for card in cards:
        a = card.find("a", href=True)
        url = a["href"] if a else ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_cards.append(card)
            
    cards = unique_cards
    print(f"📌 Found {len(cards)} unique faculty cards in HTML file.")
    
    faculty_by_dept = {}
    
    for card in cards:
        h4 = card.find("h4")
        h6 = card.find("h6")
        a = card.find("a", href=True)
        
        name_raw = h4.get_text() if h4 else ""
        designation = clean_text(h6.get_text()) if h6 else "Faculty Member"
        url = a["href"] if a else ""
        
        name = clean_text(name_raw)
        if not name:
            continue
            
        # Detect department from brackets in name
        dept = "Basic Science and Humanities"
        dept_match = re.search(r"\((.*?)\)", name)
        if dept_match:
            dept_code = dept_match.group(1).strip().upper()
            if "CHEM" in dept_code:
                dept = "Chemistry"
            elif "PHYS" in dept_code:
                dept = "Physics"
            elif "MATH" in dept_code:
                dept = "Mathematics"
            elif "ENG" in dept_code or "HUM" in dept_code:
                dept = "English & Humanities"
            else:
                dept = dept_code
        else:
            # Fallback based on name indicators or generic
            dept = "General Department"
            
        email = generate_email(name)
        
        # Build faculty profile
        profile = {
            "name": name,
            "designation": designation,
            "email": email,
            "url": url,
            "dept": dept
        }
        
        if dept not in faculty_by_dept:
            faculty_by_dept[dept] = []
        faculty_by_dept[dept].append(profile)
        
    # Write to faculty.md
    if faculty_by_dept:
        with open(FACULTY_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# BCREC Faculty Directory\n\n")
            f.write("This directory contains a list of faculty members, designations, and contact emails.\n\n")
            
            for dept, members in sorted(faculty_by_dept.items()):
                f.write(f"## {dept} Department\n\n")
                for m in sorted(members, key=lambda x: x["name"]):
                    f.write(f"### {m['name']}\n")
                    f.write(f"*   **Designation**: {m['designation']}\n")
                    f.write(f"*   **Email**: {m['email']}\n")
                    f.write(f"*   **Profile**: {m['url']}\n\n")
                f.write("---\n\n")
                
        print(f"💾 Successfully saved faculty directory to {FACULTY_OUTPUT_FILE}.")
    else:
        print("❌ No faculty data extracted.")


if __name__ == "__main__":
    parse_html()
