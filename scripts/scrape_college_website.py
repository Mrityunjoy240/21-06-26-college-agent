import os
import re
import time
import urllib.parse
from collections import deque
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Configuration
START_URL = "https://bcrec.ac.in"
MAX_PAGES = 100
MAX_DEPTH = 3
DELAY_BETWEEN_REQUESTS = 0.5  # Polite delay

# Target keywords to prioritize pages
RELEVANT_KEYWORDS = [
    "admission", "placement", "fee", "department", "syllabus", 
    "about", "contact", "course", "leadership", "principal", 
    "hod", "administration", "academic"
]

# Output files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "backend", "data", "knowledge_base", "topics")
WEBSITE_OUTPUT_FILE = os.path.join(KB_DIR, "scraped_website.md")
PDF_OUTPUT_FILE = os.path.join(KB_DIR, "scraped_pdfs.md")

# Ensure KB directory exists
os.makedirs(KB_DIR, exist_ok=True)


def is_valid_url(url, base_domain):
    parsed = urllib.parse.urlparse(url)
    # Stay within domain
    domain_match = parsed.netloc == base_domain or parsed.netloc.endswith("." + base_domain)
    # Avoid query params, hashes, and static resources except pdf
    if not domain_match:
        return False
    
    path = parsed.path.lower()
    # Skip standard non-html assets (images, fonts, styles)
    if any(path.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".css", ".js", ".png", ".woff", ".ttf", ".svg"]):
        return False
    
    return True


def clean_html_content(soup):
    # Remove script, style, nav, footer, and header elements
    for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        element.decompose()
    
    # Extract structural text
    content_lines = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        tag = element.name
        text = element.get_text(strip=True)
        if not text:
            continue
            
        if tag.startswith("h"):
            level = int(tag[1])
            content_lines.append(f"\n{'#' * level} {text}\n")
        elif tag == "li":
            content_lines.append(f"- {text}")
        elif tag == "tr":
            cols = [td.get_text(strip=True) for td in element.find_all(["td", "th"])]
            if cols:
                content_lines.append("| " + " | ".join(cols) + " |")
        else:
            content_lines.append(f"\n{text}\n")
            
    return "\n".join(content_lines)


def scrape_pdf(pdf_url):
    print(f"📄 Scraping and downloading PDF: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            # Create downloaded_docs folder in workspace root
            docs_dir = os.path.join(BASE_DIR, "downloaded_docs")
            os.makedirs(docs_dir, exist_ok=True)
            
            # Parse clean filename
            filename = urllib.parse.unquote(pdf_url.split("/")[-1])
            if not filename.lower().endswith(".pdf"):
                filename = "document.pdf"
                
            pdf_path = os.path.join(docs_dir, filename)
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            
            # Extract text
            reader = PdfReader(pdf_path)
            extracted_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(f"--- PDF Page {i+1} ---\n{page_text}")
                
            return "\n\n".join(extracted_text)
    except Exception as e:
        print(f"❌ Failed to scrape PDF {pdf_url}: {e}")
    return None


def main():
    base_domain = urllib.parse.urlparse(START_URL).netloc
    
    # (url, depth) queue
    queue = deque([(START_URL, 0)])
    visited_urls = set([START_URL])
    crawled_count = 0
    
    scraped_pages = []
    scraped_pdfs = []
    
    print("🚀 Starting website and PDF scraper...")
    
    while queue and crawled_count < MAX_PAGES:
        current_url, depth = queue.popleft()
        
        if depth > MAX_DEPTH:
            continue
            
        print(f"🔍 Crawling [{crawled_count + 1}/{MAX_PAGES}] (Depth {depth}): {current_url}")
        
        try:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            response = requests.get(current_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract page data
            page_title = soup.title.string.strip() if soup.title else current_url
            page_markdown = clean_html_content(soup)
            
            scraped_pages.append(
                f"# URL: {current_url}\n## Title: {page_title}\n\n{page_markdown}\n\n---\n"
            )
            crawled_count += 1
            
            # Extract links
            for link in soup.find_all("a", href=True):
                href = link["href"].strip()
                joined_url = urllib.parse.urljoin(current_url, href)
                # Parse to remove fragments
                joined_url = urllib.parse.urlunparse(
                    urllib.parse.urlparse(joined_url)._replace(fragment="")
                )
                
                # Check if it is a PDF
                if joined_url.lower().endswith(".pdf") and joined_url not in visited_urls:
                    visited_urls.add(joined_url)
                    pdf_text = scrape_pdf(joined_url)
                    if pdf_text:
                        scraped_pdfs.append(
                            f"# PDF URL: {joined_url}\n\n{pdf_text}\n\n---\n"
                        )
                
                # Check if it is a valid subpage to crawl
                elif is_valid_url(joined_url, base_domain) and joined_url not in visited_urls:
                    # Prioritize urls with relevant keywords
                    path_lower = urllib.parse.urlparse(joined_url).path.lower()
                    is_prioritized = any(kw in path_lower for kw in RELEVANT_KEYWORDS)
                    
                    visited_urls.add(joined_url)
                    
                    if is_prioritized:
                        queue.appendleft((joined_url, depth + 1))  # Push to front to crawl first
                    else:
                        queue.append((joined_url, depth + 1))      # Push to back
                        
        except Exception as e:
            print(f"❌ Failed to crawl {current_url}: {e}")
            
    # Write scraped pages to KB
    if scraped_pages:
        print(f"💾 Saving scraped website pages to {WEBSITE_OUTPUT_FILE}...")
        with open(WEBSITE_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# Scraped Website Data\n\n")
            f.write("\n\n".join(scraped_pages))
            
    # Write scraped PDFs to KB
    if scraped_pdfs:
        print(f"💾 Saving scraped PDF contents to {PDF_OUTPUT_FILE}...")
        with open(PDF_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# Scraped PDF Document Contents\n\n")
            f.write("\n\n".join(scraped_pdfs))
            
    print("✨ Scraper run finished.")


if __name__ == "__main__":
    main()
