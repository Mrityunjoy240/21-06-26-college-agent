import os
import sys
import logging
import asyncio
from pathlib import Path

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from langchain_core.documents import Document
from app.services.vector_store import get_vector_store
from app.services.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingest-kb")

async def ingest_kb():
    vector_store = get_vector_store()
    doc_processor = DocumentProcessor()
    
    # Clear existing to ensure fresh index
    logger.info("Clearing existing vector store...")
    vector_store.clear_collection()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    kb_dir = project_root / "backend" / "data" / "knowledge_base"
    uploads_dir = project_root / "uploads"
    
    all_docs = []
    
    # 1. Process Markdown Files from departments/ and topics/
    # (This is now the ONLY knowledge source — combined_kb.json is retired)
    markdown_dirs = [kb_dir / "departments", kb_dir / "topics"]
    for mdir in markdown_dirs:
        if not mdir.exists():
            continue
            
        logger.info(f"Processing markdown files in {mdir}...")
        for md_file in mdir.glob("*.md"):
            chunks = await doc_processor.process_file(str(md_file))
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": chunk["source"],
                        **chunk.get("metadata", {})
                    }
                ))
    
    # 2. Re-ingest any admin-uploaded files from uploads/ directory
    # (so they aren't silently lost after a reindex)
    if uploads_dir.exists():
        logger.info(f"Processing uploaded files in {uploads_dir}...")
        for upload_file in uploads_dir.iterdir():
            if upload_file.is_file() and upload_file.suffix.lower() in {
                ".pdf", ".txt", ".csv", ".xlsx", ".xls", ".md"
            }:
                try:
                    chunks = await doc_processor.process_file(str(upload_file))
                    for chunk in chunks:
                        all_docs.append(Document(
                            page_content=chunk["text"],
                            metadata={
                                "source": chunk["source"],
                                **chunk.get("metadata", {})
                            }
                        ))
                except Exception as e:
                    logger.error(f"Error processing upload {upload_file.name}: {e}")

    # 3. Add to Vector Store
    if all_docs:
        logger.info(f"Adding {len(all_docs)} document chunks to Vector Store...")
        # Add in batches to avoid overwhelming the system
        batch_size = 50
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Ingested batch {i//batch_size + 1}/{(len(all_docs)-1)//batch_size + 1}")
            
        logger.info("Knowledge Base ingestion COMPLETE.")
    else:
        logger.warning("No documents found to ingest!")

if __name__ == "__main__":
    asyncio.run(ingest_kb())
