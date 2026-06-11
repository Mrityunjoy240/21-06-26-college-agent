import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Professional Vector Store Service using ChromaDB and BGE-M3 embeddings.
    Handles semantic search for the RAG pipeline.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        
        # Use absolute path anchored to project root to avoid CWD-dependent bugs
        # vector_store.py lives at: backend/app/services/vector_store.py
        # .parent x4 = project root
        _project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.db_path = str(_project_root / "chroma_db" / "chroma_db_v2")
        os.makedirs(self.db_path, exist_ok=True)
        logger.info(f"ChromaDB path: {self.db_path}")
        
        # Using BGE-M3 for superior multilingual performance (English/Hindi/Bengali)
        # It's an industry standard for cross-lingual RAG in 2025.
        self.embedding_model_name = "BAAI/bge-m3"
        
        logger.info(f"Initializing VectorStoreService with model: {self.embedding_model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}, # Use 'cuda' if GPU available, but CPU is safer for general env
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self._collection_name = "bcrec_knowledge_base"
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=self._collection_name
            )
            
            self.initialized = True
            logger.info("VectorStoreService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreService: {e}")
            self.vector_store = None

    def search(self, query: str, k: int = 8) -> List[Document]:
        """
        Perform Hybrid Search: Semantic Similarity + Keyword Boosting.
        Ensures high-intent topics like 'fees' or 'hostel' are always retrieved.
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized. Returning empty results.")
            return []
            
        try:
            # 1. Semantic Search via Chroma
            results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
            
            # Filter by relevance threshold
            RELEVANCE_THRESHOLD = 0.3
            filtered = [
                (doc, score) for doc, score in results_with_scores
                if score >= RELEVANCE_THRESHOLD
            ]
            
            # 2. Keyword Booster (Simple BM25-lite)
            # If query contains high-intent words, try to fetch specific docs
            boosted_docs = []
            q_lower = query.lower()
            
            # Map keywords to specific topics for 100% recall
            INTENT_MAP = {
                "fees": ["fee", "payment", "scholarship", "cost", "rupees", "lakh"],
                "hostel": ["hostel", "mess", "boarding", "accommodation"],
                "contact": ["phone", "email", "contact", "address", "hod"],
                "admission": ["admission", "eligibility", "wbjee", "jee"]
            }
            
            for intent, keywords in INTENT_MAP.items():
                if any(kw in q_lower for kw in keywords):
                    # Direct query for the intent to ensure it's in the mix
                    intent_results = self.vector_store.similarity_search(intent, k=2)
                    boosted_docs.extend(intent_results)

            # 3. Merge & De-duplicate
            final_docs = [doc for doc, score in filtered]
            
            # Add boosted docs to the front (priority)
            seen_content = {doc.page_content[:100] for doc in final_docs}
            for b_doc in boosted_docs:
                short_content = b_doc.page_content[:100]
                if short_content not in seen_content:
                    final_docs.insert(0, b_doc)
                    seen_content.add(short_content)

            # 4. Fallback if still empty
            if not final_docs and results_with_scores:
                logger.warning("Threshold fallback active.")
                return [doc for doc, score in results_with_scores[:3]]
            
            return final_docs[:k] # Keep to requested size
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.
        """
        if not self.vector_store:
            logger.error("Vector store not initialized.")
            return
            
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")

    def clear_collection(self):
        """
        Clear all documents from the collection WITHOUT dropping/recreating it.
        This preserves all existing references to the singleton, preventing
        the stale-reference bug where other services hold a dead pointer.
        """
        if not self.vector_store:
            return
            
        try:
            # Get the underlying Chroma collection and delete all docs
            collection = self.vector_store._collection
            # Get count first
            count = collection.count()
            if count > 0:
                # Fetch all IDs and delete them
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
            logger.info(f"Vector store collection cleared ({count} documents removed).")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            # Fallback: recreate collection (old behavior)
            try:
                self.vector_store.delete_collection()
                self.vector_store = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings,
                    collection_name=self._collection_name
                )
                logger.info("Collection cleared via fallback (recreate).")
            except Exception as e2:
                logger.error(f"Fallback clear also failed: {e2}")

# Singleton getter
def get_vector_store():
    return VectorStoreService()
