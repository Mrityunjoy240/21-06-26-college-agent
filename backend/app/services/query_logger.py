"""
Query Logger for BCREC Voice Agent

Structured JSON-lines logger that records every query, its context,
the LLM response, and timing. Replaces the manual QA reports.

Log file: logs/queries.jsonl
Each line is a self-contained JSON object.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "queries.jsonl"


class QueryLogger:
    """Append-only structured query logger."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        os.makedirs(_LOG_DIR, exist_ok=True)
        self._initialized = True
        logger.info(f"QueryLogger writing to {_LOG_FILE}")

    def log(
        self,
        query: str,
        language: str,
        context_chunks: List[str],
        response: str,
        model: str,
        source: str = "groq_rag",
        latency_ms: Optional[float] = None,
        json_parse_ok: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Append a single query record to the JSONL log file.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "language": language,
            "context_chunks": [c[:200] for c in context_chunks],  # truncate for size
            "response": response[:500],  # truncate long responses
            "model": model,
            "source": source,
            "latency_ms": round(latency_ms, 1) if latency_ms else None,
            "json_parse_ok": json_parse_ok,
        }
        if extra:
            entry["extra"] = extra

        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write query log: {e}")


# Singleton getter
_query_logger: Optional[QueryLogger] = None


def get_query_logger() -> QueryLogger:
    global _query_logger
    if _query_logger is None:
        _query_logger = QueryLogger()
    return _query_logger
