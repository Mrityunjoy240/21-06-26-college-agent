import pytest
from app.services.llm.groq_service import GroqService, GREETING_PATTERNS
import re


@pytest.fixture
def groq_service():
    return GroqService()


def test_kb_reload(groq_service):
    """Verify reload_kb() loads voice_ready_answers correctly."""
    count = groq_service.reload_kb()
    assert count > 0, "KB should have at least 1 FAQ entry"


def test_session_memory_and_clear(groq_service):
    """Verify session memory stores turns and can be cleared."""
    sid = "test-session-123"
    groq_service._append_session_turn(sid, "hi", "hello there")
    groq_service._append_session_turn(sid, "fees?", "fees are...")
    hist = groq_service._get_session_history(sid)
    assert len(hist) == 4  # 2 user + 2 assistant
    assert hist[0]["role"] == "user"
    assert hist[0]["content"] == "hi"
    groq_service.clear_session(sid)
    assert groq_service._get_session_history(sid) == []


def test_greeting_patterns():
    """Greeting regex patterns match common greetings."""
    for greeting in ["hi", "hello", "hey", "good morning", "good evening"]:
        assert any(re.match(p, greeting, re.IGNORECASE) for p in GREETING_PATTERNS), (
            f"'{greeting}' should match a greeting pattern"
        )


def test_non_greeting_does_not_match():
    """Non-greeting queries should not match greeting patterns."""
    for query in ["what is the fees", "hod name", "placement"]:
        assert not any(re.match(p, query, re.IGNORECASE) for p in GREETING_PATTERNS), (
            f"'{query}' should NOT match greeting patterns"
        )


def test_cache_operations(groq_service):
    """Cache should be invalidatable and return stats."""
    stats_before = groq_service.get_cache_stats()
    assert "hits" in stats_before
    assert "misses" in stats_before

    cleared = groq_service.invalidate_cache()
    assert isinstance(cleared, int)
    assert cleared >= 0


def test_greeting_deterministic_response(groq_service):
    """Short greetings should be answered without LLM."""
    import asyncio

    result = asyncio.run(groq_service.generate_response("hi"))
    assert result["source"] == "greeting_deterministic"
    assert "How can I help" in result["answer"] or "help you" in result["answer"].lower()
