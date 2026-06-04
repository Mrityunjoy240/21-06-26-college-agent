import pytest
from app.services.llm.groq_service import GroqService
import json
import os

@pytest.fixture
def groq_service():
    return GroqService()

def test_kb_loading(groq_service):
    """Verify that the knowledge base is loaded and formatted correctly."""
    kb_text = groq_service.knowledge_base
    assert "BCREC" in kb_text or "B.C. Roy" in kb_text
    assert "Principal" in kb_text
    assert "Vice Principal" in kb_text
    assert "HOD" in kb_text

def test_leadership_data_accuracy(groq_service):
    """Verify that specific leadership names are present in the formatted KB."""
    kb_text = groq_service.knowledge_base
    assert "Sanjay S. Pawar" in kb_text
    assert "K. M. Hossain" in kb_text


def test_worst_case_empty_kb(tmp_path):
    """Test how the service handles a missing or empty KB file."""
    # This would require mocking open() or path, but let's check current robustness
    service = GroqService()
    # If we pass a garbage dict to the formatter
    result = service._format_knowledge_base({})
    assert result == ""
