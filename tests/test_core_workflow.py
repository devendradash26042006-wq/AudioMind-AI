import pytest

from chat_engine import ChatEngine
from concept_analyzer import ConceptAnalyzer, chunk_transcript
from llm_provider import OpenAICompatibleProvider


def test_missing_api_key():
    provider = OpenAICompatibleProvider(api_key="", model="test")
    assert not provider.is_configured
    with pytest.raises(RuntimeError, match="API key"):
        provider.chat("system", "user")


def test_empty_concept_transcript():
    class Provider:
        is_configured = True
    with pytest.raises(ValueError, match="empty"):
        ConceptAnalyzer(Provider()).analyze([])


def test_long_transcript_chunking():
    segments = [{"start": i, "end": i + 1, "text": "word " * 1000} for i in range(3)]
    chunks = chunk_transcript(segments, 1500)
    assert len(chunks) >= 3
    assert all(chunk for chunk in chunks)


class MockProvider:
    is_configured = True

    def chat(self, *args, **kwargs):
        return "No relevant transcript evidence was found."


def test_chat_without_relevant_evidence():
    engine = ChatEngine(MockProvider())
    answer = engine.ask("quantum entanglement", [{"start": 0, "end": 1, "text": "A lesson about cooking."}])
    assert "No relevant transcript evidence" in answer
