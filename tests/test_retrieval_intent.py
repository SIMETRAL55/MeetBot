"""
Unit tests for intent detection (services/rag/intent.py).

Tests cover all three retrieval levels and edge cases such as
empty queries, mixed-case input, and Japanese keywords.
"""

import pytest

from meetbot.services.rag.intent import detect_intent


class TestDetectIntentDocument:
    """Queries that should route to the document level."""

    @pytest.mark.parametrize("query", [
        "Please summarize this meeting",
        "Give me a summary",
        "What is the overview of the discussion?",
        "Provide an overall recap",
        "What were the main points?",
        "Give me a brief recap of the talk",
        # Japanese equivalents
        "この会議を要約してください",
        "まとめを教えてください",
        "概要を教えてください",
        "全体的にどんな内容でしたか",
    ])
    def test_document_intent(self, query):
        assert detect_intent(query) == "document", f"Expected 'document' for: {query!r}"

    def test_case_insensitive_summary(self):
        assert detect_intent("SUMMARY of the call") == "document"

    def test_case_insensitive_summarize(self):
        assert detect_intent("Summarize EVERYTHING") == "document"


class TestDetectIntentSegment:
    """Queries that should route to the segment level."""

    @pytest.mark.parametrize("query", [
        "Who said something about the budget?",
        "When did Alice mention the deadline?",
        "Did Bob say anything about the project?",
        "What did the speaker say about pricing?",
        "Show me the timestamp for the important part",
        # Japanese equivalents
        "誰がその点について言いましたか",
        "誰が最初に発言しましたか",
        "いつその話をしましたか",
    ])
    def test_segment_intent(self, query):
        assert detect_intent(query) == "segment", f"Expected 'segment' for: {query!r}"

    def test_case_insensitive_who_said(self):
        assert detect_intent("WHO SAID we need to ship by Friday?") == "segment"

    def test_case_insensitive_timestamp(self):
        assert detect_intent("TIMESTAMP for the key decision") == "segment"


class TestDetectIntentChunk:
    """Queries that should fall through to chunk (default) level."""

    @pytest.mark.parametrize("query", [
        "What is the budget?",
        "How much does it cost?",
        "Tell me about the project timeline",
        "What were the action items?",
        "Explain the technical approach",
        "",  # empty query → default
        "   ",  # whitespace only → default
    ])
    def test_chunk_intent(self, query):
        assert detect_intent(query) == "chunk", f"Expected 'chunk' for: {query!r}"


class TestDetectIntentReturnType:
    def test_returns_string(self):
        result = detect_intent("what happened")
        assert isinstance(result, str)

    def test_valid_levels_only(self):
        valid = {"document", "segment", "chunk"}
        for query in ["summarize", "who said", "what is", "foo bar baz"]:
            assert detect_intent(query) in valid
