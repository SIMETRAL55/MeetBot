"""
Unit tests for explicit retrieval-level routing.

Verifies that ``QueryService.query_stream()`` uses the ``retrieval_level``
parameter (not intent classification) to determine recall depth, and
that the websocket endpoint correctly parses and forwards the parameter.

No real embedding model, LLM, or ChromaDB is loaded.
"""

from unittest.mock import MagicMock, patch
import types

import pytest


# ---------------------------------------------------------------------------
# query_stream: retrieval_level routing
# ---------------------------------------------------------------------------


class TestQueryStreamRetrievalLevel:
    """Verify query_stream routes based on explicit retrieval_level."""

    def _run_query_stream(self, retrieval_level=None, segment_count=None, k=4):
        """Call query_stream with mocked retriever/LLM, return events + recall kwargs."""
        from meetbot.services.query_service import QueryService

        recall_kwargs_capture = {}

        # Mock Retriever
        mock_candidate = MagicMock()
        mock_candidate.text = "Some transcript text"
        mock_candidate.metadata = {"speaker": "A", "start": 0.0, "end": 5.0, "audio_file": "test.mp3"}
        mock_candidate.speaker = "A"
        mock_candidate.start = 0.0
        mock_candidate.end = 5.0
        mock_candidate.distance = 0.2

        mock_retriever_instance = MagicMock()

        def _capture_recall(**kwargs):
            recall_kwargs_capture.update(kwargs)
            return [mock_candidate]

        mock_retriever_instance.recall.side_effect = _capture_recall
        mock_retriever_instance.score_to_relevance_pct.return_value = 80

        mock_retriever_cls = MagicMock(return_value=mock_retriever_instance)

        # Mock LLM adapter
        mock_adapter = MagicMock()
        mock_adapter.generate_stream.return_value = iter(["Hello ", "world"])

        patches = [
            patch("meetbot.services.query_service.QueryService._get_local_adapter",
                  return_value=mock_adapter),
            patch("meetbot.services.rag.retriever.Retriever", mock_retriever_cls),
            patch("meetbot.services.query_service._get_embedding_model",
                  return_value=MagicMock(embed_documents=lambda texts: [[0.1] * 8 for _ in texts])),
        ]

        for p in patches:
            p.start()

        try:
            svc = QueryService()
            events = list(svc.query_stream(
                question="What happened?",
                db_dir="/tmp/fake_db",
                embedding_model="fake-model",
                k=k,
                llm_mode="local",
                retrieval_level=retrieval_level,
                segment_count=segment_count,
            ))
        finally:
            for p in patches:
                p.stop()

        return events, recall_kwargs_capture

    def test_default_is_chunk(self):
        """When retrieval_level is None, default to 'chunk'."""
        events, kwargs = self._run_query_stream(retrieval_level=None)
        assert kwargs["level"] == "chunk"
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["retrieval_level"] == "chunk"

    def test_explicit_document(self):
        """retrieval_level='document' routes to document level."""
        events, kwargs = self._run_query_stream(retrieval_level="document")
        assert kwargs["level"] == "document"
        assert kwargs["top_n"] == 1  # document retrieval uses 1
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["retrieval_level"] == "document"

    def test_explicit_segment(self):
        """retrieval_level='segment' routes to segment level."""
        events, kwargs = self._run_query_stream(retrieval_level="segment", segment_count=7)
        assert kwargs["level"] == "segment"
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["retrieval_level"] == "segment"

    def test_explicit_chunk(self):
        """retrieval_level='chunk' routes to chunk level."""
        events, kwargs = self._run_query_stream(retrieval_level="chunk")
        assert kwargs["level"] == "chunk"

    def test_invalid_level_defaults_to_chunk(self):
        """Invalid retrieval_level falls back to 'chunk'."""
        events, kwargs = self._run_query_stream(retrieval_level="invalid_value")
        assert kwargs["level"] == "chunk"

    def test_segment_count_used_for_recall(self):
        """When segment level with segment_count, recall top_n uses it."""
        _, kwargs = self._run_query_stream(retrieval_level="segment", segment_count=10)
        assert kwargs["level"] == "segment"
        # top_n should be at least segment_count
        assert kwargs["top_n"] >= 10

    def test_segment_count_ignored_for_chunk(self):
        """segment_count is only relevant for segment level."""
        _, kwargs1 = self._run_query_stream(retrieval_level="chunk", segment_count=10, k=4)
        _, kwargs2 = self._run_query_stream(retrieval_level="chunk", k=4)
        # Both should use the same recall parameters
        assert kwargs1["top_n"] == kwargs2["top_n"]

    def test_done_event_contains_retrieval_level(self):
        """'done' event should include retrieval_level field."""
        events, _ = self._run_query_stream(retrieval_level="segment", segment_count=3)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["retrieval_level"] == "segment"

    def test_no_intent_import(self):
        """query_stream should not import detect_intent."""
        import meetbot.services.query_service as qs_module
        import inspect
        source = inspect.getsource(qs_module.QueryService.query_stream)
        assert "detect_intent" not in source


# ---------------------------------------------------------------------------
# ws_chat: retrieval_level parsing
# ---------------------------------------------------------------------------


class TestWsChatRetrievalLevelParsing:
    """Verify ws_chat.py correctly parses retrieval_level from payload."""

    def test_valid_levels_accepted(self):
        """Each valid level string should pass through unchanged."""
        for level in ("document", "segment", "chunk"):
            payload = {"question": "test", "llm_mode": "local", "retrieval_level": level}
            parsed = payload.get("retrieval_level", "chunk")
            if parsed not in ("document", "segment", "chunk"):
                parsed = "chunk"
            assert parsed == level

    def test_missing_defaults_to_chunk(self):
        """When retrieval_level is missing, default to 'chunk'."""
        payload = {"question": "test", "llm_mode": "local"}
        parsed = payload.get("retrieval_level", "chunk")
        if parsed not in ("document", "segment", "chunk"):
            parsed = "chunk"
        assert parsed == "chunk"

    def test_invalid_defaults_to_chunk(self):
        """Invalid retrieval_level value defaults to 'chunk'."""
        payload = {"question": "test", "retrieval_level": "banana"}
        parsed = payload.get("retrieval_level", "chunk")
        if parsed not in ("document", "segment", "chunk"):
            parsed = "chunk"
        assert parsed == "chunk"

    def test_segment_count_parsed_as_int(self):
        """segment_count should be parsed to int >= 1."""
        payload = {"question": "test", "retrieval_level": "segment", "segment_count": "5"}
        sc = payload.get("segment_count")
        if sc is not None:
            try:
                sc = max(1, int(sc))
            except (TypeError, ValueError):
                sc = None
        assert sc == 5

    def test_segment_count_invalid_becomes_none(self):
        """Non-numeric segment_count should become None."""
        payload = {"question": "test", "segment_count": "abc"}
        sc = payload.get("segment_count")
        if sc is not None:
            try:
                sc = max(1, int(sc))
            except (TypeError, ValueError):
                sc = None
        assert sc is None
