"""
Unit tests for automatic LLM-based retrieval-level selection.

Verifies that ``QueryService.query_stream()`` calls ``_select_retrieval_level()``
to choose the recall depth automatically, and that the result is emitted as a
``retrieval_level_note`` WS event.

No real embedding model, LLM, or ChromaDB is loaded.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# query_stream: auto retrieval-level routing
# ---------------------------------------------------------------------------


class TestQueryStreamRetrievalLevel:
    """Verify query_stream auto-selects retrieval level via _select_retrieval_level."""

    def _run_query_stream(self, selected_level: str = "chunk", k: int = 4):
        """
        Call query_stream with mocked retriever/LLM and a fixed _select_retrieval_level
        return value.  Returns (events, recall_kwargs).
        """
        from meetbot.services.query_service import QueryService

        recall_kwargs_capture = {}

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

        mock_adapter = MagicMock()
        mock_adapter.generate_stream.return_value = iter(["Hello ", "world"])

        patches = [
            patch("meetbot.services.query_service.QueryService._get_local_adapter",
                  return_value=mock_adapter),
            patch("meetbot.services.query_service.QueryService._select_retrieval_level",
                  return_value=selected_level),
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
            ))
        finally:
            for p in patches:
                p.stop()

        return events, recall_kwargs_capture

    def test_default_is_chunk(self):
        """When LLM selects 'chunk', recall uses level='chunk'."""
        events, kwargs = self._run_query_stream(selected_level="chunk")
        assert kwargs["level"] == "chunk"
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["retrieval_level"] == "chunk"

    def test_auto_selected_document(self):
        """When LLM selects 'document', recall uses level='document'."""
        events, kwargs = self._run_query_stream(selected_level="document")
        assert kwargs["level"] == "document"
        assert kwargs["top_n"] == 1  # document retrieval uses 1
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["retrieval_level"] == "document"

    def test_auto_selected_segment(self):
        """When LLM selects 'segment', recall uses level='segment'."""
        events, kwargs = self._run_query_stream(selected_level="segment")
        assert kwargs["level"] == "segment"
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["retrieval_level"] == "segment"

    def test_llm_error_falls_back_to_chunk(self):
        """_select_retrieval_level returns 'chunk' when the LLM call raises."""
        from meetbot.services.query_service import QueryService

        mock_adapter = MagicMock()
        mock_adapter.generate.side_effect = RuntimeError("LLM unavailable")

        with patch("meetbot.services.query_service.QueryService._get_local_adapter",
                   return_value=mock_adapter):
            svc = QueryService()
            result = svc._select_retrieval_level("Summarize the meeting", llm_mode="local")

        assert result == "chunk"

    def test_retrieval_level_note_event_emitted(self):
        """A 'retrieval_level_note' event is emitted before 'done'."""
        events, _ = self._run_query_stream(selected_level="segment")
        types_in_order = [e["type"] for e in events]
        assert "retrieval_level_note" in types_in_order
        note_event = next(e for e in events if e["type"] == "retrieval_level_note")
        assert note_event["level"] == "segment"
        # Must come before done
        note_idx = types_in_order.index("retrieval_level_note")
        done_idx = types_in_order.index("done")
        assert note_idx < done_idx

    def test_done_event_contains_retrieval_level(self):
        """'done' event should include retrieval_level field."""
        events, _ = self._run_query_stream(selected_level="chunk")
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["retrieval_level"] == "chunk"

    def test_no_intent_import(self):
        """query_stream should not import detect_intent."""
        import meetbot.services.query_service as qs_module
        import inspect
        source = inspect.getsource(qs_module.QueryService.query_stream)
        assert "detect_intent" not in source
