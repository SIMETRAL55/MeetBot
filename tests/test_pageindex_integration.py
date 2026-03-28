"""
Unit tests for the PageIndex integration components.

Tests cover (all using mocks — no real LLM calls made):
- openai_adapter.pageindex_env: env var injection / restoration
- retrieval_strategy: RetrievalMethod enum, RetrievalResult dataclass
- indexer_pageindex: PageIndexAdapter (3-level tree build, content_map, load/save)
- retriever_pageindex: PageIndexRetriever (agentic tree search, sufficiency loop)
"""

import json
import os
import sys
import copy
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_seg(idx, speaker, start, end, text):
    return {"segment_index": idx, "speaker": speaker, "start": start, "end": end, "text": text}


SAMPLE_SEGMENTS = [
    make_seg(0, "SPEAKER_00", 0.0,  13.0, "Welcome to the quarterly review meeting."),
    make_seg(1, "SPEAKER_00", 13.0, 19.0, "Let us start with Q3 numbers and targets."),
    make_seg(2, "SPEAKER_01", 19.0, 45.0, "Revenue was up 15 percent year over year."),
    make_seg(3, "SPEAKER_01", 45.0, 70.0, "Costs increased by about 5 percent overall."),
    make_seg(4, "SPEAKER_00", 70.0, 90.0, "Thank you. Let us discuss next steps."),
    make_seg(5, "SPEAKER_00", 90.0, 110.0, "We need to finalize the budget proposal."),
    make_seg(6, "SPEAKER_01", 110.0, 130.0, "I can prepare the draft by Friday."),
    make_seg(7, "SPEAKER_00", 130.0, 150.0, "Great. Any other business?"),
    make_seg(8, "SPEAKER_01", 150.0, 161.0, "No, that covers everything. Thank you."),
]

# Canned LLM responses for mocking
MOCK_TOPICS = [
    {"start_segment": 0, "end_segment": 3, "topic_title": "Q3 Financial Review"},
    {"start_segment": 4, "end_segment": 6, "topic_title": "Action Items and Budget"},
    {"start_segment": 7, "end_segment": 8, "topic_title": "Closing"},
]

MOCK_SUBTOPICS_0 = [
    {"start_segment": 0, "end_segment": 1, "title": "SPEAKER_00: Opening", "speakers": ["SPEAKER_00"]},
    {"start_segment": 2, "end_segment": 3, "title": "SPEAKER_01: Financials", "speakers": ["SPEAKER_01"]},
]

MOCK_SUBTOPICS_1 = [
    {"start_segment": 4, "end_segment": 5, "title": "SPEAKER_00: Next steps", "speakers": ["SPEAKER_00"]},
    {"start_segment": 6, "end_segment": 6, "title": "SPEAKER_01: Budget draft", "speakers": ["SPEAKER_01"]},
]

MOCK_SUBTOPICS_2 = [
    {"start_segment": 7, "end_segment": 8, "title": "SPEAKER_00 & SPEAKER_01: Wrap-up", "speakers": ["SPEAKER_00", "SPEAKER_01"]},
]

MOCK_SUMMARY = {
    "summary": "Discussion of Q3 financial performance.",
    "decision": None,
    "action_items": [],
    "owner": None,
    "keywords": ["Q3", "revenue"],
}

MOCK_ROOT = {"title": "Q3 Quarterly Review Meeting", "summary": "A meeting covering Q3 finances and next steps."}


# ===========================================================================
# openai_adapter: pageindex_env
# ===========================================================================

class TestPageindexEnv:
    def _make_settings(self, backend="openrouter", base_url="", model="test-model", api_key="sk-test"):
        s = MagicMock()
        s.PAGEINDEX_LLM_BACKEND = backend
        s.PAGEINDEX_LLM_BASE_URL = base_url
        s.PAGEINDEX_LLM_MODEL = model
        s.PAGEINDEX_LLM_API_KEY = api_key
        s.get_pageindex_base_url.return_value = base_url or "https://openrouter.ai/api/v1"
        return s

    def test_remote_backend_does_not_mutate_env(self):
        """pageindex_env() must not touch os.environ for remote backends (thread-safety)."""
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(base_url="https://openrouter.ai/api/v1", api_key="sk-rkey")
        before = dict(os.environ)
        with pageindex_env(settings):
            pass
        # env must be unchanged for remote backends
        assert os.environ == before

    def test_remote_backend_yields_cleanly_on_exception(self):
        """pageindex_env() must propagate exceptions for remote backends."""
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(base_url="https://openrouter.ai/api/v1")
        before = dict(os.environ)
        with pytest.raises(ValueError):
            with pageindex_env(settings):
                raise ValueError("test error")
        assert os.environ == before

    def test_local_backend_starts_ollama(self):
        """pageindex_env() calls _ensure_ollama_running for local/ollama backends."""
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(backend="local", base_url="http://localhost:11434/v1")
        with patch("meetbot.adapters.llm.openai_adapter._ensure_ollama_running") as mock_start, \
             patch("meetbot.adapters.llm.openai_adapter._ensure_model_available"), \
             patch("meetbot.adapters.llm.openai_adapter._release_ollama"):
            with pageindex_env(settings):
                pass
            mock_start.assert_called_once()

    def test_local_backend_releases_ollama_on_exit(self):
        """pageindex_env() calls _release_ollama on exit for local/ollama backends."""
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(backend="local", base_url="http://localhost:11434/v1")
        with patch("meetbot.adapters.llm.openai_adapter._ensure_ollama_running"), \
             patch("meetbot.adapters.llm.openai_adapter._ensure_model_available"), \
             patch("meetbot.adapters.llm.openai_adapter._release_ollama") as mock_release:
            with pageindex_env(settings):
                pass
            mock_release.assert_called_once()

    def test_openrouter_backend_skips_ollama(self):
        """pageindex_env() does not start Ollama for remote backends."""
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(backend="openrouter", base_url="https://openrouter.ai/api/v1")
        with patch("meetbot.adapters.llm.openai_adapter._ensure_ollama_running") as mock_start, \
             patch("meetbot.adapters.llm.openai_adapter._release_ollama") as mock_release:
            with pageindex_env(settings):
                pass
            mock_start.assert_not_called()
            mock_release.assert_not_called()


# ===========================================================================
# retrieval_strategy: RetrievalMethod, RetrievalResult
# ===========================================================================

class TestRetrievalStrategy:
    def test_retrieval_method_values(self):
        from meetbot.services.rag.retrieval_strategy import RetrievalMethod
        assert RetrievalMethod.VECTOR == "vector"
        assert RetrievalMethod.PAGEINDEX == "pageindex"

    def test_retrieval_method_str_comparison(self):
        from meetbot.services.rag.retrieval_strategy import RetrievalMethod
        assert RetrievalMethod.VECTOR == "vector"
        assert "vector" == RetrievalMethod.VECTOR

    def test_retrieval_result_fields(self):
        from meetbot.services.rag.retrieval_strategy import RetrievalResult
        r = RetrievalResult(
            text="hello",
            metadata={"speaker": "Alice"},
            score=0.9,
            retrieval_method="vector",
            source_ref="chunk_0",
        )
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.retrieval_method == "vector"
        assert r.source_ref == "chunk_0"


# ===========================================================================
# indexer_pageindex: PageIndexAdapter — 3-level tree build
# ===========================================================================

class TestPageIndexAdapter:
    """Test the new transcript-native 3-level PageIndexAdapter."""

    def _make_adapter(self):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        return PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

    def _mock_llm_side_effect(self, call_count_state):
        """Return canned responses based on prompt content."""
        subtopic_call = [0]  # mutable counter for subtopic calls
        summary_call = [0]

        def side_effect(prompt):
            if "Identify 4-7 major topic" in prompt:
                return json.dumps(MOCK_TOPICS)
            elif "Split the segments below into 2-4 sub-clusters" in prompt:
                responses = [MOCK_SUBTOPICS_0, MOCK_SUBTOPICS_1, MOCK_SUBTOPICS_2]
                idx = min(subtopic_call[0], len(responses) - 1)
                subtopic_call[0] += 1
                return json.dumps(responses[idx])
            elif "Summarize the following meeting transcript" in prompt:
                summary_call[0] += 1
                return json.dumps(MOCK_SUMMARY)
            elif "Produce a concise overall title" in prompt:
                return json.dumps(MOCK_ROOT)
            return json.dumps({})

        return side_effect

    @pytest.mark.asyncio
    async def test_build_index_creates_3_level_tree(self, tmp_path):
        adapter = self._make_adapter()

        with patch.object(adapter, "_llm_call", side_effect=self._mock_llm_side_effect({})):
            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_s:
                mock_s.PAGEINDEX_LLM_MODEL = "test"
                mock_s.get_pageindex_output_dir.return_value = tmp_path
                mock_s.get_pageindex_base_url.return_value = "http://localhost"
                mock_s.PAGEINDEX_LLM_API_KEY = "no-key"

                result_path = await adapter.build_index(
                    segments=SAMPLE_SEGMENTS,
                    job_id="test-job-001",
                    output_dir=tmp_path,
                    filename="Test Meeting",
                )

        assert result_path.exists()
        with open(result_path) as f:
            tree = json.load(f)

        # ROOT level 0
        assert tree["node_id"] == "ROOT"
        assert tree["level"] == 0

        # Level 1 topics
        assert len(tree["children"]) == 3
        for i, child in enumerate(tree["children"]):
            assert child["node_id"] == f"T{i}"
            assert child["level"] == 1

        # Level 2 subtopics
        t0_subs = tree["children"][0]["children"]
        assert len(t0_subs) == 2
        assert t0_subs[0]["node_id"] == "T0-S0"
        assert t0_subs[0]["level"] == 2

    @pytest.mark.asyncio
    async def test_build_index_creates_content_map(self, tmp_path):
        adapter = self._make_adapter()

        with patch.object(adapter, "_llm_call", side_effect=self._mock_llm_side_effect({})):
            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_s:
                mock_s.PAGEINDEX_LLM_MODEL = "test"
                mock_s.get_pageindex_output_dir.return_value = tmp_path
                mock_s.get_pageindex_base_url.return_value = "http://localhost"
                mock_s.PAGEINDEX_LLM_API_KEY = "no-key"

                result_path = await adapter.build_index(
                    segments=SAMPLE_SEGMENTS,
                    job_id="test-job-002",
                    output_dir=tmp_path,
                    filename="Test Meeting",
                )

        cmap_path = Path(str(result_path).replace("_tree.json", "_content_map.json"))
        assert cmap_path.exists()

        with open(cmap_path) as f:
            content_map = json.load(f)

        # Content map should have entries for ROOT and all nodes
        assert "ROOT" in content_map
        assert "T0" in content_map
        assert "T0-S0" in content_map
        # Content should contain segment text
        assert "quarterly review" in content_map["T0"].lower() or "quarterly review" in content_map["T0-S0"].lower()

    @pytest.mark.asyncio
    async def test_build_index_progress_callback(self, tmp_path):
        adapter = self._make_adapter()
        progress_calls = []

        def progress_cb(stage, pct, msg):
            progress_calls.append((stage, pct, msg))

        with patch.object(adapter, "_llm_call", side_effect=self._mock_llm_side_effect({})):
            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_s:
                mock_s.PAGEINDEX_LLM_MODEL = "test"
                mock_s.get_pageindex_output_dir.return_value = tmp_path
                mock_s.get_pageindex_base_url.return_value = "http://localhost"
                mock_s.PAGEINDEX_LLM_API_KEY = "no-key"

                await adapter.build_index(
                    segments=SAMPLE_SEGMENTS,
                    job_id="test-job-003",
                    output_dir=tmp_path,
                    filename="Test Meeting",
                    progress_callback=progress_cb,
                )

        assert len(progress_calls) > 0
        stages = [c[1] for c in progress_calls]
        assert 0 in stages
        assert 100 in stages


class TestPageIndexFallback:
    """Test fallback paths when LLM calls fail."""

    def _make_adapter(self):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        return PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

    @pytest.mark.asyncio
    async def test_topic_segmentation_fallback_on_garbage(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "_llm_call", return_value="not valid json at all"):
            topics = await adapter._segment_topics(SAMPLE_SEGMENTS)
        # Should fallback to equal-split blocks
        assert len(topics) > 0
        assert topics[0]["start_segment"] == 0

    @pytest.mark.asyncio
    async def test_subtopic_segmentation_fallback_to_speaker_turns(self):
        adapter = self._make_adapter()
        topic = {"start_segment": 0, "end_segment": 3, "topic_title": "Test"}
        with patch.object(adapter, "_llm_call", return_value="garbage"):
            subs = await adapter._segment_subtopics(SAMPLE_SEGMENTS, topic)
        # Should fallback to speaker-turn grouping
        assert len(subs) > 0
        assert "speakers" in subs[0]

    @pytest.mark.asyncio
    async def test_summarize_node_fallback(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "_llm_call", return_value="not json"):
            result = await adapter._summarize_node(SAMPLE_SEGMENTS, 0, 2)
        assert "summary" in result
        assert result["decision"] is None


class TestPageIndexPersistence:
    """Test round-trip save/load."""

    @pytest.mark.asyncio
    async def test_load_index_roundtrip(self, tmp_path):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter

        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        # Mock LLM calls
        def mock_llm(prompt):
            if "Identify 4-7 major topic" in prompt:
                return json.dumps(MOCK_TOPICS)
            elif "Split the segments below into 2-4 sub-clusters" in prompt:
                return json.dumps(MOCK_SUBTOPICS_0)
            elif "Summarize the following meeting transcript" in prompt:
                return json.dumps(MOCK_SUMMARY)
            elif "Produce a concise overall title" in prompt:
                return json.dumps(MOCK_ROOT)
            return json.dumps({})

        with patch.object(adapter, "_llm_call", side_effect=mock_llm):
            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_s:
                mock_s.PAGEINDEX_LLM_MODEL = "test"
                mock_s.get_pageindex_output_dir.return_value = tmp_path
                mock_s.get_pageindex_base_url.return_value = "http://localhost"
                mock_s.PAGEINDEX_LLM_API_KEY = "no-key"

                result_path = await adapter.build_index(
                    segments=SAMPLE_SEGMENTS,
                    job_id="roundtrip-001",
                    output_dir=tmp_path,
                    filename="Test Meeting",
                )

        # Load back
        loaded = PageIndexAdapter.load_index(str(result_path))
        assert loaded.tree["node_id"] == "ROOT"
        assert isinstance(loaded.content_map, dict)
        assert len(loaded.content_map) > 0

    def test_load_index_raises_when_missing(self):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        with pytest.raises(FileNotFoundError):
            PageIndexAdapter.load_index("/nonexistent/path/tree.json")


# ===========================================================================
# retriever_pageindex: PageIndexRetriever — agentic search
# ===========================================================================

# Sample tree for retriever tests (matches new 3-level format)
SAMPLE_TREE_FOR_RETRIEVER = {
    "node_id": "ROOT",
    "level": 0,
    "title": "Q3 Review Meeting",
    "summary": "Meeting about Q3 finances and next steps.",
    "participants": ["SPEAKER_00", "SPEAKER_01"],
    "n_segments": 9,
    "children": [
        {
            "node_id": "T0",
            "level": 1,
            "title": "Q3 Financial Review",
            "start_segment": 0,
            "end_segment": 3,
            "summary": {"summary": "Discussion of Q3 revenue and costs."},
            "children": [
                {
                    "node_id": "T0-S0",
                    "level": 2,
                    "title": "SPEAKER_00: Opening",
                    "start_segment": 0,
                    "end_segment": 1,
                    "speakers": ["SPEAKER_00"],
                    "summary": {"summary": "Opening remarks about Q3 review."},
                    "children": [],
                },
                {
                    "node_id": "T0-S1",
                    "level": 2,
                    "title": "SPEAKER_01: Financials",
                    "start_segment": 2,
                    "end_segment": 3,
                    "speakers": ["SPEAKER_01"],
                    "summary": {"summary": "Revenue up 15%, costs up 5%."},
                    "children": [],
                },
            ],
        },
        {
            "node_id": "T1",
            "level": 1,
            "title": "Action Items",
            "start_segment": 4,
            "end_segment": 6,
            "summary": {"summary": "Discussion of budget and next steps."},
            "children": [
                {
                    "node_id": "T1-S0",
                    "level": 2,
                    "title": "SPEAKER_00: Next steps",
                    "start_segment": 4,
                    "end_segment": 5,
                    "speakers": ["SPEAKER_00"],
                    "summary": {"summary": "Budget proposal discussion."},
                    "children": [],
                },
            ],
        },
    ],
}

SAMPLE_CONTENT_MAP = {
    "ROOT": "Full transcript text here.",
    "T0": "SPEAKER_00 [00:00:00-00:00:13]: Welcome to the quarterly review.\nSPEAKER_01 [00:00:19-00:00:45]: Revenue was up 15 percent.",
    "T0-S0": "SPEAKER_00 [00:00:00-00:00:13]: Welcome to the quarterly review.\nSPEAKER_00 [00:00:13-00:00:19]: Let us start with Q3 numbers.",
    "T0-S1": "SPEAKER_01 [00:00:19-00:00:45]: Revenue was up 15 percent.\nSPEAKER_01 [00:00:45-00:01:10]: Costs increased by about 5 percent.",
    "T1": "SPEAKER_00 [00:01:10-00:01:30]: Thank you. Let us discuss next steps.",
    "T1-S0": "SPEAKER_00 [00:01:10-00:01:30]: Thank you. Let us discuss next steps.\nSPEAKER_00 [00:01:30-00:01:50]: We need to finalize the budget.",
}


class TestPageIndexRetriever:
    """Test the agentic retriever with mocked LLM calls."""

    def _make_retriever(self):
        from meetbot.services.rag.retriever_pageindex import PageIndexRetriever
        return PageIndexRetriever(model="test", base_url="http://localhost", api_key="no-key")

    @pytest.mark.asyncio
    async def test_search_returns_retrieval_results(self):
        r = self._make_retriever()

        def mock_llm(prompt):
            if "Select 1-3 node IDs" in prompt:
                return json.dumps({"thinking": "financials", "node_ids": ["T0-S1"]})
            elif "sufficient" in prompt.lower():
                return json.dumps({"sufficient": True, "reasoning": "answer found"})
            return json.dumps({})

        with patch.object(r, "_llm_call", side_effect=mock_llm):
            results = await r.search(
                query="What was the revenue growth?",
                tree=copy.deepcopy(SAMPLE_TREE_FOR_RETRIEVER),
                content_map=SAMPLE_CONTENT_MAP,
            )

        assert len(results) > 0
        from meetbot.services.rag.retrieval_strategy import RetrievalResult
        assert isinstance(results[0], RetrievalResult)
        assert results[0].metadata["node_id"] == "T0-S1"
        assert results[0].retrieval_method == "pageindex"
        assert "Revenue" in results[0].text or "revenue" in results[0].text

    @pytest.mark.asyncio
    async def test_search_empty_on_no_nodes(self):
        r = self._make_retriever()

        def mock_llm(prompt):
            return json.dumps({"thinking": "no match", "node_ids": []})

        with patch.object(r, "_llm_call", side_effect=mock_llm):
            results = await r.search(
                query="unrelated query",
                tree=copy.deepcopy(SAMPLE_TREE_FOR_RETRIEVER),
                content_map=SAMPLE_CONTENT_MAP,
            )

        assert results == []

    def test_build_tree_summary_includes_node_ids(self):
        r = self._make_retriever()
        summary = r._build_tree_summary(SAMPLE_TREE_FOR_RETRIEVER)
        assert "T0" in summary
        assert "T0-S0" in summary
        assert "T1" in summary
        assert "ROOT" in summary

    def test_build_flat_node_map(self):
        r = self._make_retriever()
        nmap = r._build_flat_node_map(SAMPLE_TREE_FOR_RETRIEVER)
        assert "ROOT" in nmap
        assert "T0" in nmap
        assert "T0-S0" in nmap
        assert "T0-S1" in nmap
        assert "T1" in nmap
        assert "T1-S0" in nmap


class TestPageIndexRetrieverMultiTurn:
    """Test multi-iteration retrieval (insufficiency → re-search)."""

    def _make_retriever(self):
        from meetbot.services.rag.retriever_pageindex import PageIndexRetriever
        return PageIndexRetriever(model="test", base_url="http://localhost", api_key="no-key")

    @pytest.mark.asyncio
    async def test_two_iteration_search(self):
        r = self._make_retriever()
        call_count = [0]

        def mock_llm(prompt):
            call_count[0] += 1
            if "Select 1-3 node IDs" in prompt:
                if call_count[0] <= 2:  # First iteration
                    return json.dumps({"thinking": "start with T0", "node_ids": ["T0"]})
                else:  # Second iteration
                    return json.dumps({"thinking": "also check T1", "node_ids": ["T1-S0"]})
            elif "sufficient" in prompt.lower():
                if call_count[0] <= 2:  # First sufficiency check
                    return json.dumps({"sufficient": False, "node_ids": [], "reasoning": "need more"})
                else:  # Second sufficiency check
                    return json.dumps({"sufficient": True, "reasoning": "enough"})
            return json.dumps({})

        with patch.object(r, "_llm_call", side_effect=mock_llm):
            results = await r.search(
                query="What are the action items?",
                tree=copy.deepcopy(SAMPLE_TREE_FOR_RETRIEVER),
                content_map=SAMPLE_CONTENT_MAP,
            )

        # Should have nodes from both iterations
        node_ids = [r.metadata["node_id"] for r in results]
        assert "T0" in node_ids
        assert "T1-S0" in node_ids
        # At least 4 LLM calls (2 search + 2 sufficiency)
        assert call_count[0] >= 4

    @pytest.mark.asyncio
    async def test_search_with_chat_history(self):
        r = self._make_retriever()

        def mock_llm(prompt):
            # Verify chat history is present in the prompt
            if "Select 1-3 node IDs" in prompt:
                assert "What was discussed" in prompt
                return json.dumps({"thinking": "budget", "node_ids": ["T1-S0"]})
            elif "sufficient" in prompt.lower():
                return json.dumps({"sufficient": True, "reasoning": "ok"})
            return json.dumps({})

        chat_history = [
            {"role": "user", "content": "What was discussed in the meeting?"},
            {"role": "assistant", "content": "The meeting covered Q3 finances."},
        ]

        with patch.object(r, "_llm_call", side_effect=mock_llm):
            results = await r.search(
                query="What about the budget?",
                tree=copy.deepcopy(SAMPLE_TREE_FOR_RETRIEVER),
                content_map=SAMPLE_CONTENT_MAP,
                chat_history=chat_history,
            )

        assert len(results) > 0


# ===========================================================================
# Sync-worker path: asyncio.new_event_loop() + loop.run_until_complete()
# ===========================================================================

class TestBuildIndexSyncWorkerPath:
    """
    Covers the call pattern used in pipeline_worker.py Stage 4b and
    reindex_worker.py — both synchronous threads that must drive
    the async build_index() via asyncio.new_event_loop().
    """

    def _make_adapter(self):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        return PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

    def _mock_llm(self, prompt: str) -> str:
        """Canned responses keyed on prompt keywords."""
        if "Identify 4-7 major topic" in prompt:
            return json.dumps(MOCK_TOPICS)
        elif "Split the segments below into 2-4 sub-clusters" in prompt:
            return json.dumps(MOCK_SUBTOPICS_0)
        elif "Summarize the following meeting transcript" in prompt:
            return json.dumps(MOCK_SUMMARY)
        elif "Produce a concise overall title" in prompt:
            return json.dumps(MOCK_ROOT)
        return json.dumps({})

    def test_build_index_via_new_event_loop(self, tmp_path):
        """
        Mirrors pipeline_worker.py Stage 4b exactly:
          loop = asyncio.new_event_loop()
          try:
              result = loop.run_until_complete(adapter.build_index(...))
          finally:
              loop.close()
        """
        import asyncio

        adapter = self._make_adapter()

        with patch.object(adapter, "_llm_call", side_effect=self._mock_llm):
            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_s:
                mock_s.PAGEINDEX_LLM_MODEL = "test"
                mock_s.get_pageindex_output_dir.return_value = tmp_path
                mock_s.get_pageindex_base_url.return_value = "http://localhost"
                mock_s.PAGEINDEX_LLM_API_KEY = "no-key"

                loop = asyncio.new_event_loop()
                try:
                    result_path = loop.run_until_complete(
                        adapter.build_index(
                            segments=SAMPLE_SEGMENTS,
                            job_id="sync-worker-test-001",
                            output_dir=tmp_path,
                            filename="Sync Worker Test",
                        )
                    )
                finally:
                    loop.close()

        assert result_path.exists()
        with open(result_path) as f:
            tree = json.load(f)
        assert tree["node_id"] == "ROOT"
        assert tree["level"] == 0
        assert len(tree["children"]) == 3

    def test_build_index_sync_worker_llm_failure_uses_fallback(self, tmp_path):
        """When _llm_call always fails, build_index completes via fallbacks (non-fatal design).

        Mirrors the pipeline_worker.py Stage 4b resilience: LLM errors are caught
        internally and the tree is still produced with fallback topic splits.
        The loop closes cleanly after run_until_complete returns.
        """
        import asyncio

        adapter = self._make_adapter()

        with patch.object(adapter, "_llm_call", side_effect=RuntimeError("LLM failure")):
            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_s:
                mock_s.PAGEINDEX_LLM_MODEL = "test"
                mock_s.get_pageindex_output_dir.return_value = tmp_path
                mock_s.get_pageindex_base_url.return_value = "http://localhost"
                mock_s.PAGEINDEX_LLM_API_KEY = "no-key"

                loop = asyncio.new_event_loop()
                try:
                    result_path = loop.run_until_complete(
                        adapter.build_index(
                            segments=SAMPLE_SEGMENTS,
                            job_id="sync-worker-error-test",
                            output_dir=tmp_path,
                            filename="Error Test",
                        )
                    )
                finally:
                    loop.close()

        # build_index uses fallback topic/subtopic splits — still produces a valid tree
        assert result_path.exists()
        with open(result_path) as f:
            tree = json.load(f)
        assert tree["node_id"] == "ROOT"
        assert tree["level"] == 0
        assert isinstance(tree["children"], list)
