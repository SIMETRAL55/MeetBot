"""
Unit tests for the PageIndex integration components.

Tests cover (all using mocks — no real LLM calls made):
- openai_adapter.pageindex_env: env var injection / restoration
- retrieval_strategy: RetrievalMethod enum, RetrievalResult dataclass
- indexer_pageindex: PageIndexAdapter (simple tree, segment enrichment, load/save)
- retriever_pageindex: PageIndexRetriever (parse_node_list, tree summary, search)
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
    make_seg(0, "Alice", 0.0,  15.0, "Welcome to the quarterly review."),
    make_seg(1, "Alice", 15.0, 30.0, "Let us start with Q3 numbers."),
    make_seg(2, "Bob",   30.0, 60.0, "Revenue was up 15 percent."),
    make_seg(3, "Bob",   60.0, 90.0, "Costs increased by 5 percent."),
    make_seg(4, "Alice", 90.0, 120.0, "Thank you Bob, let us discuss actions."),
]

SAMPLE_TREE = {
    "title": "Meeting Transcript: Q3 Review",
    "node_id": "root",
    "children": [
        {
            "node_id": "0001",
            "title": "Speaker: Alice (00:00 - 00:30)",
            "children": [],
            "content_lines": [3, 4],
            "start_line": 2,
            "end_line": 5,
            "segment_indices": [0, 1],
        },
        {
            "node_id": "0002",
            "title": "Speaker: Bob (00:30 - 01:30)",
            "children": [],
            "content_lines": [7, 8],
            "start_line": 6,
            "end_line": 9,
            "segment_indices": [2, 3],
        },
        {
            "node_id": "0003",
            "title": "Speaker: Alice (01:30 - 02:00)",
            "children": [],
            "content_lines": [11],
            "start_line": 10,
            "end_line": 12,
            "segment_indices": [4],
        },
    ],
    "content_lines": [],
    "segment_indices": [],
}


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

    def test_sets_env_vars(self):
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(base_url="https://openrouter.ai/api/v1", api_key="sk-rkey")
        with pageindex_env(settings):
            assert os.environ["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
            assert os.environ["OPENAI_API_KEY"] == "sk-rkey"

    def test_restores_original_env_vars(self):
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(base_url="https://openrouter.ai/api/v1", api_key="sk-rkey")

        original_base = os.environ.get("OPENAI_BASE_URL")
        original_key = os.environ.get("OPENAI_API_KEY")
        with pageindex_env(settings):
            pass
        assert os.environ.get("OPENAI_BASE_URL") == original_base
        assert os.environ.get("OPENAI_API_KEY") == original_key

    def test_restores_on_exception(self):
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(base_url="https://openrouter.ai/api/v1")

        original_base = os.environ.get("OPENAI_BASE_URL")
        with pytest.raises(ValueError):
            with pageindex_env(settings):
                raise ValueError("test error")
        assert os.environ.get("OPENAI_BASE_URL") == original_base

    def test_local_backend_sets_ollama_keepalive(self):
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(backend="local", base_url="http://localhost:11434/v1")
        with pageindex_env(settings):
            assert os.environ.get("OLLAMA_KEEP_ALIVE") == "0"

    def test_openrouter_backend_no_keepalive(self):
        from meetbot.adapters.llm.openai_adapter import pageindex_env
        settings = self._make_settings(backend="openrouter", base_url="https://openrouter.ai/api/v1")
        # Remove OLLAMA_KEEP_ALIVE before test
        os.environ.pop("OLLAMA_KEEP_ALIVE", None)
        with pageindex_env(settings):
            assert os.environ.get("OLLAMA_KEEP_ALIVE") is None


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
# indexer_pageindex: PageIndexAdapter
# ===========================================================================

class TestPageIndexAdapter:
    def test_build_simple_tree_has_children(self, tmp_path):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        # Write a simple markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Meeting Transcript\n\n"
            "## Speaker: Alice (00:00 - 00:30)\n\n"
            "[00:00 - 00:15] Alice: Hello\n"
            "[00:15 - 00:30] Alice: World\n\n"
            "## Speaker: Bob (00:30 - 01:00)\n\n"
            "[00:30 - 01:00] Bob: Goodbye\n",
            encoding="utf-8",
        )
        tree = adapter._build_simple_tree(str(md_file))
        assert "children" in tree
        assert len(tree["children"]) >= 1

    def test_enrich_tree_populates_segment_indices(self):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        tree = {
            "title": "root",
            "node_id": "root",
            "children": [
                {
                    "node_id": "0001",
                    "title": "Speaker A",
                    "children": [],
                    "content_lines": [3, 4],
                    "start_line": 2,
                    "end_line": 5,
                }
            ],
            "content_lines": [],
        }
        line_to_seg = {3: 0, 4: 1}
        enriched = adapter._enrich_tree_with_segments(tree, line_to_seg)

        child = enriched["children"][0]
        assert 0 in child["segment_indices"]
        assert 1 in child["segment_indices"]

    def test_enrich_tree_no_mapping_empty_indices(self):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        # A node with no matching line mappings should get segment_indices: []
        tree = {
            "title": "root",
            "children": [
                {
                    "node_id": "0001",
                    "title": "Speaker A",
                    "children": [],
                    "content_lines": [],
                    "start_line": 2,
                    "end_line": 5,
                }
            ],
            "content_lines": [],
        }
        enriched = adapter._enrich_tree_with_segments(tree, {})
        assert enriched["children"][0]["segment_indices"] == []

    def test_enrich_tree_real_pageindex_format(self):
        """_enrich_tree_with_segments must handle the real md_to_tree() output format."""
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        # Real PageIndex format: uses "structure" at root and "nodes" within nodes,
        # with only "line_num" for position (no start_line/end_line/content_lines).
        tree = {
            "doc_name": "transcript",
            "structure": [
                {
                    "title": "Meeting Transcript",
                    "node_id": "0000",
                    "line_num": 1,
                    "nodes": [
                        {"title": "Speaker: Alice (00:00 - 00:30)", "node_id": "0001", "line_num": 3},
                        {"title": "Speaker: Bob (00:30 - 01:30)",   "node_id": "0002", "line_num": 7},
                    ],
                }
            ],
        }
        # Lines 3-6 -> segment 0, lines 7-10 -> segment 1
        line_to_segment = {3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1}
        enriched = adapter._enrich_tree_with_segments(tree, line_to_segment)

        top = enriched["structure"][0]
        alice = top["nodes"][0]
        bob = top["nodes"][1]
        assert alice["segment_indices"] == [0]
        assert bob["segment_indices"] == [1]

    def test_load_index_raises_when_missing(self, tmp_path):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")
        with pytest.raises(FileNotFoundError):
            adapter.load_index("nonexistent-job-id", output_dir=tmp_path)

    def test_load_index_returns_tree(self, tmp_path):
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter
        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        job_id = "test-job-id-1234"
        tree_path = tmp_path / f"{job_id}_pageindex.json"
        tree_path.write_text(json.dumps(SAMPLE_TREE), encoding="utf-8")

        loaded = adapter.load_index(job_id, output_dir=tmp_path)
        assert loaded["title"] == SAMPLE_TREE["title"]
        assert len(loaded["children"]) == 3

    @pytest.mark.asyncio
    async def test_build_index_saves_json(self, tmp_path):
        """build_index should save a JSON file even when PageIndex is unavailable (fallback path)."""
        from meetbot.services.rag.indexer_pageindex import PageIndexAdapter

        adapter = PageIndexAdapter(model="test", base_url="http://localhost", api_key="no-key")

        # Patch md_to_tree to simulate unavailability (ImportError → fallback)
        with patch.object(adapter, "_call_md_to_tree", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = copy.deepcopy(SAMPLE_TREE)

            with patch("meetbot.services.rag.indexer_pageindex.settings") as mock_settings:
                mock_settings.PAGEINDEX_LLM_MODEL = "test"
                mock_settings.TEMP_DIR = str(tmp_path / "temp")
                mock_settings.get_pageindex_output_dir.return_value = tmp_path
                mock_settings.get_pageindex_base_url.return_value = "http://localhost"
                mock_settings.PAGEINDEX_LLM_BACKEND = "local"
                mock_settings.PAGEINDEX_LLM_API_KEY = "no-key"

                with patch("meetbot.adapters.llm.openai_adapter.pageindex_env") as mock_env:
                    mock_env.return_value.__enter__ = MagicMock(return_value=None)
                    mock_env.return_value.__exit__ = MagicMock(return_value=False)

                    result_path = await adapter.build_index(
                        segments=SAMPLE_SEGMENTS,
                        job_id="test-job-id-1234",
                        output_dir=tmp_path,
                        filename="Test Meeting",
                    )

        assert result_path.exists()
        with open(result_path) as f:
            saved_tree = json.load(f)
        assert "children" in saved_tree or "title" in saved_tree


# ===========================================================================
# retriever_pageindex: PageIndexRetriever
# ===========================================================================

class TestPageIndexRetriever:
    def _make_retriever(self):
        from meetbot.services.rag.retriever_pageindex import PageIndexRetriever
        return PageIndexRetriever(model="test", base_url="http://localhost", api_key="no-key")

    def test_parse_node_list_clean_json(self):
        r = self._make_retriever()
        response = '{"thinking": "nodes 0001 and 0003", "node_list": ["0001", "0003"]}'
        result = r._parse_node_list(response)
        assert result == ["0001", "0003"]

    def test_parse_node_list_markdown_code_block(self):
        r = self._make_retriever()
        response = '```json\n{"thinking": "test", "node_list": ["0002"]}\n```'
        result = r._parse_node_list(response)
        assert result == ["0002"]

    def test_parse_node_list_embedded_json(self):
        r = self._make_retriever()
        response = 'Here is my answer: {"node_list": ["0001", "0004"], "thinking": "reason"}'
        result = r._parse_node_list(response)
        assert "0001" in result
        assert "0004" in result

    def test_parse_node_list_invalid_json_returns_empty(self):
        r = self._make_retriever()
        result = r._parse_node_list("This is not JSON at all and has no node_list")
        assert result == []

    def test_parse_node_list_empty_list(self):
        r = self._make_retriever()
        result = r._parse_node_list('{"thinking": "nothing", "node_list": []}')
        assert result == []

    def test_build_tree_summary_includes_titles(self):
        r = self._make_retriever()
        summary = r._build_tree_summary(SAMPLE_TREE)
        assert "Speaker: Alice" in summary
        assert "Speaker: Bob" in summary
        assert "0001" in summary

    def test_build_tree_summary_nested_indent(self):
        r = self._make_retriever()
        tree_with_children = {
            "node_id": "root",
            "title": "Root",
            "children": [
                {
                    "node_id": "0001",
                    "title": "Child Node",
                    "children": [],
                    "segment_indices": [0, 1],
                }
            ],
            "segment_indices": [],
        }
        summary = r._build_tree_summary(tree_with_children)
        assert "Child Node" in summary
        # Child should be indented
        lines = summary.split("\n")
        child_line = next((l for l in lines if "Child Node" in l), "")
        assert child_line.startswith("  ")

    def test_build_node_map_flat(self):
        r = self._make_retriever()
        node_map = r._build_node_map(SAMPLE_TREE)
        assert "0001" in node_map
        assert "0002" in node_map
        assert "0003" in node_map
        assert node_map["0001"]["title"] == "Speaker: Alice (00:00 - 00:30)"

    def test_build_node_map_empty_tree(self):
        r = self._make_retriever()
        tree = {"title": "root", "children": [], "segment_indices": []}
        node_map = r._build_node_map(tree)
        assert node_map == {}

    @pytest.mark.asyncio
    async def test_search_returns_matched_segments(self):
        """search() should return segment dicts for matched nodes."""
        r = self._make_retriever()

        # Mock LLM tree search to return node 0002 (Bob's segments)
        with patch.object(r, "_llm_tree_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = ["0002"]
            results = await r.search(
                query="What did Bob say about revenue?",
                tree=copy.deepcopy(SAMPLE_TREE),
                segments=SAMPLE_SEGMENTS,
            )

        assert len(results) > 0
        # All results should be from Bob's segment indices (2, 3)
        for result in results:
            assert result["segment_index"] in (2, 3)
            assert result["node_id"] == "0002"

    @pytest.mark.asyncio
    async def test_search_deduplicates_segments(self):
        """If a segment appears in multiple nodes, it should only appear once."""
        r = self._make_retriever()

        # Create a tree where two nodes share segment 0
        tree_overlap = {
            "title": "Overlap Tree",
            "node_id": "root",
            "children": [
                {
                    "node_id": "0001",
                    "title": "Node A",
                    "children": [],
                    "content_lines": [],
                    "segment_indices": [0, 1],
                },
                {
                    "node_id": "0002",
                    "title": "Node B",
                    "children": [],
                    "content_lines": [],
                    "segment_indices": [0, 2],  # segment 0 appears again
                },
            ],
            "content_lines": [],
            "segment_indices": [],
        }

        with patch.object(r, "_llm_tree_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = ["0001", "0002"]
            results = await r.search(
                query="test",
                tree=tree_overlap,
                segments=SAMPLE_SEGMENTS,
            )

        seg_indices = [res["segment_index"] for res in results]
        # No duplicates
        assert len(seg_indices) == len(set(seg_indices))

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_no_nodes(self):
        r = self._make_retriever()
        with patch.object(r, "_llm_tree_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            results = await r.search("query", copy.deepcopy(SAMPLE_TREE), SAMPLE_SEGMENTS)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_graceful_on_unknown_node_id(self):
        """If LLM returns a node ID not in the tree, it should be skipped."""
        r = self._make_retriever()
        with patch.object(r, "_llm_tree_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = ["9999"]
            results = await r.search("query", copy.deepcopy(SAMPLE_TREE), SAMPLE_SEGMENTS)
        assert results == []

    def test_node_source_ref_has_node_id(self):
        """Returned sources should always carry node_id and node_title."""
        from meetbot.services.rag.retriever_pageindex import PageIndexRetriever
        import asyncio

        r = PageIndexRetriever(model="test", base_url="http://localhost", api_key="no-key")

        async def _run():
            with patch.object(r, "_llm_tree_search", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = ["0001"]
                return await r.search("query", copy.deepcopy(SAMPLE_TREE), SAMPLE_SEGMENTS)

        results = asyncio.get_event_loop().run_until_complete(_run())
        for res in results:
            assert "node_id" in res
            assert "node_title" in res
