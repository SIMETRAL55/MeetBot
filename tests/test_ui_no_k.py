"""
Guard tests for the query UI — ensure the retrieval level controls
are inside the Retrieval expansion and that the old k-input has been
removed.

These tests parse the source code of the query page rather than
running the NiceGUI runtime so they have zero UI-framework deps.
"""

import ast
import inspect
import textwrap
from pathlib import Path


# Path to the UI source we are guarding
_QUERY_PAGE = Path(__file__).parent.parent / "meetbot" / "web" / "pages" / "query.py"


def _source() -> str:
    return _QUERY_PAGE.read_text(encoding="utf-8")


class TestKInputHiddenBehindAdvanced:
    """The old k-input should be gone; retrieval controls should exist."""

    def test_expansion_advanced_present(self):
        """The page must contain a ui.expansion call with 'Retrieval' label."""
        src = _source()
        assert 'ui.expansion("Retrieval"' in src or "ui.expansion('Retrieval'" in src, (
            "query.py does not contain a ui.expansion('Retrieval') call — "
            "the retrieval controls must be wrapped in a Retrieval expansion."
        )

    def test_rag_k_input_defined_after_expansion(self):
        """retrieval_level_radio must be assigned after the expansion opener."""
        src = _source()
        expansion_pos = src.find('ui.expansion("Retrieval"')
        if expansion_pos == -1:
            expansion_pos = src.find("ui.expansion('Retrieval'")
        assert expansion_pos != -1, "ui.expansion('Retrieval') not found"

        radio_pos = src.find("retrieval_level_radio = ui.radio")
        assert radio_pos != -1, "retrieval_level_radio = ui.radio not found"

        assert radio_pos > expansion_pos, (
            f"retrieval_level_radio (pos={radio_pos}) appears *before* "
            f"ui.expansion (pos={expansion_pos}). "
            "The retrieval controls must be inside the Retrieval expansion."
        )

    def test_k_input_indented_inside_expansion(self):
        """retrieval_level_radio must be more deeply indented than the expansion."""
        src = _source()
        lines = src.splitlines()

        expansion_indent = None
        radio_indent = None

        for line in lines:
            stripped = line.lstrip()
            if 'ui.expansion("Retrieval"' in stripped or "ui.expansion('Retrieval'" in stripped:
                expansion_indent = len(line) - len(stripped)
            if "retrieval_level_radio = ui.radio" in line:
                radio_indent = len(line) - len(stripped)

        assert expansion_indent is not None, "Could not find expansion line"
        assert radio_indent is not None, "Could not find retrieval_level_radio line"
        assert radio_indent > expansion_indent, (
            f"retrieval_level_radio indent ({radio_indent}) must be greater than "
            f"expansion indent ({expansion_indent}) — it must be nested inside."
        )

    def test_no_bare_k_row_outside_expansion(self):
        """The old rag_k_input should no longer exist in the source."""
        src = _source()
        assert "rag_k_input" not in src, (
            "rag_k_input still exists in query.py — "
            "it should have been replaced by retrieval_level_radio."
        )


class TestRetrievalLevelUIControls:
    """Verify the retrieval level UI controls are properly defined."""

    def test_has_retrieval_level_radio(self):
        """The page defines a retrieval_level_radio widget."""
        src = _source()
        assert "retrieval_level_radio" in src

    def test_has_segment_count_input(self):
        """The page defines a segment_count_input widget."""
        src = _source()
        assert "segment_count_input" in src

    def test_has_document_option(self):
        """The retrieval options include 'Document'."""
        src = _source()
        assert '"document"' in src or "'document'" in src

    def test_has_segment_option(self):
        """The retrieval options include 'Segment'."""
        src = _source()
        assert '"segment"' in src or "'segment'" in src

    def test_has_chunk_option(self):
        """The retrieval options include 'Chunk'."""
        src = _source()
        assert '"chunk"' in src or "'chunk'" in src

    def test_sends_retrieval_level_to_query_stream(self):
        """The stream call must pass retrieval_level."""
        src = _source()
        assert "retrieval_level=" in src
