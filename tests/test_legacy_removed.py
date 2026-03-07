"""Guard tests: verify legacy RAG pipeline is fully removed from active code paths.

These tests scan the codebase to ensure no active (non-deprecated) module still
imports or references the removed legacy RAG components.  They act as a safety
net to prevent accidental reintroduction of legacy code.
"""

import ast
import textwrap
from pathlib import Path

import pytest

# Root of the MeetBot package
_PKG_ROOT = Path(__file__).resolve().parent.parent / "meetbot"

# Legacy symbols that must NOT appear in active modules
_LEGACY_SYMBOLS = {
    "IndexerService",
    "PrepareDocsService",
    "_run_reindex_legacy",
    "query_stream_v2",
}

# Modules that are themselves deprecated — they may reference legacy symbols
_DEPRECATED_MODULES = {
    "services/indexer.py",
    "services/prepare_docs.py",
}

# Test files are excluded — they may reference legacy names in guard tests
_EXCLUDED_DIRS = {"__pycache__", ".git"}


def _active_py_files():
    """Yield (relative_path_str, full_path) for all non-deprecated .py files."""
    for p in _PKG_ROOT.rglob("*.py"):
        rel = p.relative_to(_PKG_ROOT).as_posix()
        if any(part in _EXCLUDED_DIRS for part in p.parts):
            continue
        if rel in _DEPRECATED_MODULES:
            continue
        yield rel, p


class TestLegacyRemoved:
    """Verify no active code path references legacy RAG components."""

    @pytest.mark.parametrize(
        "symbol", sorted(_LEGACY_SYMBOLS),
        ids=sorted(_LEGACY_SYMBOLS),
    )
    def test_no_legacy_symbol_in_active_code(self, symbol: str):
        """Ensure *symbol* does not appear in any active module source."""
        violations = []
        for rel, path in _active_py_files():
            source = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(source.splitlines(), 1):
                if symbol in line:
                    violations.append(f"  {rel}:{lineno}: {line.strip()}")
        if violations:
            msg = (
                f"Legacy symbol '{symbol}' found in active modules:\n"
                + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_no_feature_flag_dispatch(self):
        """RAG_V2_ENABLED must not be read for branching in active code."""
        violations = []
        for rel, path in _active_py_files():
            # Skip the config definition itself
            if rel == "config.py":
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(source.splitlines(), 1):
                if "RAG_V2_ENABLED" in line:
                    violations.append(f"  {rel}:{lineno}: {line.strip()}")
        if violations:
            msg = (
                "RAG_V2_ENABLED used in active code (should only be in config.py):\n"
                + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_deprecated_modules_have_warning(self):
        """Deprecated modules must emit a DeprecationWarning on import."""
        for rel in _DEPRECATED_MODULES:
            path = _PKG_ROOT / rel
            if not path.exists():
                continue  # Already deleted
            source = path.read_text(encoding="utf-8")
            assert "DeprecationWarning" in source, (
                f"Deprecated module {rel} lacks a DeprecationWarning"
            )
