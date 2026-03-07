#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# scripts/restore_legacy.sh
#
# Restore the legacy RAG pipeline from the archive tarball.
# This script is intended for emergency rollback only.
#
# Usage:
#   bash scripts/restore_legacy.sh /path/to/legacy-rag-archive-YYYY-MM-DD.tgz
#
# After restoring:
#   1. Set RAG_V2_ENABLED=false in your environment
#   2. Restart the server
#
# The archive contains the following files as they existed before removal:
#   - meetbot/services/indexer.py          (legacy IndexerService)
#   - meetbot/services/prepare_docs.py     (legacy PrepareDocsService)
#   - meetbot/workers/reindex_worker.py    (with _run_reindex_legacy)
#   - meetbot/services/query_service.py    (with legacy query_stream)
#   - meetbot/web/pages/query.py           (with RAG_V2_ENABLED branching)
#   - meetbot/web/ws_chat.py              (with legacy query_stream call)
#   - meetbot/config.py                   (with active RAG_V2_ENABLED flag)
# ──────────────────────────────────────────────────────────────

set -euo pipefail

ARCHIVE="${1:-}"

if [[ -z "$ARCHIVE" ]]; then
    echo "Usage: bash scripts/restore_legacy.sh <archive.tgz>"
    echo ""
    echo "To create the archive before removal, run:"
    echo "  bash scripts/create_legacy_archive.sh"
    exit 1
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: Archive not found: $ARCHIVE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Restoring legacy RAG files from: $ARCHIVE"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "WARNING: This will overwrite the following files:"
echo "  - meetbot/services/indexer.py"
echo "  - meetbot/services/prepare_docs.py"
echo "  - meetbot/workers/reindex_worker.py"
echo "  - meetbot/services/query_service.py"
echo "  - meetbot/web/pages/query.py"
echo "  - meetbot/web/ws_chat.py"
echo "  - meetbot/config.py"
echo ""
read -p "Continue? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

cd "$PROJECT_ROOT"
tar -xzf "$ARCHIVE"

echo ""
echo "✓ Legacy files restored."
echo ""
echo "Next steps:"
echo "  1. export RAG_V2_ENABLED=false"
echo "  2. Restart the server"
echo "  3. Verify legacy pipeline works"
