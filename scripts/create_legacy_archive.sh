#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# scripts/create_legacy_archive.sh
#
# Archive legacy RAG pipeline files before removal.
# Run this BEFORE deleting legacy code.
#
# Usage:
#   cd MeetBot && bash scripts/create_legacy_archive.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATE=$(date +%F)
ARCHIVE="${PROJECT_ROOT}/legacy-rag-archive-${DATE}.tgz"

cd "$PROJECT_ROOT"

FILES_TO_ARCHIVE=(
    "meetbot/services/indexer.py"
    "meetbot/services/prepare_docs.py"
    "meetbot/workers/reindex_worker.py"
    "meetbot/services/query_service.py"
    "meetbot/web/pages/query.py"
    "meetbot/web/ws_chat.py"
    "meetbot/config.py"
)

# Filter to only files that exist
EXISTING=()
for f in "${FILES_TO_ARCHIVE[@]}"; do
    if [[ -f "$f" ]]; then
        EXISTING+=("$f")
    else
        echo "WARN: $f does not exist, skipping"
    fi
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
    echo "ERROR: No files to archive"
    exit 1
fi

tar -czf "$ARCHIVE" "${EXISTING[@]}"

echo "✓ Legacy archive created: $ARCHIVE"
echo "  Contains ${#EXISTING[@]} files"
echo ""
echo "To restore later:"
echo "  bash scripts/restore_legacy.sh $ARCHIVE"
