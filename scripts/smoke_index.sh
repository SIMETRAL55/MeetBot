#!/usr/bin/env bash
# ── scripts/smoke_index.sh ──────────────────────────────────────────────
# Smoke test: exercises the batched indexing path with synthetic data.
#
# Usage:
#   bash scripts/smoke_index.sh          # uses default venv
#   PYTHON=/path/to/python scripts/smoke_index.sh
#
# What it does:
#   1. Generates a large synthetic JSONL of transcript segments.
#   2. Invokes build_multilevel_index_atomic via a small Python script.
#   3. Validates the resulting Chroma index exists and contains docs.
#   4. Cleans up temp artefacts.
#
# Exit codes:
#   0  — success
#   1  — indexing failed or output validation failed
# ────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-${SCRIPT_DIR}/../venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

SMOKE_DIR="${SCRIPT_DIR}/temp/smoke_test_$$"
mkdir -p "$SMOKE_DIR"

echo "=== MeetBot Memory-Safe Indexing Smoke Test ==="
echo "  Python:   $PYTHON"
echo "  Work dir: $SMOKE_DIR"
echo ""

# ── Generate synthetic transcript segments ─────────────────────────────
N_SEGMENTS=200
echo "Generating $N_SEGMENTS synthetic segments..."

"$PYTHON" -c "
import json, sys, os
sys.path.insert(0, '${SCRIPT_DIR}')
os.environ.setdefault('MEETBOT_DB_URL', 'sqlite:///temp/smoke_db.sqlite3')

segs = []
for i in range($N_SEGMENTS):
    segs.append({
        'text': f'Speaker {i % 5} says sentence number {i}. ' * 8,
        'speaker': f'SPEAKER_{i % 5:02d}',
        'start': float(i * 3),
        'end': float(i * 3 + 2.5),
    })

# Write segments to JSON
with open('${SMOKE_DIR}/segments.json', 'w') as f:
    json.dump(segs, f)

print(f'  Generated {len(segs)} segments ({os.path.getsize(\"${SMOKE_DIR}/segments.json\")} bytes)')
"

# ── Run batched indexing ───────────────────────────────────────────────
echo ""
echo "Running batched multilevel indexing (EMBED_BATCH_SIZE=8)..."

export EMBED_BATCH_SIZE=8
export MEMORY_WATCH_ENABLED=true
export TEMP_DIR="${SMOKE_DIR}/temp"
export MEETBOT_DB_URL="sqlite:///${SMOKE_DIR}/smoke_db.sqlite3"

"$PYTHON" -c "
import json, sys, os, time
sys.path.insert(0, '${SCRIPT_DIR}')

from meetbot.services.rag.indexer import RAGIndexer

with open('${SMOKE_DIR}/segments.json') as f:
    segments = json.load(f)

progress_log = []
def progress_cb(stage, pct, msg):
    progress_log.append((stage, pct, msg))
    print(f'  [{stage}] {pct:5.1f}%  {msg}')

start = time.monotonic()

try:
    result = RAGIndexer.build_multilevel_index_atomic(
        segments=segments,
        persist_root='${SMOKE_DIR}/db',
        collection_name='smoke',
        embedding_model=os.environ.get('EMBEDDING_MODEL', '${SCRIPT_DIR}/models/sarashina-embedding-v1-1b'),
        device='cpu',
        progress_callback=progress_cb,
        job_id='smoke_test_job',
        version=1,
    )
    elapsed = time.monotonic() - start
    print(f'')
    print(f'  ✓ Index built at: {result}')
    print(f'  ✓ Elapsed: {elapsed:.1f}s')

    doc_count = RAGIndexer.get_doc_count(result)
    print(f'  ✓ Chunk-level doc count: {doc_count}')

    if doc_count == 0:
        print('  ✗ ERROR: No docs in index!')
        sys.exit(1)

    print(f'  ✓ Progress callbacks: {len(progress_log)}')
    print(f'')
    print('  ✓ SMOKE TEST PASSED')

except Exception as e:
    print(f'  ✗ INDEXING FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || {
    echo "  ✗ Smoke test FAILED"
    rm -rf "$SMOKE_DIR"
    exit 1
}

# ── Cleanup ────────────────────────────────────────────────────────────
echo ""
echo "Cleaning up $SMOKE_DIR..."
rm -rf "$SMOKE_DIR"
echo "Done."
