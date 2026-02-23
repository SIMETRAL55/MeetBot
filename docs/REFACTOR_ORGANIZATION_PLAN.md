# MeetBot Refactor & Organization Plan

## Why the current layout feels hard to maintain

The current code works, but there are a few structural patterns that make changes risky:

1. **Everything sits as flat scripts under `src/source/`**
   - Pipeline logic, model adapters, CLI handling, and utility code are mixed at the same level.
2. **Core modules contain CLI + business logic together**
   - `prepare_docs.py`, `build_index.py`, and `query.py` all define reusable logic and their own command-line entrypoint in the same file.
3. **Module boundaries are blurry**
   - `HFInferenceClient` handles both remote Whisper calls and local pyannote execution in one class.
   - `transcribe.py` contains pipeline logic and response normalization.
4. **Inconsistent output contract**
   - `align.format_result_as_json()` writes one object (`transcript`) but returns another (`out`), which can cause confusion at call sites.
5. **No tests or package structure**
   - No dedicated test tree and no installable package namespace, which slows down safe iteration.

## Target architecture (recommended)

Move from script-centric code to a package-oriented layout:

```text
src/
  meetbot/
    __init__.py
    config/
      settings.py
    cli/
      app.py
      transcribe_cmd.py
      diarize_cmd.py
      pipeline_cmd.py
      rag_prepare_cmd.py
      rag_index_cmd.py
      rag_query_cmd.py
    domain/
      models.py              # typed DTOs / dataclasses
      contracts.py           # protocol interfaces
    services/
      pipeline_service.py    # transcribe + diarize + align orchestration
      alignment_service.py
      document_prep_service.py
      indexing_service.py
      query_service.py
    adapters/
      asr/
        hf_whisper.py
      diarization/
        pyannote_local.py
      llm/
        hf_chat.py
      vectorstore/
        chroma_store.py
    infra/
      cache/
        hf_cache.py
      audio/
        convert.py
        chunking.py
      logging.py
    tests/
      unit/
      integration/
```

## Refactor roadmap (safe, incremental)

### Phase 1 — Stabilize interfaces (no behavior change)

- Define typed result shapes (`TranscriptionResult`, `DiarizationResult`, `AlignedSegment`) in `domain/models.py`.
- Wrap current top-level functions behind service methods.
- Keep existing scripts but make them thin wrappers over services.

### Phase 2 — Split by responsibility

- Extract `HFInferenceClient` into narrower adapters:
  - `adapters/asr/hf_whisper.py`
  - `adapters/diarization/pyannote_local.py`
- Move chunking and cache into `infra/`.
- Move alignment logic from `align.py` into `services/alignment_service.py`.

### Phase 3 — Normalize I/O and contracts

- Standardize all service returns to typed objects or explicit dict schemas.
- Fix output contract mismatch in alignment output writer (write and return same structure).
- Remove print-heavy diagnostics in favor of structured logging.

### Phase 4 — CLI and developer ergonomics

- Consolidate command-line entrypoints under a single CLI (`meetbot ...`) using `argparse` subcommands or `typer`.
- Add a root `README.md` with quickstart and command examples.
- Introduce `pyproject.toml` and package install (`pip install -e .`).

### Phase 5 — Testing and quality gates

- Add unit tests for:
  - alignment overlap/splitting
  - cache key stability
  - transcript normalization logic
- Add integration smoke tests for pipeline using sample audio.
- Add lint/format/type checks (`ruff`, `black`, optionally `mypy`).

## Practical mapping from current files

- `main.py` -> `cli/pipeline_cmd.py` + `services/pipeline_service.py`
- `transcribe.py` -> `services/pipeline_service.py` + `adapters/asr/hf_whisper.py` + `infra/audio/chunking.py`
- `diarize.py` -> `adapters/diarization/pyannote_local.py` + `services/pipeline_service.py`
- `align.py` -> `services/alignment_service.py`
- `prepare_docs.py` -> `services/document_prep_service.py` + `cli/rag_prepare_cmd.py`
- `build_index.py` -> `services/indexing_service.py` + `adapters/vectorstore/chroma_store.py` + `cli/rag_index_cmd.py`
- `query.py` -> `services/query_service.py` + `adapters/llm/hf_chat.py` + `cli/rag_query_cmd.py`
- `config.py` -> `config/settings.py`

## Immediate quick wins (can do in 1-2 PRs)

1. Create `src/meetbot/` package and move `config.py` + `utils/` first.
2. Add thin wrappers so old script paths still work while refactor is in progress.
3. Fix alignment output consistency and remove dead/commented code blocks.
4. Add at least 3 unit tests around alignment and cache behavior.

## Definition of done for the refactor

- Single package namespace (`meetbot`) with clear layers (cli/services/adapters/infra/domain).
- No business logic in CLI files.
- All data handoffs use typed models or explicit schemas.
- Legacy script entrypoints either removed or reduced to compatibility shims.
- Core modules covered by automated tests and basic CI checks.
