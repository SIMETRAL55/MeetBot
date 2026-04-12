"""
Tests for incremental ChromaDB reindex (per-segment content_hash skip).

Covers:
(a) Unchanged segment: content_hash matches → embedding carried over,
    build_multilevel_index_atomic receives it in precomputed_embeddings
(b) Changed segment: content_hash differs → excluded from precomputed,
    must be re-embedded by the indexer
(c) New segment: no stored hash (empty string) → not in precomputed,
    must be freshly embedded

All tests mock Chroma and the worker's heavy dependencies to avoid GPU/disk.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JOB_ID = "incr0000-aaaa-bbbb-cccc-dddddddddddd"
COLLECTION = JOB_ID[:8]  # "incr0000"
_WORKER_BASE = "meetbot.workers.reindex_worker"


def _content_hash(text: str, speaker: str) -> str:
    return hashlib.sha256((text + speaker).encode()).hexdigest()


def _make_segment(idx: int, text: str, speaker: str, content_hash: str = "") -> MagicMock:
    seg = MagicMock()
    seg.segment_index = idx
    seg.content_hash = content_hash
    seg.to_dict.return_value = {
        "segment_index": idx,
        "start": float(idx),
        "end": float(idx + 1),
        "speaker": speaker,
        "text": text,
    }
    return seg


def _make_job(segments_hash: str = "") -> MagicMock:
    job = MagicMock()
    job.result_json_path = "/fake/result.json"
    job.segments_hash = segments_hash
    job.original_filename = "meeting.mp3"
    job.transcript_version = 1
    job.pageindex_path = None
    job.pageindex_status = None
    return job


def _make_chroma_result(entries: list[tuple[str, list[float], dict]]) -> dict:
    """Build the dict returned by collection.get(include=['embeddings','metadatas'])."""
    ids, embeddings, metadatas = [], [], []
    for doc_id, emb, meta in entries:
        ids.append(doc_id)
        embeddings.append(emb)
        metadatas.append(meta)
    return {"ids": ids, "embeddings": embeddings, "metadatas": metadatas}


def _base_patches(job, segments, mock_path_exists=True):
    """Return the common set of patches needed for all incremental tests."""
    db = MagicMock()
    db.close = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    session_factory = MagicMock(return_value=db)
    return db, session_factory


# ---------------------------------------------------------------------------
# (a) Unchanged segments → precomputed_embeddings carries them over
# ---------------------------------------------------------------------------


def test_unchanged_segments_are_carried_over():
    """
    Both segments have matching content_hash stored in DB.
    Live Chroma returns embeddings for both.
    build_multilevel_index_atomic must be called with precomputed_embeddings
    containing both segment doc IDs.
    """
    text_a, spk_a = "Hello world", "SPEAKER_00"
    text_b, spk_b = "How are you", "SPEAKER_01"

    hash_a = _content_hash(text_a, spk_a)
    hash_b = _content_hash(text_b, spk_b)

    seg_a = _make_segment(0, text_a, spk_a, content_hash=hash_a)
    seg_b = _make_segment(1, text_b, spk_b, content_hash=hash_b)
    segments = [seg_a, seg_b]

    job = _make_job(segments_hash="")  # job-level hash differs → won't skip
    db, session_factory = _base_patches(job, segments)

    emb_a = [0.1, 0.2, 0.3]
    emb_b = [0.4, 0.5, 0.6]

    chroma_result = _make_chroma_result([
        (f"{COLLECTION}_seg_0", emb_a, {"level": "segment", "segment_id": 0}),
        (f"{COLLECTION}_seg_1", emb_b, {"level": "segment", "segment_id": 1}),
    ])

    mock_col = MagicMock()
    mock_col.get.return_value = chroma_result
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_col

    captured = {}

    def fake_build_multilevel(**kwargs):
        captured["precomputed"] = kwargs.get("precomputed_embeddings")
        return "/fake/chroma"

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status"),
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=2),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.db.crud.update_segments_content_hash"),
        patch(
            "meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
            side_effect=fake_build_multilevel,
        ),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
        patch("chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("chromadb.Settings", return_value=MagicMock()),
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = f"/fake/chroma/{COLLECTION}"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20

        # result_json_path exists; live_dir exists
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.return_value.__truediv__ = lambda self, other: mock_path_instance

        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()
        mock_pm.update = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    precomputed = captured.get("precomputed") or {}
    assert f"{COLLECTION}_seg_0" in precomputed, "seg_0 should be carried over (hash matched)"
    assert f"{COLLECTION}_seg_1" in precomputed, "seg_1 should be carried over (hash matched)"
    assert precomputed[f"{COLLECTION}_seg_0"] == emb_a
    assert precomputed[f"{COLLECTION}_seg_1"] == emb_b


# ---------------------------------------------------------------------------
# (b) Changed segment → excluded from precomputed
# ---------------------------------------------------------------------------


def test_changed_segment_excluded_from_precomputed():
    """
    seg_0 has a matching hash; seg_1 has a *stale* stored hash (it was edited).
    Only seg_0 should appear in precomputed_embeddings.
    """
    text_a, spk_a = "Unchanged text", "SPEAKER_00"
    text_b, spk_b = "Edited text now", "SPEAKER_01"

    hash_a = _content_hash(text_a, spk_a)
    stale_hash = _content_hash("Original text before edit", spk_b)  # outdated

    seg_a = _make_segment(0, text_a, spk_a, content_hash=hash_a)
    seg_b = _make_segment(1, text_b, spk_b, content_hash=stale_hash)  # stored hash is stale
    segments = [seg_a, seg_b]

    job = _make_job(segments_hash="")
    db, session_factory = _base_patches(job, segments)

    emb_a = [1.0, 2.0, 3.0]
    emb_b_old = [9.9, 8.8, 7.7]  # stale; should NOT be reused

    chroma_result = _make_chroma_result([
        (f"{COLLECTION}_seg_0", emb_a, {"level": "segment", "segment_id": 0}),
        (f"{COLLECTION}_seg_1", emb_b_old, {"level": "segment", "segment_id": 1}),
    ])

    mock_col = MagicMock()
    mock_col.get.return_value = chroma_result
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_col

    captured = {}

    def fake_build_multilevel(**kwargs):
        captured["precomputed"] = kwargs.get("precomputed_embeddings")
        return "/fake/chroma"

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status"),
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=2),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.db.crud.update_segments_content_hash"),
        patch(
            "meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
            side_effect=fake_build_multilevel,
        ),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
        patch("chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("chromadb.Settings", return_value=MagicMock()),
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = f"/fake/chroma/{COLLECTION}"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.return_value.__truediv__ = lambda self, other: mock_path_instance

        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()
        mock_pm.update = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    precomputed = captured.get("precomputed") or {}
    assert f"{COLLECTION}_seg_0" in precomputed, "unchanged seg_0 should be carried over"
    assert f"{COLLECTION}_seg_1" not in precomputed, "edited seg_1 must NOT be carried over"


# ---------------------------------------------------------------------------
# (c) New segment (no stored hash) → not in precomputed
# ---------------------------------------------------------------------------


def test_new_segment_not_in_precomputed():
    """
    seg_0 and seg_1 have matching hashes (existing); seg_2 is brand-new (empty hash).
    Live Chroma has only seg_0 and seg_1.
    precomputed_embeddings should contain seg_0 and seg_1 but NOT seg_2.
    """
    text_a, spk_a = "Old segment A", "SPEAKER_00"
    text_b, spk_b = "Old segment B", "SPEAKER_01"
    text_c, spk_c = "Brand new segment", "SPEAKER_00"

    hash_a = _content_hash(text_a, spk_a)
    hash_b = _content_hash(text_b, spk_b)

    seg_a = _make_segment(0, text_a, spk_a, content_hash=hash_a)
    seg_b = _make_segment(1, text_b, spk_b, content_hash=hash_b)
    seg_c = _make_segment(2, text_c, spk_c, content_hash="")  # new, no hash stored
    segments = [seg_a, seg_b, seg_c]

    job = _make_job(segments_hash="")
    db, session_factory = _base_patches(job, segments)

    emb_a = [0.1, 0.2]
    emb_b = [0.3, 0.4]

    chroma_result = _make_chroma_result([
        (f"{COLLECTION}_seg_0", emb_a, {"level": "segment", "segment_id": 0}),
        (f"{COLLECTION}_seg_1", emb_b, {"level": "segment", "segment_id": 1}),
        # seg_2 does NOT exist in live collection
    ])

    mock_col = MagicMock()
    mock_col.get.return_value = chroma_result
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_col

    captured = {}

    def fake_build_multilevel(**kwargs):
        captured["precomputed"] = kwargs.get("precomputed_embeddings")
        return "/fake/chroma"

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status"),
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=segments),
        patch("meetbot.db.crud.bump_transcript_version", return_value=2),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.db.crud.update_segments_content_hash"),
        patch(
            "meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
            side_effect=fake_build_multilevel,
        ),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
        patch("chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("chromadb.Settings", return_value=MagicMock()),
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = f"/fake/chroma/{COLLECTION}"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.return_value.__truediv__ = lambda self, other: mock_path_instance

        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()
        mock_pm.update = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    precomputed = captured.get("precomputed") or {}
    assert f"{COLLECTION}_seg_0" in precomputed, "existing seg_0 should be carried over"
    assert f"{COLLECTION}_seg_1" in precomputed, "existing seg_1 should be carried over"
    assert f"{COLLECTION}_seg_2" not in precomputed, "new seg_2 must NOT be in precomputed"


# ---------------------------------------------------------------------------
# (d) content_hash persisted after successful reindex
# ---------------------------------------------------------------------------


def test_content_hashes_persisted_after_reindex():
    """
    After a successful reindex, update_segments_content_hash should be called
    with the correct {segment_index: sha256_hex} mapping.
    """
    text, spk = "Some text", "SPEAKER_00"
    expected_hash = _content_hash(text, spk)

    seg = _make_segment(0, text, spk, content_hash="")  # not yet hashed
    job = _make_job()
    db, session_factory = _base_patches(job, [seg])

    with (
        patch(f"{_WORKER_BASE}.get_session", return_value=session_factory),
        patch(f"{_WORKER_BASE}.get_job", return_value=job),
        patch(f"{_WORKER_BASE}.update_job_status"),
        patch(f"{_WORKER_BASE}.update_job_result"),
        patch(f"{_WORKER_BASE}.progress_manager") as mock_pm,
        patch(f"{_WORKER_BASE}.cancel_registry") as mock_cancel,
        patch(f"{_WORKER_BASE}.Path") as mock_path_cls,
        patch("meetbot.db.crud.get_segments_for_job", return_value=[seg]),
        patch("meetbot.db.crud.bump_transcript_version", return_value=1),
        patch("meetbot.db.crud.flush_segments_to_json"),
        patch("meetbot.db.crud.update_segments_content_hash") as mock_update_hashes,
        patch(
            "meetbot.services.rag.indexer.RAGIndexer.build_multilevel_index_atomic",
            return_value="/fake/chroma",
        ),
        patch("meetbot.workers.reindex_worker._cleanup_gpu"),
        patch("meetbot.workers.reindex_worker.settings") as mock_settings,
        patch("chromadb.PersistentClient", side_effect=Exception("no live collection")),
        patch("chromadb.Settings", return_value=MagicMock()),
    ):
        mock_settings.PAGEINDEX_ENABLED = False
        mock_settings.VECTOR_DB_PATH = f"/fake/chroma/{COLLECTION}"
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.EMBEDDING_DEVICE = "cpu"
        mock_settings.CHUNK_TOKENS = 200
        mock_settings.CHUNK_OVERLAP = 20

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance
        mock_path_cls.return_value.__truediv__ = lambda self, other: mock_path_instance

        mock_cancel.check_and_raise = MagicMock()
        mock_pm.make_callback.return_value = MagicMock()
        mock_pm.update = MagicMock()

        from meetbot.workers.reindex_worker import run_reindex
        run_reindex(JOB_ID)

    mock_update_hashes.assert_called_once()
    _, kwargs_or_args = mock_update_hashes.call_args[0], mock_update_hashes.call_args
    # Extract the hashes argument (3rd positional or keyword)
    call_args = mock_update_hashes.call_args
    hashes_arg = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("hashes")
    assert hashes_arg[0] == expected_hash, "segment_index 0 should have correct content hash"
