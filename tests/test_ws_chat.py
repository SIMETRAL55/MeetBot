"""
Tests for meetbot/web/ws_chat.py.

Covers:
- Valid query → done event → ChatMessage saved with status="completed"
- Invalid job_id → error message sent, close(4004)
- Job not COMPLETED → error message, close(4009)
- retrieval_method="pageindex" is forwarded to QueryService
- Missing question in payload → error, close(4010)
- Empty question → error, close(4010)
"""

import asyncio
import json
from typing import Any, Dict, Iterator
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from meetbot.db.models import JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_websocket(*recv_messages: str) -> AsyncMock:
    """Build a minimal mock WebSocket that returns the given receive_text values."""
    ws = AsyncMock()
    ws.receive_text.side_effect = list(recv_messages)
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_job(
    status: JobStatus = JobStatus.COMPLETED,
    db_dir: str = "/tmp/testdb",
    original_filename: str = "meeting.mp3",
) -> MagicMock:
    job = MagicMock()
    job.status = status
    job.db_dir = db_dir
    job.original_filename = original_filename
    return job


def _make_chat_session(session_id: str = "sess-001") -> MagicMock:
    sess = MagicMock()
    sess.id = session_id
    return sess


def _simple_stream(answer: str = "Hello world") -> Iterator[Dict[str, Any]]:
    """A minimal streaming generator that yields sources → token → done."""
    yield {"type": "sources", "data": []}
    yield {"type": "token", "data": answer}
    yield {
        "type": "done",
        "llm_backend": "hf",
        "full_answer": answer,
        "filtered_sources": [],
    }


# Common patch targets
_BASE = "meetbot.web.ws_chat"
_GET_SESSION = f"{_BASE}.get_session"
_GET_JOB = f"{_BASE}.get_job"
_GET_OR_CREATE = f"{_BASE}.get_or_create_chat_session"
_CREATE_MSG = f"{_BASE}.create_chat_message"
_QUERY_SVC = "meetbot.services.query_service.QueryService"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_not_found_sends_error_and_closes():
    """If job_id doesn't exist, send error + close(4004)."""
    from meetbot.web.ws_chat import ws_chat

    ws = _make_websocket()
    mock_db = MagicMock()
    mock_db.__enter__ = MagicMock(return_value=mock_db)
    mock_db.__exit__ = MagicMock(return_value=False)
    mock_session_factory = MagicMock(return_value=mock_db)

    with (
        patch(_GET_SESSION, return_value=mock_session_factory),
        patch(_GET_JOB, return_value=None),
    ):
        await ws_chat(ws, "nonexistent-job-id")

    # Should have sent an error
    sent_texts = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
    assert any(m.get("type") == "error" for m in sent_texts)
    ws.close.assert_awaited_once_with(code=4004)


@pytest.mark.asyncio
async def test_job_not_completed_sends_error_and_closes():
    """If job is still INDEXING, send error + close(4009)."""
    from meetbot.web.ws_chat import ws_chat

    ws = _make_websocket()
    mock_db = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_db)
    job = _make_job(status=JobStatus.INDEXING, db_dir=None)

    with (
        patch(_GET_SESSION, return_value=mock_session_factory),
        patch(_GET_JOB, return_value=job),
    ):
        await ws_chat(ws, "job-abc-123")

    sent_texts = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
    assert any(m.get("type") == "error" for m in sent_texts)
    ws.close.assert_awaited_once_with(code=4009)


@pytest.mark.asyncio
async def test_missing_question_sends_error():
    """Empty question payload → error + close(4010)."""
    from meetbot.web.ws_chat import ws_chat

    ws = _make_websocket(json.dumps({"question": ""}))
    mock_db = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_db)
    job = _make_job()
    session = _make_chat_session()

    with (
        patch(_GET_SESSION, return_value=mock_session_factory),
        patch(_GET_JOB, return_value=job),
        patch(_GET_OR_CREATE, return_value=session),
    ):
        await ws_chat(ws, "job-abc-123")

    sent_texts = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
    assert any(m.get("type") == "error" for m in sent_texts)
    ws.close.assert_awaited_with(code=4010)


@pytest.mark.asyncio
async def test_valid_query_saves_assistant_message():
    """Successful query saves user + assistant messages to DB."""
    from meetbot.web.ws_chat import ws_chat

    ws = _make_websocket(
        json.dumps({"question": "What was discussed?", "llm_mode": "hf"})
    )
    mock_db = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_db)
    job = _make_job()
    session = _make_chat_session("sess-xyz")

    mock_query_svc_instance = MagicMock()
    mock_query_svc_instance.query_stream.return_value = _simple_stream("Test answer")

    with (
        patch(_GET_SESSION, return_value=mock_session_factory),
        patch(_GET_JOB, return_value=job),
        patch(_GET_OR_CREATE, return_value=session),
        patch(_CREATE_MSG) as mock_create_msg,
        patch(_QUERY_SVC, return_value=mock_query_svc_instance),
    ):
        await ws_chat(ws, "job-abc-123")

    # Should have called create_chat_message twice: user + assistant
    assert mock_create_msg.call_count >= 2
    roles = [c.kwargs.get("role") or c.args[2] for c in mock_create_msg.call_args_list]
    assert "user" in roles
    assert "assistant" in roles

    # The assistant message should have status="completed"
    assistant_calls = [
        c for c in mock_create_msg.call_args_list
        if (c.kwargs.get("role") or (c.args[2] if len(c.args) > 2 else "")) == "assistant"
    ]
    assert assistant_calls
    last_asst = assistant_calls[-1]
    status = last_asst.kwargs.get("status")
    assert status == "completed"


@pytest.mark.asyncio
async def test_retrieval_method_pageindex_forwarded():
    """retrieval_method='pageindex' must be passed to query_stream."""
    from meetbot.web.ws_chat import ws_chat

    ws = _make_websocket(
        json.dumps({
            "question": "Summarize",
            "llm_mode": "hf",
            "retrieval_method": "pageindex",
        })
    )
    mock_db = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_db)
    job = _make_job()
    session = _make_chat_session()

    mock_query_svc_instance = MagicMock()
    mock_query_svc_instance.query_stream.return_value = _simple_stream("Summary")

    with (
        patch(_GET_SESSION, return_value=mock_session_factory),
        patch(_GET_JOB, return_value=job),
        patch(_GET_OR_CREATE, return_value=session),
        patch(_CREATE_MSG),
        patch(_QUERY_SVC, return_value=mock_query_svc_instance),
    ):
        await ws_chat(ws, "job-abc-123")

    mock_query_svc_instance.query_stream.assert_called_once()
    call_kwargs = mock_query_svc_instance.query_stream.call_args.kwargs
    assert call_kwargs.get("retrieval_method") == "pageindex"


@pytest.mark.asyncio
async def test_invalid_json_payload_sends_error():
    """Malformed JSON in receive_text → error + close(4010)."""
    from meetbot.web.ws_chat import ws_chat

    ws = _make_websocket("not-valid-json{{{")
    mock_db = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_db)
    job = _make_job()
    session = _make_chat_session()

    with (
        patch(_GET_SESSION, return_value=mock_session_factory),
        patch(_GET_JOB, return_value=job),
        patch(_GET_OR_CREATE, return_value=session),
    ):
        await ws_chat(ws, "job-abc-123")

    sent_texts = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
    assert any(m.get("type") == "error" for m in sent_texts)
    ws.close.assert_awaited_with(code=4010)
