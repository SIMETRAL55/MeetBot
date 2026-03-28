"""
Tests for upload and PageIndex edge cases in meetbot/web/api.py.

Covers:
- Upload of a non-audio MIME type → 415 Unsupported Media Type
- Upload exceeding MAX_UPLOAD_SIZE_MB via Content-Length header → 413
- Upload exceeding MAX_UPLOAD_SIZE_MB post-write (no Content-Length) → 413
- Upload with disallowed file extension → 400
- PageIndex build when PAGEINDEX_ENABLED=False → 409
- PageIndex build on a non-COMPLETED job → 409
- PageIndex build on a job not found → 404
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    content_length: Optional[int] = None,
    remote_addr: str = "127.0.0.1",
) -> MagicMock:
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = remote_addr
    headers: dict = {}
    if content_length is not None:
        headers["content-length"] = str(content_length)
    req.headers = headers
    return req


def _make_upload_file(
    filename: str = "meeting.mp3",
    content: bytes = b"fake audio data",
) -> MagicMock:
    uf = MagicMock()
    uf.filename = filename
    uf.file = io.BytesIO(content)
    return uf


def _make_settings_mock(
    max_upload_mb: int = 500,
    extensions: Optional[list] = None,
    output_dir: str = "/tmp/meetbot_test/output",
    rate_limit_upload: str = "10/hour",
    pageindex_enabled: bool = True,
) -> MagicMock:
    """Build a settings mock with sensible defaults."""
    m = MagicMock()
    m.MAX_UPLOAD_SIZE_MB = max_upload_mb
    m.get_allowed_extensions.return_value = extensions or [".mp3", ".wav", ".m4a", ".mp4"]
    m.OUTPUT_DIR = output_dir
    m.RATE_LIMIT_UPLOAD = rate_limit_upload
    m.PAGEINDEX_ENABLED = pageindex_enabled
    return m


# ---------------------------------------------------------------------------
# Upload endpoint — edge cases
# ---------------------------------------------------------------------------

class TestApiUploadEdgeCases:

    @pytest.mark.asyncio
    async def test_disallowed_extension_returns_400(self, tmp_path):
        """Files with disallowed extensions are rejected with HTTP 400."""
        from meetbot.web.api import api_upload_job

        # api_upload_job is wrapped by @limiter.limit; use __wrapped__ to bypass slowapi
        inner = api_upload_job.__wrapped__

        request = _make_request()
        upload = _make_upload_file(filename="script.exe")
        mock_cfg = _make_settings_mock(output_dir=str(tmp_path / "output"))

        # Patch the settings object imported inside api_upload_job's function body
        with patch("meetbot.config.settings", mock_cfg):
            with pytest.raises(HTTPException) as exc_info:
                await inner(request=request, file=upload, min_speakers=None, max_speakers=None)

        assert exc_info.value.status_code == 400
        assert ".exe" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_oversized_content_length_returns_413(self, tmp_path):
        """
        When Content-Length header exceeds MAX_UPLOAD_SIZE_MB the request
        is rejected before writing any data to disk.
        """
        from meetbot.web.api import api_upload_job
        inner = api_upload_job.__wrapped__

        max_mb = 1
        max_bytes = max_mb * 1024 * 1024
        request = _make_request(content_length=max_bytes + 1)
        upload = _make_upload_file(filename="big.mp3")
        mock_cfg = _make_settings_mock(max_upload_mb=max_mb, output_dir=str(tmp_path / "output"))

        with patch("meetbot.config.settings", mock_cfg):
            with pytest.raises(HTTPException) as exc_info:
                await inner(request=request, file=upload, min_speakers=None, max_speakers=None)

        assert exc_info.value.status_code == 413
        assert str(max_mb) in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_oversized_post_write_returns_413(self, tmp_path):
        """
        When file is larger than the limit but no Content-Length header is sent,
        the check after writing to disk returns 413.
        """
        from meetbot.web.api import api_upload_job
        inner = api_upload_job.__wrapped__

        max_mb = 1
        max_bytes = max_mb * 1024 * 1024
        request = _make_request()  # no content-length
        upload = _make_upload_file(filename="big.mp3", content=b"x" * 10)
        mock_cfg = _make_settings_mock(max_upload_mb=max_mb, output_dir=str(tmp_path / "output"))

        mock_db = MagicMock()
        mock_db.query.return_value.first.return_value = MagicMock(id="user-001")
        mock_db.close = MagicMock()

        with patch("meetbot.config.settings", mock_cfg), \
             patch("meetbot.web.api.get_session") as mock_get_session, \
             patch("meetbot.web.auth_middleware.get_optional_user", return_value=None), \
             patch("shutil.copyfileobj"), \
             patch("builtins.open", mock_open()), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.unlink"):

            # Post-write stat reports size exceeding limit
            mock_stat.return_value.st_size = max_bytes + 1
            mock_get_session.return_value = lambda: mock_db

            with pytest.raises(HTTPException) as exc_info:
                await inner(request=request, file=upload, min_speakers=None, max_speakers=None)

        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_non_audio_mime_type_returns_415(self, tmp_path):
        """
        A file with audio extension but non-audio MIME (disguised executable)
        is rejected with HTTP 415.
        """
        from meetbot.web.api import api_upload_job
        inner = api_upload_job.__wrapped__

        request = _make_request()
        upload = _make_upload_file(filename="evil.mp3", content=b"<html></html>")
        mock_cfg = _make_settings_mock(output_dir=str(tmp_path / "output"))

        mock_db = MagicMock()
        mock_db.query.return_value.first.return_value = MagicMock(id="user-001")
        mock_db.close = MagicMock()

        with patch("meetbot.config.settings", mock_cfg), \
             patch("meetbot.web.api.get_session") as mock_get_session, \
             patch("meetbot.web.auth_middleware.get_optional_user", return_value=None), \
             patch("shutil.copyfileobj"), \
             patch("magic.from_file", return_value="text/html"), \
             patch("builtins.open", mock_open()), \
             patch("pathlib.Path.stat") as mock_stat, \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.unlink"):

            mock_stat.return_value.st_size = len(b"<html></html>")
            mock_get_session.return_value = lambda: mock_db

            with pytest.raises(HTTPException) as exc_info:
                await inner(request=request, file=upload, min_speakers=None, max_speakers=None)

        assert exc_info.value.status_code == 415
        assert "text/html" in exc_info.value.detail


# ---------------------------------------------------------------------------
# PageIndex build endpoint — edge cases
# ---------------------------------------------------------------------------

class TestApiBuildPageindexEdgeCases:

    @pytest.mark.asyncio
    async def test_pageindex_disabled_returns_409(self):
        """POST /api/jobs/{job_id}/build-pageindex returns 409 when PAGEINDEX_ENABLED=False."""
        from meetbot.web.api import api_build_pageindex

        # _settings is the module-level alias for meetbot.config.settings
        with patch("meetbot.config.settings") as mock_cfg:
            mock_cfg.PAGEINDEX_ENABLED = False

            with pytest.raises(HTTPException) as exc_info:
                await api_build_pageindex(job_id="any-job-id")

        assert exc_info.value.status_code == 409
        assert "PAGEINDEX_ENABLED" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_pageindex_job_not_found_returns_404(self):
        """POST build-pageindex returns 404 when the job does not exist."""
        from meetbot.web.api import api_build_pageindex

        mock_db = MagicMock()
        mock_db.close = MagicMock()

        with patch("meetbot.config.settings") as mock_cfg, \
             patch("meetbot.web.api.get_session") as mock_get_session, \
             patch("meetbot.web.api.get_job", return_value=None):

            mock_cfg.PAGEINDEX_ENABLED = True
            mock_get_session.return_value = lambda: mock_db

            with pytest.raises(HTTPException) as exc_info:
                await api_build_pageindex(job_id="nonexistent-job")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_pageindex_job_not_completed_returns_409(self):
        """POST build-pageindex returns 409 when job is not COMPLETED."""
        from meetbot.web.api import api_build_pageindex
        from meetbot.db.models import JobStatus

        mock_job = MagicMock()
        mock_job.status = JobStatus.PENDING
        mock_job.pageindex_status = None

        mock_db = MagicMock()
        mock_db.close = MagicMock()

        with patch("meetbot.config.settings") as mock_cfg, \
             patch("meetbot.web.api.get_session") as mock_get_session, \
             patch("meetbot.web.api.get_job", return_value=mock_job):

            mock_cfg.PAGEINDEX_ENABLED = True
            mock_get_session.return_value = lambda: mock_db

            with pytest.raises(HTTPException) as exc_info:
                await api_build_pageindex(job_id="pending-job")

        assert exc_info.value.status_code == 409
        assert "COMPLETED" in exc_info.value.detail
