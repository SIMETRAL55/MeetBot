"""
Tests for authentication module — password hashing and verification.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch

import pytest

from meetbot.web.auth import generate_otp, hash_password, send_otp_email, verify_otp, verify_password


class TestPasswordHashing:
    """Test bcrypt password hashing and verification."""

    def test_hash_password_returns_string(self):
        hashed = hash_password("mypassword")
        assert isinstance(hashed, str)
        assert len(hashed) > 20

    def test_hash_differs_from_plain(self):
        hashed = hash_password("mypassword")
        assert hashed != "mypassword"

    def test_verify_correct_password(self):
        hashed = hash_password("correct-password")
        assert verify_password("correct-password", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_verify_empty_password(self):
        hashed = hash_password("something")
        assert verify_password("", hashed) is False

    def test_hash_is_unique_each_time(self):
        hash1 = hash_password("same-password")
        hash2 = hash_password("same-password")
        # bcrypt generates unique salts, so hashes should differ
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password("same-password", hash1) is True
        assert verify_password("same-password", hash2) is True

    def test_unicode_password(self):
        hashed = hash_password("パスワード123")
        assert verify_password("パスワード123", hashed) is True
        assert verify_password("パスワード124", hashed) is False

    def test_long_password(self):
        # bcrypt has a 72-byte limit but should handle long strings gracefully
        long_pw = "a" * 72
        hashed = hash_password(long_pw)
        assert verify_password(long_pw, hashed) is True

    def test_verify_invalid_hash(self):
        # Should return False, not crash
        assert verify_password("test", "not-a-valid-hash") is False


@dataclass
class _MockUser:
    otp_hash: Optional[str]
    otp_expires: Optional[datetime]
    otp_purpose: Optional[str]


class TestOtpHelpers:
    """Test OTP generation and verification helpers."""

    def test_generate_otp_is_six_chars(self):
        raw, _ = generate_otp()
        assert len(raw) == 6
        assert raw.isdigit()

    def test_generate_otp_is_zero_padded(self):
        for _ in range(200):
            raw, _ = generate_otp()
            assert len(raw) == 6, f"Expected 6 chars, got {len(raw)}: {raw!r}"

    def test_generate_otp_hash_is_sha256_hex(self):
        _, hashed = generate_otp()
        assert len(hashed) == 64
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_verify_otp_correct_register(self):
        raw, hashed = generate_otp()
        user = _MockUser(
            otp_hash=hashed,
            otp_expires=datetime.now(timezone.utc) + timedelta(minutes=10),
            otp_purpose="register",
        )
        assert verify_otp(user, raw, "register") is True

    def test_verify_otp_correct_reset(self):
        raw, hashed = generate_otp()
        user = _MockUser(
            otp_hash=hashed,
            otp_expires=datetime.now(timezone.utc) + timedelta(minutes=10),
            otp_purpose="reset",
        )
        assert verify_otp(user, raw, "reset") is True

    def test_verify_otp_wrong_code(self):
        _, hashed = generate_otp()
        user = _MockUser(
            otp_hash=hashed,
            otp_expires=datetime.now(timezone.utc) + timedelta(minutes=10),
            otp_purpose="register",
        )
        assert verify_otp(user, "000000", "register") is False

    def test_verify_otp_expired(self):
        raw, hashed = generate_otp()
        user = _MockUser(
            otp_hash=hashed,
            otp_expires=datetime.now(timezone.utc) - timedelta(seconds=1),
            otp_purpose="register",
        )
        assert verify_otp(user, raw, "register") is False

    def test_verify_otp_wrong_purpose(self):
        raw, hashed = generate_otp()
        user = _MockUser(
            otp_hash=hashed,
            otp_expires=datetime.now(timezone.utc) + timedelta(minutes=10),
            otp_purpose="register",
        )
        assert verify_otp(user, raw, "reset") is False

    def test_verify_otp_no_hash(self):
        user = _MockUser(
            otp_hash=None,
            otp_expires=datetime.now(timezone.utc) + timedelta(minutes=10),
            otp_purpose="register",
        )
        assert verify_otp(user, "123456", "register") is False

    def test_send_otp_raises_when_smtp_not_configured(self):
        from meetbot.config import settings

        with patch.object(settings, "SMTP_HOST", ""), patch.object(settings, "SMTP_USER", ""):
            with pytest.raises(ValueError, match="SMTP not configured"):
                send_otp_email("x@example.com", "123456", "register")
