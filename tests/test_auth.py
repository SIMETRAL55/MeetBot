"""
Tests for authentication module — password hashing and verification.
"""

import pytest

from meetbot.web.auth import hash_password, verify_password


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
