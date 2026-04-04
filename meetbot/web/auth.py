"""
Authentication module for MeetBot web application.

Provides:
- Password hashing and verification (bcrypt)
- Session-based authentication middleware for NiceGUI
- Login/logout helpers
- Firebase token verification (lazy imports)
- Account lockout, password reset, SMTP email
"""

import hashlib
import hmac as _hmac
import hmac
import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Optional

import bcrypt
from starlette.middleware.sessions import SessionMiddleware

from ..config import settings
from ..db.database import get_session
from ..db.crud import get_user_by_username, update_user_last_login
from ..db.models import User

logger = logging.getLogger(__name__)


def hash_password(plain_password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        plain_password: The plain-text password.

    Returns:
        Bcrypt-hashed password string.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its bcrypt hash.

    Args:
        plain_password: The plain-text password to verify.
        hashed_password: The stored bcrypt hash.

    Returns:
        True if the password matches.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password.

    Args:
        username: The username to authenticate.
        password: The plain-text password.

    Returns:
        User instance if authentication succeeds, None otherwise.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        user = get_user_by_username(db, username)
        if user is None:
            logger.warning(f"Login failed: user '{username}' not found")
            return None

        if not verify_password(password, user.password_hash):
            logger.warning(f"Login failed: wrong password for '{username}'")
            return None

        update_user_last_login(db, user)
        # After commit(), SQLAlchemy expires all attributes on the ORM object.
        # db.refresh() reloads them while the session is still open.
        # db.expunge() then detaches the object cleanly — its __dict__ retains
        # the loaded primitives so they can be safely accessed after close().
        db.refresh(user)
        db.expunge(user)
        logger.info(f"User '{username}' authenticated successfully")
        return user
    finally:
        db.close()


def get_current_user_id() -> Optional[str]:
    """
    Get the current authenticated user's ID from the NiceGUI session.

    Returns:
        User ID string or None if not authenticated.
    """
    from nicegui import app
    return app.storage.user.get("user_id")


def get_current_username() -> Optional[str]:
    """
    Get the current authenticated user's username from the NiceGUI session.

    Returns:
        Username string or None if not authenticated.
    """
    from nicegui import app
    return app.storage.user.get("username")


def is_authenticated() -> bool:
    """Check if the current user is authenticated."""
    return get_current_user_id() is not None


def login_user(user: User) -> None:
    """
    Store user info in the NiceGUI session after successful authentication.

    Args:
        user: Authenticated User instance.
    """
    from nicegui import app
    app.storage.user["user_id"] = user.id
    app.storage.user["username"] = user.username
    app.storage.user["display_name"] = user.display_name or user.username
    app.storage.user["is_admin"] = user.is_admin
    logger.info(f"Session created for user '{user.username}'")


def logout_user() -> None:
    """Clear the current user's session."""
    from nicegui import app
    username = app.storage.user.get("username", "unknown")
    app.storage.user.clear()
    logger.info(f"Session cleared for user '{username}'")


# =============================================================================
# Firebase token verification (lazy imports — firebase not a hard dependency)
# =============================================================================

_FIREBASE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
_firebase_certs_cache: dict = {}
_firebase_certs_expiry: Optional[datetime] = None


def _fetch_firebase_certs() -> dict:
    """Fetch and cache Google's public certs for Firebase token verification."""
    import requests as _requests

    global _firebase_certs_cache, _firebase_certs_expiry
    now = datetime.now(timezone.utc)
    if _firebase_certs_cache and _firebase_certs_expiry and now < _firebase_certs_expiry:
        return _firebase_certs_cache

    resp = _requests.get(_FIREBASE_CERTS_URL, timeout=10)
    resp.raise_for_status()
    _firebase_certs_cache = resp.json()
    # Cache for 1 hour regardless of Cache-Control header
    _firebase_certs_expiry = now + timedelta(hours=1)
    return _firebase_certs_cache


def verify_firebase_token(id_token: str) -> Optional[dict]:
    """
    Verify a Firebase ID token using PyJWT + Google's RSA public certs.

    Returns the decoded claims dict on success, None on any failure.
    Imports PyJWT and cryptography lazily so they're optional at startup.
    """
    try:
        import jwt as _jwt

        project_id = settings.FIREBASE_PROJECT_ID
        if not project_id:
            logger.warning("FIREBASE_PROJECT_ID not configured")
            return None

        certs = _fetch_firebase_certs()

        # Decode header to find the key id
        unverified_header = _jwt.get_unverified_header(id_token)
        kid = unverified_header.get("kid")
        if not kid or kid not in certs:
            logger.warning("Firebase token kid not found in certs")
            return None

        from cryptography import x509 as _x509
        from cryptography.hazmat.backends import default_backend as _default_backend

        cert = _x509.load_pem_x509_certificate(
            certs[kid].encode("utf-8"), _default_backend()
        )
        public_key = cert.public_key()

        claims = _jwt.decode(
            id_token,
            public_key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
        )
        return claims
    except Exception as exc:
        logger.warning(f"Firebase token verification failed: {exc}")
        return None


def get_or_create_firebase_user(db, claims: dict) -> Optional[User]:
    """
    Look up or create a User from decoded Firebase claims.

    Args:
        db: SQLAlchemy session.
        claims: Decoded Firebase JWT claims.

    Returns:
        User instance (detached), or None on failure.
    """
    from ..db.crud import get_user_by_firebase_uid, create_user_firebase

    uid = claims.get("sub") or claims.get("user_id")
    email = claims.get("email", "")
    if not uid:
        return None

    user = get_user_by_firebase_uid(db, uid)
    if user is None:
        username = claims.get("name") or (email.split("@")[0] if email else f"fb_{uid[:8]}")
        # Ensure username uniqueness with a suffix if needed
        base = username[:60]
        candidate = base
        counter = 1
        while get_user_by_username(db, candidate) is not None:
            candidate = f"{base}_{counter}"
            counter += 1
        # Dummy password hash — Firebase users never use local password auth
        dummy_hash = hash_password(secrets.token_hex(32))
        user = create_user_firebase(
            db,
            username=candidate,
            password_hash=dummy_hash,
            email=email or None,
            firebase_uid=uid,
            email_verified=claims.get("email_verified", False),
        )
    else:
        # Keep email and verification status in sync with Firebase claims
        if email and user.email != email:
            user.email = email
        user.email_verified = claims.get("email_verified", False)
        db.add(user)
        db.commit()

    update_user_last_login(db, user)
    db.refresh(user)
    db.expunge(user)
    return user


# =============================================================================
# Account lockout
# =============================================================================

_LOCKOUT_THRESHOLDS = [
    (5, timedelta(minutes=15)),
    (10, timedelta(hours=1)),
]


def check_account_lockout(user: User) -> Optional[datetime]:
    """
    Return the locked_until datetime if the account is currently locked, else None.
    """
    if user.locked_until is None:
        return None
    now = datetime.now(timezone.utc)
    locked = user.locked_until
    if locked.tzinfo is None:
        locked = locked.replace(tzinfo=timezone.utc)
    if now < locked:
        return locked
    return None


def record_failed_login(db, user: User) -> None:
    """Increment failed_login_attempts and apply lockout if thresholds are met."""
    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
    lock_duration: Optional[timedelta] = None
    for threshold, duration in sorted(_LOCKOUT_THRESHOLDS, reverse=True):
        if user.failed_login_attempts >= threshold:
            lock_duration = duration
            break
    if lock_duration is not None:
        user.locked_until = datetime.now(timezone.utc) + lock_duration
        logger.warning(
            f"Account '{user.username}' locked until {user.locked_until} "
            f"after {user.failed_login_attempts} failed attempts"
        )
    db.add(user)
    db.commit()


def record_successful_login(db, user: User) -> None:
    """Reset lockout counters after a successful login."""
    user.failed_login_attempts = 0
    user.locked_until = None
    update_user_last_login(db, user)
    db.refresh(user)


# =============================================================================
# Password reset
# =============================================================================

def generate_password_reset_token(db, user: User) -> str:
    """
    Generate a secure reset token, store its hash on the user, and return the raw token.

    The raw token is sent to the user; only the hash is stored in the DB.
    """
    raw_token = secrets.token_urlsafe(48)
    # Store SHA-256 hex digest so the raw token is never in the DB
    import hashlib
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    user.password_reset_token = token_hash
    user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
    db.add(user)
    db.commit()
    return raw_token


def generate_otp() -> tuple[str, str]:
    """Return (zero-padded-6-digit string, sha256-hex-hash)."""
    raw = str(secrets.randbelow(1_000_000)).zfill(6)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed


def store_otp(db, user: User, otp_hash: str, purpose: str) -> None:
    """Persist OTP hash + expiry + purpose onto user, commit."""
    user.otp_hash = otp_hash
    user.otp_expires = datetime.now(timezone.utc) + timedelta(minutes=15)
    user.otp_purpose = purpose
    db.add(user)
    db.commit()


def verify_otp(user: User, raw_otp: str, purpose: str) -> bool:
    """Return True iff the raw OTP matches the stored hash and is unexpired."""
    if not user.otp_hash or not user.otp_expires or user.otp_purpose != purpose:
        return False
    expires = user.otp_expires
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return False
    candidate_hash = hashlib.sha256(raw_otp.encode()).hexdigest()
    return _hmac.compare_digest(user.otp_hash, candidate_hash)


def clear_otp(db, user: User) -> None:
    """Wipe OTP fields from user row and commit."""
    user.otp_hash = None
    user.otp_expires = None
    user.otp_purpose = None
    db.add(user)
    db.commit()


def send_otp_email(email: str, otp: str, purpose: str) -> None:
    """Send a 6-digit OTP via SMTP.  Raises ValueError if SMTP not configured."""
    if not settings.SMTP_HOST or not settings.SMTP_USER:
        raise ValueError("SMTP not configured — set SMTP_HOST and SMTP_USER")
    if purpose == "register":
        subject = "Verify your MeetBot account"
        body_prefix = "Your verification code is:"
    else:
        subject = "Reset your MeetBot password"
        body_prefix = "Your password reset code is:"
    msg = MIMEText(f"{body_prefix} {otp}\n\nExpires in 15 minutes.", "plain")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM
    msg["To"] = email
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
        server.ehlo()
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_FROM, [email], msg.as_string())


def send_password_reset_email(email: str, reset_url: str) -> bool:
    """
    Send a password reset email via SMTP (STARTTLS).

    Returns True on success, False on any failure.
    """
    if not settings.SMTP_HOST or not settings.SMTP_USER:
        logger.warning("SMTP not configured — skipping password reset email")
        return False

    msg = MIMEText(
        f"You requested a password reset for your MeetBot account.\n\n"
        f"Click the link below to reset your password (expires in 1 hour):\n\n"
        f"{reset_url}\n\n"
        f"If you did not request this, you can safely ignore this email.",
        "plain",
    )
    msg["Subject"] = "MeetBot — Password Reset"
    msg["From"] = settings.SMTP_FROM
    msg["To"] = email

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM, [email], msg.as_string())
        logger.info(f"Password reset email sent to {email}")
        return True
    except Exception as exc:
        logger.error(f"Failed to send password reset email: {exc}")
        return False
