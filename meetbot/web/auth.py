"""
Authentication module for MeetBot web application.

Provides:
- Password hashing and verification (bcrypt)
- Session-based authentication middleware for NiceGUI
- Login/logout helpers
"""

import logging
from typing import Optional

import bcrypt
from starlette.middleware.sessions import SessionMiddleware

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
