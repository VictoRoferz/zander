"""
Authentication helpers for the dashboard.

- bcrypt verification against the SQLite users table
- Random opaque session IDs in the SQLite sessions table
- Cookie-based browser sessions
- A FastAPI dependency `current_user_email` that reads the cookie
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Cookie, HTTPException, status
from passlib.hash import bcrypt

import db

SESSION_COOKIE = "dashboard_session"
SESSION_HOURS = int(os.environ.get("SESSION_HOURS", "8"))

# Pre-computed at import time so verify_password() can spend a comparable
# amount of CPU when the email is unknown — keeps "no such user" from being
# observably faster than "wrong password" via timing.
_DUMMY_HASH = bcrypt.hash("dummy-password-for-timing-equalization")


def verify_password(email: str, password: str) -> bool:
    """Look up user by email and bcrypt-verify the password."""
    row = db.get_user(email)
    if row is None:
        try:
            bcrypt.verify(password, _DUMMY_HASH)
        except ValueError:
            pass
        return False
    try:
        return bcrypt.verify(password, row["password_hash"])
    except ValueError:
        return False


def login(email: str) -> tuple[str, datetime]:
    """Create a session for `email`, return (session_id, expires_at)."""
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    db.create_session(session_id, email, expires_at)
    return session_id, expires_at


def logout(session_id: str) -> None:
    db.delete_session(session_id)


def session_to_email(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    return db.get_session_email(session_id)


# ---- FastAPI dependencies -------------------------------------------------

def current_user_email(
    dashboard_session: Optional[str] = Cookie(default=None),
) -> str:
    """Require an authenticated session; 401 otherwise (HTML routes redirect)."""
    email = session_to_email(dashboard_session)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return email


def maybe_current_user_email(
    dashboard_session: Optional[str] = Cookie(default=None),
) -> Optional[str]:
    """Return email if logged in, else None (no exception)."""
    return session_to_email(dashboard_session)
