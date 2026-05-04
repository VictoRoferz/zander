"""
Tiny SQLite layer for the dashboard.

Two tables:
  users    — email PK, bcrypt password hash, optional full name
  sessions — session_id PK, user email, expiry timestamp

The DB file lives next to mirrored captures so it's discoverable but
ignored by git (zander-data/ is in .gitignore).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


def _data_root() -> Path:
    override = os.environ.get("DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "zander-data"


DB_PATH = _data_root() / "dashboard.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    full_name      TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    email        TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_email   ON sessions(email);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
"""


def initialize() -> None:
    """Create the DB file and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """A short-lived connection. SQLite handles locking; we keep it simple."""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        # Enforce foreign keys (off by default in SQLite).
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


# ---- Users ---------------------------------------------------------------

def add_user(email: str, password_hash: str, full_name: Optional[str]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (email, password_hash, full_name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (email, password_hash, full_name, _now_iso()),
        )


def get_user(email: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT email, password_hash, full_name, created_at "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()


def list_users() -> list[sqlite3.Row]:
    with _connect() as conn:
        return list(
            conn.execute(
                "SELECT email, full_name, created_at FROM users ORDER BY email"
            )
        )


def delete_user(email: str) -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE email = ?", (email,))
        return cur.rowcount


def update_password(email: str, password_hash: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (password_hash, email),
        )
        return cur.rowcount


# ---- Sessions ------------------------------------------------------------

def create_session(session_id: str, email: str, expires_at: datetime) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, email, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, email, _now_iso(), expires_at.isoformat()),
        )


def get_session_email(session_id: str) -> Optional[str]:
    """Return the email if session exists and is not expired; else None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT email, expires_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    expires = datetime.fromisoformat(row["expires_at"])
    if expires < datetime.now(timezone.utc):
        delete_session(session_id)
        return None
    return row["email"]


def delete_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def latest_active_email() -> Optional[str]:
    """
    Email of whoever owns the most recently created non-expired session.
    Used by camerapi (Phase C) to attribute GPIO-button captures.
    Returns None if no active session exists.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT email FROM sessions "
            "WHERE expires_at > ? "
            "ORDER BY created_at DESC LIMIT 1",
            (_now_iso(),),
        ).fetchone()
    return row["email"] if row else None


def purge_expired_sessions() -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_now_iso(),))
        return cur.rowcount


# ---- Helpers -------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
