import sqlite3
import json
import os
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / ".cache"
DB_PATH = DB_DIR / "sessions.db"

# Keys that must never be written to disk. The user's model API key is meant to
# live only in their browser and in-memory per request.
_SECRET_KEYS = ("model_api_key",)

def init_db():
    """Ensure cache folder exists and initialize SQLite sessions table."""
    os.makedirs(DB_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _scrub_persisted_secrets(conn)
        conn.commit()
    try:
        from utils.vector_db import init_vector_db
        init_vector_db()
    except ImportError:
        pass

def _scrub_persisted_secrets(conn: sqlite3.Connection) -> None:
    """One-time migration: strip secrets from sessions saved by older versions."""
    rows = conn.execute("SELECT session_id, session_data FROM sessions").fetchall()
    for session_id, data in rows:
        try:
            state = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(state, dict) and any(k in state for k in _SECRET_KEYS):
            for k in _SECRET_KEYS:
                state.pop(k, None)
            conn.execute(
                "UPDATE sessions SET session_data = ? WHERE session_id = ?",
                (json.dumps(state), session_id),
            )

def save_session(session_id: str, state: dict) -> None:
    """Serialize the session state to JSON and write to SQLite. Secrets are
    stripped — the user's API key must never be persisted server-side."""
    serialized = json.dumps({k: v for k, v in state.items() if k not in _SECRET_KEYS})
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, session_data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (session_id, serialized)
        )
        conn.commit()

def get_session(session_id: str) -> dict | None:
    """Retrieve and deserialize a session state by ID from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT session_data FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
    return None


def get_sessions_count() -> int:
    """Get the total count of active sessions stored in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM sessions")
        row = cursor.fetchone()
        return row[0] if row else 0
