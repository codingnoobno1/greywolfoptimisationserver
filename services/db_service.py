"""
Greywolf — Tier 2.5 Metadata Layer
SQLite-backed device registry and session tracking.

Tables:
  devices  — one row per unique device_id (UUID from Flutter)
  sessions — one row per WebSocket connect/disconnect lifecycle
  events   — system event log (replaces old CSV logger)
"""
import sqlite3
import uuid
import os
from datetime import datetime, timezone
from threading import Lock
from contextlib import contextmanager
from core.logger import logger

DB_PATH = os.getenv("GREYWOLF_DB_PATH", "greywolf.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteDatabase:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._lock = Lock()
        self._init_schema()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _conn(self):
        """Thread-safe connection context manager."""
        with self._lock:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id    TEXT PRIMARY KEY,
                    first_seen   TEXT NOT NULL,
                    last_seen    TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'offline'
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    device_id       TEXT NOT NULL,
                    connected_at    TEXT NOT NULL,
                    disconnected_at TEXT,
                    FOREIGN KEY(device_id) REFERENCES devices(device_id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    device_id   TEXT,
                    details     TEXT
                );
            """)
        logger.info(f"SQLite DB ready: {self.path}")

    # ------------------------------------------------------------------ #
    #  Device registry                                                      #
    # ------------------------------------------------------------------ #

    def register_device(self, device_id: str) -> None:
        """
        Upsert a device on connect.
        - First time: creates the row, sets first_seen = now
        - Subsequent: updates last_seen + status = 'online'
        """
        now = _now_iso()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO devices (device_id, first_seen, last_seen, status)
                VALUES (?, ?, ?, 'online')
                ON CONFLICT(device_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    status    = 'online'
            """, (device_id, now, now))
        logger.info(f"Device registered/updated: {device_id}")

    def mark_device_offline(self, device_id: str) -> None:
        with self._conn() as conn:
            conn.execute("""
                UPDATE devices SET status = 'offline', last_seen = ?
                WHERE device_id = ?
            """, (_now_iso(), device_id))

    def get_all_devices(self) -> list[dict]:
        """Return all known devices (past + current)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM devices ORDER BY last_seen DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_device(self, device_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    #  Session tracking                                                     #
    # ------------------------------------------------------------------ #

    def open_session(self, device_id: str) -> str:
        """Record a new connection. Returns the new session_id."""
        session_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, device_id, connected_at)
                VALUES (?, ?, ?)
            """, (session_id, device_id, _now_iso()))
        logger.info(f"Session opened: {session_id} for device {device_id}")
        return session_id

    def close_session(self, session_id: str) -> None:
        """Stamp the disconnected_at time for a session."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE sessions SET disconnected_at = ?
                WHERE session_id = ?
            """, (_now_iso(), session_id))
        logger.info(f"Session closed: {session_id}")

    def get_active_sessions(self) -> list[dict]:
        """Sessions that are currently open (no disconnected_at)."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT s.*, d.status FROM sessions s
                JOIN devices d ON s.device_id = d.device_id
                WHERE s.disconnected_at IS NULL
                ORDER BY s.connected_at DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_device_sessions(self, device_id: str) -> list[dict]:
        """Full session history for one device, newest first."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM sessions
                WHERE device_id = ?
                ORDER BY connected_at DESC
            """, (device_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Event log (replaces CSV logger)                                      #
    # ------------------------------------------------------------------ #

    def log_event(self, event_type: str, details: str, device_id: str = None) -> None:
        """Generic event logger (system startup, errors, etc.)."""
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO events (timestamp, event_type, device_id, details)
                    VALUES (?, ?, ?, ?)
                """, (_now_iso(), event_type, device_id, details))
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# Singleton
db = SQLiteDatabase()
