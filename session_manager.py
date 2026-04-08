"""
Lead-Ops · Session Manager
============================
Manages isolated per-session database clones for the RL environment.

On every ``reset()``, the master.db is cloned to ``/tmp/session_{uuid}.db``
so each agent episode operates on a fresh, isolated copy. Sessions are
automatically cleaned up after expiration.

Usage::

    from session_manager import SessionManager

    sm = SessionManager()
    session_id = sm.create_session(task_id="enrich_lead", target_lead_id=1)
    db_session = sm.get_db_session(session_id)
    # ... agent interacts with the database ...
    sm.destroy_session(session_id)
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session as DBSession, sessionmaker

from db_models import create_db_engine


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MASTER_DB = Path(__file__).resolve().parent / "master.db"
SESSION_DIR = Path("/tmp/lead_ops_sessions")
MAX_CONCURRENT_SESSIONS = 10
SESSION_TTL_SECONDS = 3600  # 1 hour


# ── Session Info ──────────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    """Metadata for an active session — fully typed, no monkey-patching."""

    session_id: str
    db_path: Path
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    step_count: int = 0
    is_active: bool = True

    # ── Episode-specific state ────────────────────────────────────────────
    target_lead_id: Optional[int] = None
    task_id: Optional[str] = None
    step_rewards_accum: float = 0.0

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        return self.age_seconds > SESSION_TTL_SECONDS

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "db_path": str(self.db_path),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "age_seconds": self.age_seconds,
            "step_count": self.step_count,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "target_lead_id": self.target_lead_id,
            "task_id": self.task_id,
            "step_rewards_accum": self.step_rewards_accum,
        }


# ── Session Manager ──────────────────────────────────────────────────────────

class SessionManager:
    """
    Manages per-episode database sessions.

    Each ``create_session()`` clones the master database to an isolated
    SQLite file in /tmp. This ensures:
      - Episodes don't interfere with each other
      - The master data is never corrupted
      - Sessions are cheap to create (~1ms for small DBs)
    """

    def __init__(
        self,
        master_db_path: Path | str = DEFAULT_MASTER_DB,
        session_dir: Path | str = SESSION_DIR,
        max_sessions: int = MAX_CONCURRENT_SESSIONS,
        ttl_seconds: int = SESSION_TTL_SECONDS,
    ) -> None:
        self.master_db_path = Path(master_db_path)
        self.session_dir = Path(session_dir)
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds

        # In-memory registry of active sessions
        self._sessions: dict[str, SessionInfo] = {}
        self._engines: dict[str, object] = {}

        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _validate_master_exists(self) -> None:
        """Ensure master.db exists."""
        if not self.master_db_path.exists():
            raise FileNotFoundError(
                f"Master database not found at {self.master_db_path}. "
                f"Run 'python database_init.py' first."
            )

    def _cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of cleaned sessions."""
        expired = [
            sid for sid, info in self._sessions.items()
            if info.is_expired
        ]
        for sid in expired:
            self.destroy_session(sid)
        return len(expired)

    def create_session(
        self,
        task_id: str | None = None,
        target_lead_id: int | None = None,
    ) -> str:
        """
        Clone master.db to a new session database.

        Args:
            task_id: The OpenEnv task identifier for this episode.
            target_lead_id: The lead ID the agent should work on.

        Returns:
            session_id: Unique session identifier.

        Raises:
            FileNotFoundError: If master.db doesn't exist.
            RuntimeError: If max concurrent sessions exceeded.
        """
        self._validate_master_exists()

        # Clean expired sessions first
        self._cleanup_expired()

        # Check capacity
        active = sum(1 for s in self._sessions.values() if s.is_active)
        if active >= self.max_sessions:
            raise RuntimeError(
                f"Maximum concurrent sessions ({self.max_sessions}) reached. "
                f"Destroy an existing session or wait for expiration."
            )

        # Generate session ID and path
        session_id = str(uuid.uuid4())
        db_path = self.session_dir / f"session_{session_id}.db"

        # Clone master.db (fast file copy)
        start = time.monotonic()
        shutil.copy2(str(self.master_db_path), str(db_path))
        elapsed_ms = (time.monotonic() - start) * 1000

        # Also copy WAL and SHM files if they exist
        for suffix in ["-wal", "-shm"]:
            wal_path = Path(str(self.master_db_path) + suffix)
            if wal_path.exists():
                shutil.copy2(str(wal_path), str(db_path) + suffix)

        # Create engine for this session
        engine = create_db_engine(f"sqlite:///{db_path}")
        self._engines[session_id] = engine

        # Register session with proper typed fields
        info = SessionInfo(
            session_id=session_id,
            db_path=db_path,
            task_id=task_id,
            target_lead_id=target_lead_id,
            step_rewards_accum=0.0,
        )
        self._sessions[session_id] = info

        return session_id

    def get_db_session(self, session_id: str) -> DBSession:
        """
        Get a SQLAlchemy Session for the given session ID.

        Raises:
            KeyError: If session_id is not found
            RuntimeError: If session has expired
        """
        info = self._sessions.get(session_id)
        if info is None:
            raise KeyError(f"Session '{session_id}' not found.")

        if info.is_expired:
            self.destroy_session(session_id)
            raise RuntimeError(
                f"Session '{session_id}' has expired "
                f"(age: {info.age_seconds:.0f}s, TTL: {self.ttl_seconds}s)."
            )

        info.touch()
        engine = self._engines.get(session_id)
        if engine is None:
            engine = create_db_engine(f"sqlite:///{info.db_path}")
            self._engines[session_id] = engine

        return DBSession(engine)

    def get_session_info(self, session_id: str) -> SessionInfo:
        """Get metadata for a session."""
        info = self._sessions.get(session_id)
        if info is None:
            raise KeyError(f"Session '{session_id}' not found.")
        return info

    def increment_step(self, session_id: str) -> int:
        """Increment and return the step count for a session."""
        info = self._sessions.get(session_id)
        if info is None:
            raise KeyError(f"Session '{session_id}' not found.")
        info.step_count += 1
        info.touch()
        return info.step_count

    def destroy_session(self, session_id: str) -> None:
        """
        Destroy a session and clean up its database file.

        Safe to call multiple times — no-op if already destroyed.
        """
        info = self._sessions.pop(session_id, None)

        # Dispose engine
        engine = self._engines.pop(session_id, None)
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass

        # Delete database files
        if info is not None and info.db_path.exists():
            try:
                info.db_path.unlink()
                # Also clean up WAL/SHM files
                for suffix in ["-wal", "-shm"]:
                    wal = Path(str(info.db_path) + suffix)
                    if wal.exists():
                        wal.unlink()
            except OSError:
                pass

    def reset(self, session_id: str | None = None) -> str:
        """
        Reset an environment episode.

        If session_id is given, destroys that session and creates a new one.
        If session_id is None, creates a brand new session.

        Returns:
            new_session_id: The ID of the fresh session.
        """
        if session_id is not None:
            self.destroy_session(session_id)
        return self.create_session()

    def list_sessions(self) -> list[dict]:
        """List all active sessions with metadata."""
        return [
            info.to_dict()
            for info in self._sessions.values()
            if info.is_active
        ]

    def destroy_all(self) -> int:
        """Destroy all sessions. Returns count destroyed."""
        ids = list(self._sessions.keys())
        for sid in ids:
            self.destroy_session(sid)
        return len(ids)

    @property
    def active_count(self) -> int:
        """Number of active (non-expired) sessions."""
        return sum(
            1 for s in self._sessions.values()
            if s.is_active and not s.is_expired
        )

    def __repr__(self) -> str:
        return (
            f"<SessionManager("
            f"master='{self.master_db_path.name}', "
            f"active={self.active_count}/{self.max_sessions}"
            f")>"
        )
