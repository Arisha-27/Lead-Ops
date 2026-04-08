"""
Lead-Ops · Configuration
========================
Centralised configuration loader.

Reads environment variables from a ``.env`` file (via python-dotenv)
and exposes them as a validated ``Settings`` dataclass.

**Key design decision:** Settings are *lazy-loaded* — missing env vars
won't crash the process on import.  They raise ``RuntimeError`` only
when the value is actually accessed.  This lets the FastAPI server
boot and report a 503 instead of crashing in an infinite restart loop
on Hugging Face Spaces.

Usage::

    from config import get_settings

    settings = get_settings()
    print(settings.TAVILY_API_KEY)   # raises if not set
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require(name: str) -> str:
    """Return an env-var value or raise RuntimeError (NOT sys.exit)."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"[Lead-Ops Config] Required environment variable '{name}' "
            f"is not set. Create a .env file in the project root or "
            f"export it in your shell.  See .env.example."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    """Return an env-var value or a default."""
    return os.getenv(name, default)


# ── Settings ──────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Application-wide settings.

    All fields use ``_optional`` during construction so the class never
    crashes on import.  Use the ``require_*`` properties to access
    values that are mandatory — they raise ``RuntimeError`` at
    access-time if unset.
    """

    # Store raw values (may be empty strings)
    _hf_token: str = field(default_factory=lambda: _optional("HF_TOKEN"))
    _api_base_url: str = field(default_factory=lambda: _optional("API_BASE_URL"))
    _tavily_api_key: str = field(default_factory=lambda: _optional("TAVILY_API_KEY"))
    _model_name: str = field(default_factory=lambda: _optional("MODEL_NAME", ""))
    _openai_api_key: str = field(default_factory=lambda: _optional("OPENAI_API_KEY", ""))

    # Optional — always safe to read
    DATABASE_URL: str = field(
        default_factory=lambda: _optional("DATABASE_URL", "sqlite:///lead_ops.db"),
    )
    LOG_LEVEL: str = field(
        default_factory=lambda: _optional("LOG_LEVEL", "INFO"),
    )
    PROJECT_ROOT: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent,
    )

    # ── Safe accessors (raise only when called) ──────────────────────────

    @property
    def HF_TOKEN(self) -> str:
        if not self._hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. Export it or add to .env."
            )
        return self._hf_token

    @property
    def API_BASE_URL(self) -> str:
        if not self._api_base_url:
            raise RuntimeError(
                "API_BASE_URL is not set. Export it or add to .env."
            )
        return self._api_base_url

    @property
    def TAVILY_API_KEY(self) -> str:
        if not self._tavily_api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Export it or add to .env."
            )
        return self._tavily_api_key

    @property
    def MODEL_NAME(self) -> str:
        if not self._model_name:
            raise RuntimeError(
                "MODEL_NAME is not set. Export it or add to .env."
            )
        return self._model_name

    @property
    def OPENAI_API_KEY(self) -> str:
        if not self._openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or add to .env."
            )
        return self._openai_api_key

    # ── Convenience checks ───────────────────────────────────────────────

    @property
    def has_tavily(self) -> bool:
        return bool(self._tavily_api_key) and "xxxx" not in self._tavily_api_key

    @property
    def has_model(self) -> bool:
        return bool(self._model_name) and bool(self._api_base_url)

    @property
    def is_configured(self) -> bool:
        """True if ALL required vars are set."""
        return all([
            self._hf_token,
            self._api_base_url,
            self._tavily_api_key,
        ])


# ── Singleton ─────────────────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy singleton — safe to call from anywhere."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Backward compat: module-level access (won't crash on import)
settings = get_settings()
