"""Runtime configuration for vision-mcp, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:27b"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    """Resolved settings for the MCP server and Ollama client."""

    ollama_host: str
    vision_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_host=_env("OLLAMA_HOST", DEFAULT_HOST),
            vision_model=_env("OLLAMA_VISION_MODEL", DEFAULT_MODEL),
        )


def get_settings() -> Settings:
    """Read settings from the environment on each call (cheap, env-driven)."""
    return Settings.from_env()
