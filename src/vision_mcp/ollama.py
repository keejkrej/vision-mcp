"""Async client that drives a local Ollama vision model over its REST API.

Ollama's /api/chat endpoint accepts an `images` array of base64-encoded bytes
alongside text content. This module loads images from local paths or remote
HTTP(S) URLs, encodes them, and sends a single non-streaming chat completion.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx

from vision_mcp.config import Settings

# Conservative size cap to keep MCP tool calls responsive. Ollama itself can
# handle larger inputs, but we avoid accidentally shipping multi-MB payloads
# through the stdio transport.
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB
HTTP_TIMEOUT = 120.0


@dataclass
class VisionResult:
    """Outcome of a vision chat completion."""

    content: str
    model: str
    done: bool
    eval_duration_ns: Optional[int]
    total_duration_ns: Optional[int]


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _looks_like_base64(value: str) -> bool:
    """Heuristic: long string with only base64 alphabet chars and no scheme."""
    if len(value) < 64 or _is_url(value):
        return False
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value.strip()))


async def _load_image_bytes(source: str, client: httpx.AsyncClient) -> bytes:
    """Resolve an image source to raw bytes.

    Accepted sources:
      - Local file path (absolute or relative to CWD)
      - http(s) URL
      - Pre-encoded base64 string (decoded back to bytes)
    """
    if _is_url(source):
        response = await client.get(source, follow_redirects=True)
        response.raise_for_status()
        data = response.content
    elif _looks_like_base64(source):
        try:
            data = base64.b64decode(source, validate=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Image looked like base64 but failed to decode: {exc}")
    else:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        data = path.read_bytes()

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(data)} bytes, exceeds {MAX_IMAGE_BYTES} byte limit. "
            "Resize or compress the image before sending."
        )
    return data


async def _load_images(
    sources: Iterable[str], client: httpx.AsyncClient
) -> list[str]:
    """Load and base64-encode all image sources for Ollama's `images` array."""
    encoded: list[str] = []
    for source in sources:
        raw = await _load_image_bytes(source, client)
        encoded.append(base64.b64encode(raw).decode("ascii"))
    if not encoded:
        raise ValueError("At least one image is required.")
    return encoded


async def vision_chat(
    settings: Settings,
    prompt: str,
    images: Iterable[str],
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> VisionResult:
    """Send prompt + images to the Ollama chat endpoint and return the reply.

    Args:
        settings: Resolved server settings (host + default model).
        prompt: User question or instruction about the image(s).
        images: Iterable of paths, URLs, or base64 strings.
        system: Optional system prompt to steer the vision model.
        model: Override the configured vision model for this call.
        temperature: Sampling temperature; low for faithful description.
    """
    target_model = model or settings.vision_model
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": prompt,
            "images": await _load_images(images, httpx.AsyncClient(timeout=HTTP_TIMEOUT)),
        }
    )

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        if response.status_code != 200:
            body = response.text
            raise RuntimeError(
                f"Ollama /api/chat returned {response.status_code}: {body[:500]}"
            )
        data = response.json()

    message = data.get("message") or {}
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response content.")

    return VisionResult(
        content=content,
        model=data.get("model", target_model),
        done=bool(data.get("done", False)),
        eval_duration_ns=data.get("eval_duration"),
        total_duration_ns=data.get("total_duration"),
    )


async def check_model_available(settings: Settings, model: Optional[str] = None) -> bool:
    """Best-effort check that the configured vision model is locally available."""
    target = model or settings.vision_model
    url = f"{settings.ollama_host.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return False
            tags = response.json().get("models", [])
            return any(m.get("name", "").startswith(target) for m in tags)
    except Exception:
        return False
