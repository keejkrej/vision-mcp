"""Video frame extraction for the video_analysis tool.

Uses ffmpeg (expected on PATH) to sample `count` evenly-spaced frames from a
local or remote video into a temporary directory, returning their paths so the
vision client can read and base64-encode them. Falls back to downloading a
remote video to a temp file first so ffmpeg can seek into it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def _run_ffmpeg(args: List[str]) -> None:
    """Run ffmpeg, raising with stderr on non-zero exit."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip() or "unknown error"
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {message}")


async def _probe_duration(source: str) -> float:
    """Get duration in seconds via ffprobe (best-effort)."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH; cannot determine duration.")
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed: {message}")
    try:
        return float(stdout.decode().strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {exc}")


async def extract_frame_paths(source: str, *, count: int) -> List[str]:
    """Sample `count` evenly-spaced frames from a video and return their paths.

    Args:
        source: Local path or http(s) URL of the video.
        count: Number of frames to sample (1-8 enforced by the tool schema).

    Returns:
        List of local file paths to PNG frames in chronological order.
    """
    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed. Install it (e.g. `brew install ffmpeg`) to "
            "use video_analysis."
        )

    tmpdir = tempfile.mkdtemp(prefix="vision-mcp-video-")
    remote_tmp: Path | None = None

    try:
        if _is_url(source):
            remote_tmp = Path(tmpdir) / "source_video"
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(source, follow_redirects=True)
                response.raise_for_status()
                remote_tmp.write_bytes(response.content)
            local_source = str(remote_tmp)
        else:
            path = Path(source).expanduser()
            if not path.is_absolute():
                path = Path(os.getcwd()) / path
            if not path.exists():
                raise FileNotFoundError(f"Video not found: {path}")
            local_source = str(path)

        duration = await _probe_duration(local_source)
        if duration <= 0:
            raise RuntimeError(f"Video has non-positive duration ({duration}s).")

        # Pick timestamps strictly inside the video to avoid empty last frame.
        if count == 1:
            timestamps = [duration / 2.0]
        else:
            step = duration / (count + 1)
            timestamps = [step * (i + 1) for i in range(count)]

        frame_paths: List[str] = []
        for i, ts in enumerate(timestamps, start=1):
            out_path = Path(tmpdir) / f"frame_{i:02d}.png"
            await _run_ffmpeg(
                [
                    "-ss",
                    f"{ts:.3f}",
                    "-i",
                    local_source,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(out_path),
                ]
            )
            frame_paths.append(str(out_path))

        if len(frame_paths) != count:
            raise RuntimeError(
                f"Expected {count} frames, extracted {len(frame_paths)}."
            )
        return frame_paths
    except Exception:
        # Clean up the temp dir on failure; on success the caller's image
        # loader will read the PNGs and we let the OS reclaim tmp on reboot.
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
