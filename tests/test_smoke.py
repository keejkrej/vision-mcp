"""Smoke tests that don't require a running Ollama server or real models.

These verify the package imports, the FastMCP server registers the expected
tools, the CLI parses args, and the Ollama client handles a missing image
gracefully. Run with: `uv run pytest -q` after `uv sync --dev`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable when running from a source checkout without install.
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_imports():
    import vision_mcp
    from vision_mcp import ollama, server, tools, video, config

    assert vision_mcp.__version__
    assert server.mcp
    assert callable(tools.register_tools)
    assert hasattr(ollama, "vision_chat")
    assert hasattr(video, "extract_frame_paths")
    assert hasattr(config, "get_settings")


def test_settings_from_env(monkeypatch):
    from vision_mcp.config import get_settings

    monkeypatch.setenv("OLLAMA_HOST", "http://myhost:1234")
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "gemma4:31b-cloud")
    s = get_settings()
    assert s.ollama_host == "http://myhost:1234"
    assert s.vision_model == "gemma4:31b-cloud"


def test_settings_defaults(monkeypatch):
    from vision_mcp.config import get_settings

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    s = get_settings()
    assert s.ollama_host == "http://localhost:11434"
    assert s.vision_model == "gemma3:27b"


def test_expected_tools_registered():
    from vision_mcp.server import mcp

    # FastMCP exposes registered tools via the tool manager.
    tools_map = mcp._tool_manager._tools  # type: ignore[attr-defined]
    expected = {
        "image_analysis",
        "ui_to_artifact",
        "extract_text_from_screenshot",
        "diagnose_error_screenshot",
        "understand_technical_diagram",
        "analyze_data_visualization",
        "ui_diff_check",
        "video_analysis",
    }
    registered = set(tools_map.keys())
    missing = expected - registered
    assert not missing, f"Missing tools: {missing}"


def test_cli_parser_smoke():
    from vision_mcp.server import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(["--transport", "streamable-http", "--port", "9000", "--model", "x:1"])
    assert ns.transport == "streamable-http"
    assert ns.port == 9000
    assert ns.model == "x:1"

    ns2 = parser.parse_args([])
    assert ns2.transport == "stdio"


def test_load_image_missing_file_raises():
    from vision_mcp.ollama import _load_image_bytes
    import httpx

    async def go():
        async with httpx.AsyncClient() as client:
            with pytest.raises(FileNotFoundError):
                await _load_image_bytes("/definitely/does/not/exist.png", client)

    asyncio.run(go())


def test_load_image_local_file(tmp_path):
    from vision_mcp.ollama import _load_image_bytes
    import httpx

    img = tmp_path / "t.bin"
    img.write_bytes(b"fake-png-bytes")

    async def go():
        async with httpx.AsyncClient() as client:
            data = await _load_image_bytes(str(img), client)
            return data

    assert asyncio.run(go()) == b"fake-png-bytes"


def test_format_result_and_error():
    from vision_mcp.ollama import VisionResult
    from vision_mcp.tools import _format_result, _format_error
    import json

    res = VisionResult(
        content="hello",
        model="m",
        done=True,
        eval_duration_ns=1_000_000,
        total_duration_ns=2_000_000,
    )
    parsed = json.loads(_format_result(res, tool_name="image_analysis"))
    assert parsed["description"] == "hello"
    assert parsed["tool"] == "image_analysis"
    assert parsed["eval_duration_ms"] == 1.0

    err = json.loads(_format_error("boom", tool_name="image_analysis", model="m"))
    assert err["error"] == "boom"
