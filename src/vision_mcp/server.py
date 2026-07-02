"""vision-mcp server entrypoint.

Exposes a stdio MCP server that adds vision capabilities to text-only models
by routing image/video analysis requests to a local Ollama vision model.

Configuration (environment variables):
    OLLAMA_HOST          Base URL of the Ollama server (default http://localhost:11434)
    OLLAMA_VISION_MODEL  Vision model to use (default gemma3:27b)

Run directly:
    python -m vision_mcp
    vision-mcp            # via the console script
"""

from __future__ import annotations

import argparse
import sys

from mcp.server.fastmcp import FastMCP

from vision_mcp.config import Settings, get_settings
from vision_mcp.tools import register_tools

# Single FastMCP instance shared by CLI, module import, and tests.
mcp = FastMCP("vision_mcp")
register_tools(mcp)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-mcp",
        description=(
            "MCP server that adds vision capabilities to text-only models via a "
            "local Ollama vision model (e.g. gemma3:27b, gemma4:31b-cloud)."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio for local clients).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for streamable-http transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for streamable-http transport (default: 8765).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override OLLAMA_VISION_MODEL for this process. Does not persist to "
            "the environment."
        ),
    )
    parser.add_argument(
        "--ollama-host-url",
        default=None,
        help="Override OLLAMA_HOST for this process.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Parses args, applies overrides, and runs the MCP server."""
    args = _build_parser().parse_args(argv)

    if args.model or args.ollama_host_url:
        import os

        if args.ollama_host_url:
            os.environ["OLLAMA_HOST"] = args.ollama_host_url
        if args.model:
            os.environ["OLLAMA_VISION_MODEL"] = args.model

    settings: Settings = get_settings()
    print(
        f"[vision-mcp] ollama={settings.ollama_host} model={settings.vision_model} "
        f"transport={args.transport}",
        file=sys.stderr,
    )

    if args.transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )
    else:
        mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
