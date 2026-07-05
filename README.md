# vision-mcp

An MCP server that **adds vision capabilities to text-only models** by routing image and video analysis requests to a **local Ollama vision model** (e.g. `gemma4:31b-cloud`). No API key, no data leaves your machine.

## Prerequisites

1. **Ollama** installed and running: <https://ollama.com/>
2. A **vision model** pulled locally:
   ```sh
   ollama pull gemma4:31b-cloud
   ```
3. **Python 3.10+** with `uv`.
4. (Optional, only for `video_analysis`) **ffmpeg** on your PATH.

## Install

```sh
uv tool install git+https://github.com/keejkrej/vision-mcp.git
vision-mcp --help
```

Upgrade with `uv tool upgrade vision-mcp`.

## Connect to an MCP client

The server runs over `stdio` by default.

### Claude Code

```sh
claude mcp add -s user vision-mcp -- vision-mcp
```

### Claude Desktop / Cline / other MCP clients

```json
{
  "mcpServers": {
    "vision-mcp": {
      "type": "stdio",
      "command": "vision-mcp",
      "env": {
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_VISION_MODEL": "gemma4:31b-cloud"
      }
    }
  }
}
```

### Streamable HTTP transport

```sh
vision-mcp --transport streamable-http --host 127.0.0.1 --port 8765
```

## Configuration

| Environment variable   | Default                 | Description                                            |
| ---------------------- | ----------------------- | ---------------------------------------------------- |
| `OLLAMA_HOST`          | `http://localhost:11434`| Base URL of your Ollama server.                       |
| `OLLAMA_VISION_MODEL`  | `gemma4:31b-cloud`      | Vision model used for all tools unless overridden per-call. |

Override the model per-call via each tool's optional `model` argument.

## Tools

| Tool                          | Inputs                                      | Purpose                                                  |
| ----------------------------- | ------------------------------------------- | -------------------------------------------------------- |
| `image_analysis`              | `image`, `question?`, `model?`              | General image description.                               |
| `ui_to_artifact`              | `image`, `question?`, `model?`              | Reconstruct UI screenshots as code/specs.                |
| `extract_text_from_screenshot`| `image`, `question?`, `model?`              | Extract visible text from screenshots.                   |
| `diagnose_error_screenshot`   | `image`, `question?`, `model?`              | Diagnose errors and propose fixes.                       |
| `understand_technical_diagram`| `image`, `question?`, `model?`              | Interpret technical/architecture diagrams.               |
| `analyze_data_visualization`  | `image`, `question?`, `model?`              | Read charts and dashboards.                              |
| `ui_diff_check`               | `image_a`, `image_b`, `question?`, `model?` | Compare two UI screenshots.                              |
| `video_analysis`              | `video`, `question?`, `frames?`, `model?`   | Sample frames from a video and describe scenes.          |

Images can be local files, `http(s)` URLs, or raw base64 strings (capped at 20 MiB).

## Troubleshooting

- **`Model 'X' is not available locally`** - run `ollama pull <model>` or set `OLLAMA_VISION_MODEL`.
- **Connection refused / timeout** - confirm Ollama is running: `ollama serve`.
- **`ffmpeg is not installed`** - only needed for `video_analysis`.

## License

MIT
