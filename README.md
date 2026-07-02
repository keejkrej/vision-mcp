# vision-mcp

An MCP (Model Context Protocol) server that **adds vision capabilities to text-only models** by routing image and video analysis requests to a **local Ollama vision model** (e.g. `gemma3:27b`, `gemma4:31b-cloud`, `llama3.2-vision`).

It mirrors the tool surface of the [Z.AI vision-mcp-server](https://docs.z.ai/devpack/mcp/vision-mcp-server) but replaces the hosted GLM-4.6V backend with your local Ollama instance via the [`/api/chat`](https://docs.ollama.com/capabilities/vision) endpoint, so no API key and no data leaves your machine.

## Features

- 8 vision tools, identical in spirit to the Z.AI server:
  - `image_analysis` - general-purpose image understanding
  - `ui_to_artifact` - turn UI screenshots into code/specs/descriptions
  - `extract_text_from_screenshot` - OCR-style text extraction
  - `diagnose_error_screenshot` - analyze error snapshots and propose fixes
  - `understand_technical_diagram` - interpret architecture/flow/UML/ER diagrams
  - `analyze_data_visualization` - read charts and dashboards
  - `ui_diff_check` - compare two UI screenshots for drift
  - `video_analysis` - sample frames from a video and describe scenes
- Local files, `http(s)` URLs, and raw base64 strings all accepted as image sources
- No external API keys; talks to `http://localhost:11434` by default
- `stdio` (default, for local clients) and `streamable-http` transports

## Prerequisites

1. **Ollama** installed and running: <https://ollama.com/>
2. A **vision model** pulled locally. Verify with:
   ```sh
   ollama list
   # If you don't have one yet:
   ollama pull gemma3:27b
   # Or the cloud variant (requires Ollama cloud access):
   ollama pull gemma4:31b-cloud
   ```
3. **Python 3.10+** with `uv` (recommended) or plain `pip`.
4. (Optional, only for `video_analysis`) **ffmpeg** on your PATH: `brew install ffmpeg`.

## Configuration

| Environment variable   | Default                 | Description                                            |
| ---------------------- | ----------------------- | ------------------------------------------------------ |
| `OLLAMA_HOST`          | `http://localhost:11434`| Base URL of your Ollama server.                         |
| `OLLAMA_VISION_MODEL`  | `gemma3:27b`            | Vision model used for all tools unless overridden per-call. |

Override the model per-call via each tool's optional `model` argument.

## Install

### Option A - run from source with `uv` (no install)

```sh
git clone <this repo> vision-mcp
cd vision-mcp
uv sync
uv run vision-mcp --help
```

### Option B - install as a package

```sh
uv pip install .
vision-mcp --help
```

## Connect to an MCP client

The server runs over `stdio` by default, which is what local MCP clients expect.

### Claude Code

```sh
claude mcp add -s user vision-mcp -- uv --directory /absolute/path/to/vision-mcp run vision-mcp
```

To target a non-default model:

```sh
claude mcp add -s user vision-mcp \
  --env OLLAMA_VISION_MODEL=gemma4:31b-cloud \
  -- uv --directory /absolute/path/to/vision-mcp run vision-mcp
```

### Claude Desktop / Cline / other MCP clients

Edit the client's `mcpServers` config:

```json
{
  "mcpServers": {
    "vision-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/vision-mcp", "run", "vision-mcp"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_VISION_MODEL": "gemma3:27b"
      }
    }
  }
}
```

If you installed it as a package, you can use `command": "vision-mcp"` with no args instead.

### Streamable HTTP transport

```sh
uv run vision-mcp --transport streamable-http --host 127.0.0.1 --port 8765
```

## Usage

After connecting, just ask your client about an image in your working directory:

```
What does demo.png describe?
OCR this terminal screenshot: error.png
Compare design_a.png and design_b.png for visual drift.
```

The client picks the right tool, the server loads the image, base64-encodes it, and sends it to your local Ollama vision model. The model's reply is returned to the (text-only) client as plain text.

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

## Troubleshooting

- **`Model 'X' is not available locally`** - run `ollama pull <model>` or set `OLLAMA_VISION_MODEL` to a model shown by `ollama list`.
- **Connection refused / timeout** - confirm Ollama is running: `ollama serve` or check `OLLAMA_HOST`.
- **`ffmpeg is not installed`** - only needed for `video_analysis`. Install it (`brew install ffmpeg`) or use the image tools instead.
- **Image too large** - resize/compress; the server caps images at 20 MiB to keep MCP calls responsive.

## License

MIT
