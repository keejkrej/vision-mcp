"""MCP tools that add vision capabilities to text-only models via local Ollama.

The tool set mirrors the Z.AI vision-mcp server (ui_to_artifact,
extract_text_from_screenshot, diagnose_error_screenshot,
understand_technical_diagram, analyze_data_visualization, ui_diff_check,
image_analysis, video_analysis) but routes every call to a local Ollama
vision model such as gemma3:27b or gemma4:31b-cloud instead of a hosted API.
"""

from __future__ import annotations

import json
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from vision_mcp.config import Settings, get_settings
from vision_mcp.ollama import VisionResult, check_model_available, vision_chat

# Tool annotations reused across read-only vision tools.
_READ_ONLY_ANNOTATIONS = {
    "title": "Vision analysis (local Ollama)",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}


# ---------------------------------------------------------------------------
# Shared input models (module-level so FastMCP can evaluate annotations)
# ---------------------------------------------------------------------------


class _BaseImageInput(BaseModel):
    """Common fields shared by every vision tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    image: str = Field(
        ...,
        description=(
            "Path or URL to the image. Local paths are resolved relative to the "
            "current working directory; http(s) URLs are downloaded. A raw base64 "
            "string is also accepted."
        ),
        min_length=1,
        max_length=2048,
    )
    question: Optional[str] = Field(
        default=None,
        description=(
            "Optional follow-up question or focus. If omitted, the tool uses its "
            "default task-specific prompt."
        ),
        max_length=2000,
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Override the configured Ollama vision model for this call "
            "(e.g. 'gemma3:27b', 'gemma4:31b-cloud', 'llama3.2-vision'). "
            "Defaults to OLLAMA_VISION_MODEL."
        ),
        max_length=120,
    )


class _DiffInput(BaseModel):
    """Input for the two-image ui_diff_check tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    image_a: str = Field(
        ...,
        description="First image (path, URL, or base64) for comparison.",
        min_length=1,
        max_length=2048,
    )
    image_b: str = Field(
        ...,
        description="Second image (path, URL, or base64) for comparison.",
        min_length=1,
        max_length=2048,
    )
    question: Optional[str] = Field(
        default=None,
        description="Optional focus, e.g. 'only color differences'.",
        max_length=2000,
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the configured Ollama vision model for this call.",
        max_length=120,
    )


class _VideoInput(BaseModel):
    """Input for the video_analysis tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video: str = Field(
        ...,
        description=(
            "Path or URL to a local/remote video. MP4/MOV/M4V recommended. "
            "The server extracts up to `frames` evenly-spaced frames and "
            "sends them to the vision model."
        ),
        min_length=1,
        max_length=2048,
    )
    question: Optional[str] = Field(
        default=None,
        description="Optional focus, e.g. 'describe the scene at 0:10'.",
        max_length=2000,
    )
    frames: int = Field(
        default=4,
        description="Number of evenly-spaced frames to sample (1-8).",
        ge=1,
        le=8,
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the configured Ollama vision model for this call.",
        max_length=120,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_result(result: VisionResult, *, tool_name: str) -> str:
    """Render a VisionResult as a compact JSON string for the MCP client."""
    payload = {
        "tool": tool_name,
        "model": result.model,
        "done": result.done,
        "description": result.content,
    }
    if result.total_duration_ns is not None:
        payload["total_duration_ms"] = round(result.total_duration_ns / 1e6, 1)
    if result.eval_duration_ns is not None:
        payload["eval_duration_ms"] = round(result.eval_duration_ns / 1e6, 1)
    return json.dumps(payload, indent=2)


def _format_error(message: str, *, tool_name: str, model: Optional[str] = None) -> str:
    return json.dumps(
        {"tool": tool_name, "error": message, "model": model}, indent=2
    )


async def _run(
    settings: Settings,
    *,
    tool_name: str,
    prompt: str,
    images: List[str],
    model: Optional[str],
    system: Optional[str] = None,
) -> str:
    """Shared body: check model, run vision_chat, format output or error."""
    target = model or settings.vision_model

    available = await check_model_available(settings, target)
    if not available:
        return _format_error(
            f"Model '{target}' is not available locally. Pull it with "
            f"`ollama pull {target}` or set OLLAMA_VISION_MODEL to an installed "
            "vision model (see `ollama list`).",
            tool_name=tool_name,
            model=target,
        )

    try:
        result = await vision_chat(
            settings,
            prompt=prompt,
            images=images,
            model=target,
            system=system,
        )
    except FileNotFoundError as exc:
        return _format_error(str(exc), tool_name=tool_name, model=target)
    except Exception as exc:  # surface actionable messages to the agent
        return _format_error(f"{type(exc).__name__}: {exc}", tool_name=tool_name, model=target)

    return _format_result(result, tool_name=tool_name)


def _compose_prompt(default: str, question: Optional[str]) -> str:
    if question and question.strip():
        return f"{default}\n\nFollow-up / question: {question.strip()}"
    return default


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def register_tools(mcp: FastMCP) -> None:
    """Register all vision tools on the given FastMCP server."""

    @mcp.tool(name="image_analysis", annotations=_READ_ONLY_ANNOTATIONS)
    async def image_analysis(params: _BaseImageInput) -> str:
        """General-purpose image understanding backed by a local Ollama vision model.

        Use this when no other, more specific vision tool fits. Describe the
        contents of a photo, screenshot, diagram, or rendered image. The image
        is sent to the configured Ollama vision model (OLLAMA_VISION_MODEL) and
        the model's description is returned.

        Args:
            params (_BaseImageInput): Validated input containing:
                - image (str): Local path, http(s) URL, or base64 string.
                - question (Optional[str]): Optional focus or follow-up question.
                - model (Optional[str]): Override the vision model for this call.

        Returns:
            str: JSON with {tool, model, done, description, *duration_ms} on
            success, or {tool, error, model} on failure.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "Describe this image clearly and concisely. Cover the main subjects, "
            "scene, notable objects, text, colors, and any relevant context. "
            "If the image is a screenshot or UI, describe the layout and visible "
            "elements.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="image_analysis",
            prompt=prompt,
            images=[params.image],
            model=params.model,
        )

    @mcp.tool(name="ui_to_artifact", annotations=_READ_ONLY_ANNOTATIONS)
    async def ui_to_artifact(params: _BaseImageInput) -> str:
        """Turn a UI screenshot into code, a spec, or a description via Ollama vision.

        Provide a screenshot of a user interface and receive a structured
        description suitable for rebuilding it (layout, components, typography,
        spacing, colors, and suggested HTML/CSS or component structure).

        Args:
            params (_BaseImageInput): Validated input containing:
                - image (str): Screenshot path, URL, or base64 string.
                - question (Optional[str]): Optional target (e.g. 'React + Tailwind',
                  'a spec only', 'a SwiftUI view').
                - model (Optional[str]): Override the vision model for this call.

        Returns:
            str: JSON with a `description` field containing the reconstruction
            guidance, plus model + timing metadata.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "You are a senior frontend engineer. Analyze this UI screenshot and "
            "produce a faithful reconstruction plan: layout structure, components, "
            "typography, spacing, color palette (with approximate hex values), and "
            "interactive elements. Provide a clean HTML/CSS sketch (or the framework "
            "implied by the question) that reproduces the look. Note any responsive "
            "considerations.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="ui_to_artifact",
            prompt=prompt,
            images=[params.image],
            model=params.model,
        )

    @mcp.tool(name="extract_text_from_screenshot", annotations=_READ_ONLY_ANNOTATIONS)
    async def extract_text_from_screenshot(params: _BaseImageInput) -> str:
        """OCR-style text extraction from a screenshot using a local vision model.

        Extracts visible text from screenshots of code, terminals, documents, or
        general UI. Returns the text as faithfully as possible, preserving line
        breaks and structure. Not a pixel-perfect OCR engine, but works well for
        readable screenshots without needing a dedicated OCR dependency.

        Args:
            params (_BaseImageInput): Validated input containing:
                - image (str): Screenshot path, URL, or base64 string.
                - question (Optional[str]): Optional filter, e.g. 'only the error
                  message' or 'the third paragraph'.
                - model (Optional[str]): Override the vision model for this call.

        Returns:
            str: JSON with a `description` field containing the extracted text.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "Extract all visible text from this screenshot as faithfully as "
            "possible. Preserve line breaks, indentation, and ordering. Output "
            "only the extracted text inside a single fenced code block; do not "
            "add commentary. If no text is visible, output an empty block.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="extract_text_from_screenshot",
            prompt=prompt,
            images=[params.image],
            model=params.model,
        )

    @mcp.tool(name="diagnose_error_screenshot", annotations=_READ_ONLY_ANNOTATIONS)
    async def diagnose_error_screenshot(params: _BaseImageInput) -> str:
        """Analyze an error snapshot and propose actionable fixes via Ollama vision.

        Reads a screenshot of an error, stack trace, failed test, or crashed UI
        and returns: (1) a restatement of the error, (2) likely root cause(s),
        and (3) concrete next steps to fix it.

        Args:
            params (_BaseImageInput): Validated input containing:
                - image (str): Screenshot path, URL, or base64 string.
                - question (Optional[str]): Optional context, e.g. the language,
                  framework, or what you were trying to do.
                - model (Optional[str]): Override the vision model for this call.

        Returns:
            str: JSON with a `description` field containing the diagnosis and
            suggested fixes.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "You are a debugging assistant. This screenshot shows an error or "
            "failure. (1) Restate the error message verbatim. (2) Identify the "
            "most likely root cause(s). (3) Propose concrete, ordered steps to "
            "fix it. If the error is ambiguous, say what additional information "
            "would help.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="diagnose_error_screenshot",
            prompt=prompt,
            images=[params.image],
            model=params.model,
        )

    @mcp.tool(name="understand_technical_diagram", annotations=_READ_ONLY_ANNOTATIONS)
    async def understand_technical_diagram(params: _BaseImageInput) -> str:
        """Interpret architecture, flow, UML, ER, or system diagrams via Ollama vision.

        Reads a technical diagram and explains its components, relationships, and
        overall purpose in plain language, then summarizes the system it depicts.

        Args:
            params (_BaseImageInput): Validated input containing:
                - image (str): Diagram path, URL, or base64 string.
                - question (Optional[str]): Optional focus, e.g. 'the data flow
                  from A to B' or 'list every entity'.
                - model (Optional[str]): Override the vision model for this call.

        Returns:
            str: JSON with a `description` field containing the interpretation.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "You are a senior software architect. Interpret this technical "
            "diagram. Identify the type (architecture, sequence, flow, UML, ER, "
            "system, etc.), enumerate the key components and their roles, describe "
            "the relationships and data flow, and summarize what the overall "
            "system does. Be precise about labeled arrows and annotations.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="understand_technical_diagram",
            prompt=prompt,
            images=[params.image],
            model=params.model,
        )

    @mcp.tool(name="analyze_data_visualization", annotations=_READ_ONLY_ANNOTATIONS)
    async def analyze_data_visualization(params: _BaseImageInput) -> str:
        """Read charts and dashboards to surface insights and trends via Ollama vision.

        Analyzes a chart, graph, plot, or dashboard image and reports the chart
        type, axes, series, key values, trends, outliers, and takeaways.

        Args:
            params (_BaseImageInput): Validated input containing:
                - image (str): Chart/dashboard path, URL, or base64 string.
                - question (Optional[str]): Optional focus, e.g. 'the trend in
                  series B after 2023'.
                - model (Optional[str]): Override the vision model for this call.

        Returns:
            str: JSON with a `description` field containing the analysis.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "You are a data analyst. Analyze this data visualization. Identify "
            "the chart type, axes, units, and series. Report key values, trends, "
            "comparisons, outliers, and the main takeaway. If values are hard to "
            "read, estimate them and say so. Do not invent data.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="analyze_data_visualization",
            prompt=prompt,
            images=[params.image],
            model=params.model,
        )

    @mcp.tool(name="ui_diff_check", annotations=_READ_ONLY_ANNOTATIONS)
    async def ui_diff_check(params: _DiffInput) -> str:
        """Compare two UI screenshots to flag visual or implementation drift.

        Sends both images to the vision model in a single call and reports
        differences in layout, spacing, typography, color, components, and
        visible behavior. Useful for regression checks between design and
        implementation.

        Args:
            params (_DiffInput): Validated input with image_a, image_b, optional
            question, and optional model override.

        Returns:
            str: JSON with a `description` field listing the differences, plus
            model + timing metadata.
        """
        settings = get_settings()
        prompt = _compose_prompt(
            "Compare these two UI screenshots (image A and image B). Enumerate "
            "concrete differences in layout, spacing, typography, color, "
            "components, and visible behavior. Group differences by severity "
            "(breaking / notable / trivial). If they are visually equivalent, "
            "say so explicitly.",
            params.question,
        )
        return await _run(
            settings,
            tool_name="ui_diff_check",
            prompt=prompt,
            images=[params.image_a, params.image_b],
            model=params.model,
        )

    @mcp.tool(name="video_analysis", annotations=_READ_ONLY_ANNOTATIONS)
    async def video_analysis(params: _VideoInput) -> str:
        """Inspect a video by sampling frames and describing scenes via Ollama vision.

        Extracts a small number of evenly-spaced frames from a local or remote
        video, sends them as a sequence to the vision model, and returns a
        description of scenes, moments, and entities. Requires `ffmpeg` on PATH
        for frame extraction.

        Args:
            params (_VideoInput): Validated input with video path/URL, optional
            question, frame count (1-8, default 4), and optional model override.

        Returns:
            str: JSON with a `description` field containing the scene description,
            plus model + timing metadata. Returns an error JSON if ffmpeg is
            missing or frame extraction fails.
        """
        from vision_mcp.video import extract_frame_paths

        settings = get_settings()
        try:
            frame_paths = await extract_frame_paths(
                params.video, count=params.frames
            )
        except Exception as exc:
            return _format_error(
                f"{type(exc).__name__}: {exc}",
                tool_name="video_analysis",
                model=params.model or settings.vision_model,
            )

        prompt = _compose_prompt(
            "These images are evenly-spaced frames sampled from a video, shown "
            "in chronological order. Describe the video: the overall scene, "
            "notable moments, people or objects, actions, and any visible text. "
            "Reference frames by their order (frame 1, frame 2, ...).",
            params.question,
        )
        return await _run(
            settings,
            tool_name="video_analysis",
            prompt=prompt,
            images=frame_paths,
            model=params.model,
        )
