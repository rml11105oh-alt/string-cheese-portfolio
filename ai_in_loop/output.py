"""Output formatting for String Cheese dietary assistant results.

This module formats workflow results for CLI or Streamlit output.
Supports JSON (for programmatic use) and Markdown (for human reading).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import StringCheeseState


def format_json(state: "StringCheeseState") -> str:
    """Format workflow results as JSON.

    Example output:
    {
        "request": "...",
        "request_type": "dietary_plan",
        "plan": "...",
        "recipes": [...],
        "context": "...",
        "response": "...",
        "warnings": []
    }
    """

    output: dict[str, Any] = {
        "request": state.get("user_input"),
        "request_type": state.get("request_type"),
        "plan": state.get("plan"),
        "recipes": state.get("recipes"),
        "context": state.get("retrieved_context"),
        "response": state.get("response"),
        "warnings": state.get("warnings", []),
    }

    return json.dumps(output, indent=2, ensure_ascii=False)


def format_markdown(state: "StringCheeseState") -> str:
    """Format workflow results as Markdown for human-readable output."""

    lines: list[str] = []

    user_input = state.get("user_input")
    request_type = state.get("request_type")

    if user_input:
        lines.append("# String Cheese Result")
        lines.append("")
        lines.append(f"**Request:** {user_input}")
        lines.append("")

    if request_type:
        lines.append(f"**Request type:** {request_type}")
        lines.append("")

    plan = state.get("plan")
    if plan:
        lines.append("## Dietary Plan")
        lines.append("")
        lines.append(plan)
        lines.append("")

    recipes = state.get("recipes")
    if recipes:
        lines.append("## Recipe Suggestions")
        lines.append("")
        for recipe in recipes:
            lines.append(f"- {recipe}")
        lines.append("")

    context = state.get("retrieved_context")
    if context:
        lines.append("## Retrieved Dietary Information")
        lines.append("")
        lines.append(context)
        lines.append("")

    response = state.get("response")
    if response:
        lines.append("## Response")
        lines.append("")
        lines.append(response)
        lines.append("")

    warnings = state.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def format_output(state: "StringCheeseState", fmt: str = "json") -> str:
    """Format results in the specified format."""

    if fmt == "json":
        return format_json(state)
    elif fmt == "markdown":
        return format_markdown(state)
    else:
        raise ValueError(f"Unknown output format: {fmt}")