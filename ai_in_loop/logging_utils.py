from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return str(uuid.uuid4())


def log_event(event: dict[str, Any], log_path: str | Path = "logs/runs.jsonl") -> None:
    """Append a JSONL event to logs/runs.jsonl (creates directories if needed)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    event = dict(event)  # copy
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_tool_call(
    run_id: str,
    tool_name: str,
    args: dict[str, Any],
    log_path: str | Path = "logs/runs.jsonl",
) -> None:
    """Log a tool call event.

    Args:
        run_id: Unique identifier for this run
        tool_name: Name of the tool being called
        args: Arguments passed to the tool
        log_path: Path to the log file
    """
    log_event(
        {
            "run_id": run_id,
            "event": "tool_call",
            "tool_name": tool_name,
            "args": args,
        },
        log_path,
    )


def log_tool_result(
    run_id: str,
    tool_name: str,
    result: str,
    elapsed_seconds: float | None = None,
    is_error: bool = False,
    log_path: str | Path = "logs/runs.jsonl",
) -> None:
    """Log a tool result event.

    Args:
        run_id: Unique identifier for this run
        tool_name: Name of the tool that was called
        result: The result returned by the tool (truncated if too long)
        elapsed_seconds: Time taken to execute the tool
        is_error: Whether the result is an error
        log_path: Path to the log file
    """
    # Truncate long results
    max_result_length = 1000
    if len(result) > max_result_length:
        result = result[:max_result_length] + "... (truncated)"

    event = {
        "run_id": run_id,
        "event": "tool_result",
        "tool_name": tool_name,
        "result": result,
        "is_error": is_error,
    }

    if elapsed_seconds is not None:
        event["elapsed_seconds"] = elapsed_seconds

    log_event(event, log_path)
