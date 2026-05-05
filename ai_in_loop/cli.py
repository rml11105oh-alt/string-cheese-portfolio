from __future__ import annotations

import typer
from dotenv import load_dotenv
from rich.console import Console

from .config import Config
from .graph import build_app, get_graph_image
from .logging import log_event, new_run_id


console = Console()
app = typer.Typer(help="String Cheese CLI - dietary planning and recipe assistant")


def _print_state_result(result: dict) -> str:
    """Print the most useful result field from workflow state and return it."""
    final_text = (
        result.get("response")
        or result.get("plan")
        or "\n".join(result.get("recipes", [])) if result.get("recipes") else ""
    )

    # The above expression can be a little hard to read, so normalize it.
    if isinstance(final_text, bool):
        final_text = ""

    if not final_text:
        if result.get("recipes"):
            final_text = "\n".join(f"- {recipe}" for recipe in result["recipes"])
        elif result.get("retrieved_context"):
            final_text = result["retrieved_context"]
        else:
            final_text = ""

    warnings = result.get("warnings", [])
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")
        console.print()

    if final_text:
        console.print(final_text)
    else:
        console.print("[yellow]No response generated.[/yellow]")

    return final_text


def _read_multiline_input() -> str:
    """Read multi-line input until a blank line is entered."""
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


@app.command()
def plan(
    goal: str,
    calories: int | None = None,
    dietary_preference: str | None = None,
    allergies: str | None = None,
) -> None:
    """Generate a dietary plan based on user goals."""
    load_dotenv()
    cfg = Config.from_env()
    run_id = new_run_id()

    prompt_lines = [
        f"Goal: {goal}",
        f"Calories target: {calories}" if calories is not None else None,
        f"Dietary preference: {dietary_preference}" if dietary_preference else None,
        f"Allergies: {allergies}" if allergies else None,
        "",
        "Generate a simple daily meal plan and brief explanation.",
    ]
    prompt = "\n".join(line for line in prompt_lines if line is not None)

    graph_app = build_app(cfg)
    result = graph_app.invoke(
        {
            "user_input": prompt,
            "warnings": [],
        }
    )

    final_response = _print_state_result(result)

    log_event(
        {
            "run_id": run_id,
            "event": "plan",
            "prompt": prompt,
            "response": final_response,
            "request_type": result.get("request_type"),
            "warnings": result.get("warnings", []),
            "use_gemini": cfg.use_gemini,
            "gemini_model": cfg.gemini_model,
            "temperature": cfg.temperature,
        }
    )


@app.command()
def ask(prompt: str) -> None:
    """Run a single String Cheese request."""
    load_dotenv()
    cfg = Config.from_env()
    run_id = new_run_id()

    graph_app = build_app(cfg)
    result = graph_app.invoke(
        {
            "user_input": prompt,
            "warnings": [],
        }
    )

    final_response = _print_state_result(result)

    log_event(
        {
            "run_id": run_id,
            "event": "ask",
            "prompt": prompt,
            "response": final_response,
            "request_type": result.get("request_type"),
            "warnings": result.get("warnings", []),
            "use_gemini": cfg.use_gemini,
            "gemini_model": cfg.gemini_model,
            "temperature": cfg.temperature,
        }
    )


@app.command()
def chat() -> None:
    """Interactive chat loop for String Cheese."""
    load_dotenv()
    cfg = Config.from_env()

    console.print("[bold]Chat mode[/bold]")
    console.print("  - Enter a blank line to send your message")
    console.print("  - Type 'exit' to quit")
    console.print("  - Type '/reset' to clear the session\n")

    graph_app = build_app(cfg)

    while True:
        console.print("[bold cyan]You>[/bold cyan]")
        prompt = _read_multiline_input().strip()

        if prompt.lower() in {"exit", "quit"}:
            break

        if prompt.lower() in {"/reset", "reset"}:
            console.print("[yellow]Conversation reset.[/yellow]\n")
            continue

        if not prompt:
            continue

        run_id = new_run_id()

        result = graph_app.invoke(
            {
                "user_input": prompt,
                "warnings": [],
            }
        )

        console.print()
        final_response = (
            result.get("response")
            or result.get("plan")
            or ""
        )

        if not final_response and result.get("recipes"):
            final_response = "\n".join(f"- {recipe}" for recipe in result["recipes"])
        if not final_response and result.get("retrieved_context"):
            final_response = result["retrieved_context"]

        if final_response:
            console.print(f"[bold green]Model>[/bold green] {final_response}\n")
        else:
            console.print("[yellow]Model didn't respond. Try rephrasing your question.[/yellow]\n")

        warnings = result.get("warnings", [])
        if warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  - {warning}")
            console.print()

        log_event(
            {
                "run_id": run_id,
                "event": "chat_turn",
                "prompt": prompt,
                "response": final_response,
                "request_type": result.get("request_type"),
                "warnings": warnings,
                "use_gemini": cfg.use_gemini,
                "gemini_model": cfg.gemini_model,
                "temperature": cfg.temperature,
            }
        )


@app.command()
def visualize(output_path: str):
    """Generate a PNG visualization of the workflow graph."""
    load_dotenv()
    cfg = Config.from_env()

    image_bytes = get_graph_image(cfg, output_path)

    if image_bytes is None:
        console.print(
            "[bold red]Could not generate graph image.[/bold red]\n"
            "Make sure graph visualization dependencies are installed."
        )
        raise typer.Exit(code=1)

    console.print(f"[bold green]Graph saved to:[/bold green] {output_path}")


if __name__ == "__main__":
    app()