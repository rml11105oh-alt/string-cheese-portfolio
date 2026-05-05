from dataclasses import dataclass
import os
import sys
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    use_gemini: bool
    gemini_api_key: str | None
    gemini_model: str
    temperature: float
    thinking_level: str | None  # For Gemini 3+: None, "low", "medium", "high"
    thinking_budget: int | None  # For Gemini 2.5: None or 0 = disabled, positive int = token budget
    system_prompt_file: str
    resources_dir: str
    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def from_env() -> "Config":
        use_gemini = os.getenv("USE_GEMINI", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_api_key = gemini_api_key.strip() if gemini_api_key else None

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        temp_str = os.getenv("GEMINI_TEMPERATURE", "0.7").strip()
        thinking_level_str = os.getenv("GEMINI_THINKING_LEVEL", "none").strip().lower()
        system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "prompts/empty.md").strip()

        try:
            temperature = float(temp_str)
        except ValueError:
            print(f"Warning: GEMINI_TEMPERATURE '{temp_str}' is not a valid number, using 0.7", file=sys.stderr)
            temperature = 0.7

        # Parse thinking settings - only warn about the one relevant to the current model
        is_gemini_3 = gemini_model.startswith("gemini-3")

        # Thinking level for Gemini 3+ models
        valid_thinking_levels = {"none", "low", "medium", "high"}
        if thinking_level_str in valid_thinking_levels:
            thinking_level = None if thinking_level_str == "none" else thinking_level_str
        else:
            if is_gemini_3:
                print(f"Warning: GEMINI_THINKING_LEVEL '{thinking_level_str}' is not valid (use none/low/medium/high), using none", file=sys.stderr)
            thinking_level = None

        # Thinking budget for Gemini 2.5 models
        thinking_budget_str = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
        if thinking_budget_str:
            try:
                thinking_budget = int(thinking_budget_str)
            except ValueError:
                if not is_gemini_3:
                    print(f"Warning: GEMINI_THINKING_BUDGET '{thinking_budget_str}' is not a valid integer, using 0", file=sys.stderr)
                thinking_budget = 0
        else:
            thinking_budget = 0  # Default: no thinking

        if system_prompt_file and not Path(system_prompt_file).exists():
            print(f"Warning: SYSTEM_PROMPT_FILE '{system_prompt_file}' not found, using no system instruction", file=sys.stderr)
            system_prompt_file = ""

        # Retrieval configuration
        resources_dir = os.getenv("RESOURCES_DIR", "resources").strip()

        try:
            chunk_size = int(os.getenv("CHUNK_SIZE", "1000").strip())
            chunk_size = max(100, min(chunk_size, 10000))  # Bound to reasonable range
        except ValueError:
            print("Warning: CHUNK_SIZE is not a valid integer, using 1000", file=sys.stderr)
            chunk_size = 1000

        try:
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100").strip())
            chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))  # Can't exceed half of chunk_size
        except ValueError:
            print("Warning: CHUNK_OVERLAP is not a valid integer, using 100", file=sys.stderr)
            chunk_overlap = 100

        return Config(
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            temperature=temperature,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            system_prompt_file=system_prompt_file,
            resources_dir=resources_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
