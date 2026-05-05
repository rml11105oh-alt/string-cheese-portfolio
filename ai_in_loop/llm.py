import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import Config


# Patterns that suggest a math calculation request
MATH_PATTERNS = [
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bwhat is\b.*\d",
    r"\bhow much\b",
    r"\bsum of\b",
    r"\bproduct of\b",
    r"\bsquare root\b",
    r"\bsqrt\b",
    r"\d+\s*[\+\-\*\/\^]\s*\d+",  # Basic arithmetic like "2 + 2"
    r"\d+\s*\*\*\s*\d+",  # Python power operator
]

# Patterns that suggest a document search request
SEARCH_PATTERNS = [
    r"\bsearch\b",
    r"\bfind\b.*document",
    r"\blook up\b",
    r"\bwhat does.*say about\b",
    r"\baccording to\b",
]


def load_system_prompt(file_path: str) -> str | None:
    """Load system prompt from a markdown file. Returns None if file is empty or missing."""
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content if content else None


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing without API calls.

    Returns a predictable response based on the input prompt, useful for
    stable tests and offline development. Supports tool binding for testing
    tool-calling workflows.
    """

    # List of bound tools (empty by default)
    tools: list = []

    @property
    def _llm_type(self) -> str:
        return "mock"

    def bind_tools(self, tools: list, **kwargs: Any) -> "MockChatModel":
        """Return a new MockChatModel with tools bound.

        This mimics the behavior of real LLM's bind_tools() method,
        allowing the mock to simulate tool calling behavior.

        Args:
            tools: List of tools to bind
            **kwargs: Additional arguments (ignored)

        Returns:
            A new MockChatModel instance with tools bound
        """
        new_model = MockChatModel()
        new_model.tools = list(tools)
        return new_model

    def _is_math_request(self, text: str) -> bool:
        """Check if the text appears to be a math calculation request."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in MATH_PATTERNS)

    def _is_search_request(self, text: str) -> bool:
        """Check if the text appears to be a document search request."""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in SEARCH_PATTERNS)

    def _extract_expression(self, text: str) -> str:
        """Extract a math expression from the text.

        This is a simple heuristic that looks for common patterns.
        For more complex extraction, the real LLM would handle it.
        """
        text_lower = text.lower()

        # Try to find function calls like sqrt(144), sin(30), log(100)
        func_match = re.search(
            r"\b(sqrt|sin|cos|tan|log|log10|log2|exp|abs|floor|ceil)\s*\(\s*([^)]+)\s*\)",
            text_lower,
        )
        if func_match:
            func_name = func_match.group(1)
            func_arg = func_match.group(2).strip()
            return f"{func_name}({func_arg})"

        # Try to find explicit arithmetic expressions
        # Pattern: numbers with operators
        match = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)+)", text)
        if match:
            # Convert ^ to ** for Python
            return match.group(1).replace("^", "**")

        # Try to find "X times Y" or "X plus Y" style
        word_ops = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "to the power of": "**",
        }
        for word, op in word_ops.items():
            match = re.search(rf"(\d+(?:\.\d+)?)\s*{word}\s*(\d+(?:\.\d+)?)", text_lower)
            if match:
                return f"{match.group(1)} {op} {match.group(2)}"

        # Default: just return any numbers found with a + between them
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if len(numbers) >= 2:
            return " + ".join(numbers[:2])
        elif len(numbers) == 1:
            return numbers[0]

        return "0"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Check if there's already a tool result in the messages
        # If so, generate a final response based on the tool result
        tool_result = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_result = str(msg.content)
                break

        if tool_result is not None:
            # Generate a response based on the tool result
            mock_text = f"[MOCK] The calculation result is: {tool_result}"
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=mock_text))]
            )

        # Find the last human message content
        human_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                human_content = str(msg.content)
                break

        prompt = human_content

        # If tools are bound and this looks like a search request,
        # generate a search_docs tool call
        if self.tools and self._is_search_request(prompt):
            search_tool = None
            for tool in self.tools:
                if hasattr(tool, "name") and tool.name == "search_docs":
                    search_tool = tool
                    break

            if search_tool:
                tool_call = {
                    "name": "search_docs",
                    "args": {"query": prompt},
                    "id": str(uuid.uuid4()),
                }
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content="", tool_calls=[tool_call])
                        )
                    ]
                )

        # If tools are bound and this looks like a math request,
        # generate a tool call instead of a regular response
        if self.tools and self._is_math_request(prompt):
            # Find the python_calc tool
            calc_tool = None
            for tool in self.tools:
                if hasattr(tool, "name") and tool.name == "python_calc":
                    calc_tool = tool
                    break

            if calc_tool:
                expression = self._extract_expression(prompt)
                tool_call = {
                    "name": "python_calc",
                    "args": {"expression": expression},
                    "id": str(uuid.uuid4()),
                }
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content="", tool_calls=[tool_call])
                        )
                    ]
                )

        # Default mock response
        line_count = prompt.count("\n") + 1
        char_count = len(prompt)
        mock_text = f"[MOCK] Received {line_count} line(s), {char_count} char(s):\n---\n{prompt}\n---"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=mock_text))]
        )


def get_llm(cfg: Config) -> BaseChatModel:
    """Get a LangChain chat model based on configuration.

    Returns a MockChatModel if USE_GEMINI is disabled or no API key is provided.
    Otherwise returns a ChatGoogleGenerativeAI instance.

    This abstraction allows easy swapping of LLM providers by changing the
    implementation here.
    """
    if not cfg.use_gemini:
        return MockChatModel()

    if not cfg.gemini_api_key:
        # Fall back to mock rather than crash
        return MockChatModel()

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {
        "model": cfg.gemini_model,
        "temperature": cfg.temperature,
        "google_api_key": cfg.gemini_api_key,
    }

    # Pass appropriate thinking parameter based on model
    if cfg.gemini_model.startswith("gemini-3"):
        # Gemini 3+: use thinking_level if set
        if cfg.thinking_level:
            kwargs["thinking_level"] = cfg.thinking_level
    else:
        # Gemini 2.5 and earlier: use thinking_budget
        if cfg.thinking_budget is not None:
            kwargs["thinking_budget"] = cfg.thinking_budget

    return ChatGoogleGenerativeAI(**kwargs)

def get_text(response) -> str:
    """Extract text from LLM response.

    Gemini 3+ models return structured content (for extended thinking).
    This helper normalizes both string and list content to plain text.

    Args:
        response: The LLM response object (AIMessage or similar)

    Returns:
        Extracted text content as a string
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        ).strip()
    return ""
