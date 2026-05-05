from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import Config
from .llm import get_llm
from .nodes import (
    collect_plan_context,
    collect_recipe_context,
    calculate_calories,
    calculate_macros,
    format_response,
    general_chat_response,
    generate_plan,
    parse_input,
    route_request,
    search_recipes,
    set_llm,
    weightloss_plan,
)
from .state import StringCheeseState
from .tools import set_search_config


def build_app(cfg: Config) -> CompiledStateGraph:
    """Build and compile the String Cheese workflow graph."""
    llm = get_llm(cfg)
    set_llm(llm)
    set_search_config(cfg)

    graph = StateGraph(StringCheeseState)

    graph.add_node("parse_input", parse_input)

    # Dietary plan nodes
    graph.add_node("collect_plan_context", collect_plan_context)
    graph.add_node("generate_plan", generate_plan)

    # Nutrition tool nodes
    graph.add_node("weightloss_plan", weightloss_plan)
    graph.add_node("calculate_calories", calculate_calories)
    graph.add_node("calculate_macros", calculate_macros)

    # Recipe search nodes
    graph.add_node("collect_recipe_context", collect_recipe_context)
    graph.add_node("search_recipes", search_recipes)

    # General response / formatting nodes
    graph.add_node("general_chat_response", general_chat_response)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "parse_input")

    graph.add_conditional_edges(
        "parse_input",
        route_request,
        {
            "recipe_search": "collect_recipe_context",
            "weightloss_plan": "weightloss_plan",
            "calculate_calories": "calculate_calories",
            "calculate_macros": "calculate_macros",
            "dietary_plan": "collect_plan_context",
            "general_chat_response": "general_chat_response",
        },
    )

    graph.add_edge("collect_plan_context", "generate_plan")
    graph.add_edge("generate_plan", "format_response")

    graph.add_edge("weightloss_plan", "format_response")
    graph.add_edge("calculate_calories", "format_response")
    graph.add_edge("calculate_macros", "format_response")

    graph.add_edge("collect_recipe_context", "search_recipes")
    graph.add_edge("search_recipes", "format_response")

    graph.add_edge("general_chat_response", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


def get_graph_image(cfg: Config, output_path: str | None = None) -> bytes | None:
    """Generate a PNG image of the workflow graph."""
    try:
        graph = build_app(cfg)
        image_bytes = graph.get_graph().draw_mermaid_png()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(image_bytes)

        return image_bytes

    except Exception as e:
        print(f"Error generating graph image: {e}")
        return None