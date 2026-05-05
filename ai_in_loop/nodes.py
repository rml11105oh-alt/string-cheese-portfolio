from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage

from .state import StringCheeseState
#from .tools import search_docs
from .tools import search_docs, weight_planning, calorie_counter, macro_tracker

from .dietary_plan.planner import (
    detect_plan_warnings,
    extract_dietary_context,
    generate_dietary_plan,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


_llm: BaseChatModel | None = None


def set_llm(llm: BaseChatModel) -> None:
    global _llm
    _llm = llm


def parse_input(state: StringCheeseState) -> StringCheeseState:
    """Normalize and validate incoming user input."""
    user_input = (state.get("user_input") or "").strip()
    warnings = list(state.get("warnings", []))

    if not user_input:
        warnings.append("No user input provided.")

    return {
        **state,
        "user_input": user_input,
        "warnings": warnings,
    }


def route_request(
    state: StringCheeseState,
    ) -> Literal[
        "weightloss_plan",
        "calculate_calories",
        "calculate_macros",
        "dietary_plan",
        "recipe_search",
        "general_chat_response",
    ]:
    """Route based on classified request type."""
    request_type = state.get("request_type")

    if request_type == "weightloss_plan":
        return "weightloss_plan"

    if request_type in ["calculate_calories", "calorie_counter"]:
        return "calculate_calories"

    if request_type == "calculate_macros":
        return "calculate_macros"
      
    if request_type == "dietary_plan":
        return "dietary_plan"

    if request_type == "recipe_search":
        return "recipe_search"

    return "general_chat_response"


def extract_weight_plan_values(weight_plan_result) -> tuple[str | None, float | None]:
    text = str(weight_plan_result)

    goal = None
    calorie_target = None

    goal_match = re.search(r"Goal:\s*([A-Za-z ]+)", text, flags=re.IGNORECASE)
    if goal_match:
        goal = goal_match.group(1).strip()

    calorie_match = re.search(
        r"Target Calories:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if calorie_match:
        calorie_target = float(calorie_match.group(1))

    return goal, calorie_target


def collect_plan_context(state: StringCheeseState) -> StringCheeseState:
    user_input = state.get("user_input", "")
    warnings = list(state.get("warnings", []))

    streamlit_input = state.get("user_input_from_streamlit", {}) or {}

    weight_plan_result = (
        streamlit_input.get("weight_plan_result")
        or state.get("weight_plan_result")
    )

    calorie_counter_result = (
        streamlit_input.get("calorie_counter_result")
        or state.get("calorie_counter_result")
    )

    goal = streamlit_input.get("goal")
    calorie_target = streamlit_input.get("calories_target")
    dietary_preference = streamlit_input.get("dietary_preference")
    allergies = streamlit_input.get("allergies")
    days = streamlit_input.get("days")

    extra_context = ""

    if weight_plan_result:
        extra_context += f"\n\nWeight Planning output:\n{weight_plan_result}"

    if calorie_counter_result:
        extra_context += f"\n\nCalorie Counter output:\n{calorie_counter_result}"

    combined_input = user_input + extra_context

    context = extract_dietary_context(combined_input)

    # Extract values from Weight Planner
    weight_goal = None
    weight_calorie_target = None

    if weight_plan_result:
        weight_goal, weight_calorie_target = extract_weight_plan_values(weight_plan_result)

    # FINAL VALUES
    final_goal = goal or weight_goal or context.goal
    final_calorie_target = calorie_target or weight_calorie_target or context.calorie_target
    final_dietary_preference = dietary_preference or context.dietary_preference
    final_allergies = allergies or context.allergies
    final_days = days or context.plan_days

    warnings.extend(
        detect_plan_warnings(
            user_input=combined_input,
            calorie_target=final_calorie_target,
        )
    )

    return {
        **state,
        "user_input": combined_input,
        "goal": final_goal,
        "calorie_target": final_calorie_target,
        "dietary_preference": final_dietary_preference,
        "allergies": final_allergies,
        "plan_days": final_days,
        "weight_plan_result": weight_plan_result,
        "calorie_counter_result": calorie_counter_result,
        "warnings": warnings,
    }


def generate_plan(state: StringCheeseState) -> StringCheeseState:
    """Generate a dietary plan using the dietary_plan module."""
    warnings = list(state.get("warnings", []))

    if _llm is None:
        warnings.append("LLM not configured for dietary plan generation.")
        return {
            **state,
            "plan": None,
            "response": None,
            "warnings": warnings,
        }

    try:
        plan = generate_dietary_plan(
            llm=_llm,
            user_input=state.get("user_input", ""),
            goal=state.get("goal") or state.get("user_input"),
            calorie_target=state.get("calorie_target"),
            dietary_preference=state.get("dietary_preference"),
            allergies=state.get("allergies"),
            days=state.get("plan_days") or 3,
        )

        if not plan.strip():
            warnings.append("Dietary plan generation returned an empty response.")
            return {
                **state,
                "plan": None,
                "response": None,
                "warnings": warnings,
            }

        return {
            **state,
            "plan": plan,
            "response": plan,
            "warnings": warnings,
        }

    except Exception as e:
        warnings.append(f"Plan generation failed: {e}")
        return {
            **state,
            "plan": None,
            "response": None,
            "warnings": warnings,
        }


def collect_recipe_context(state: StringCheeseState) -> StringCheeseState:
    """Use structured Streamlit input first, then fall back to parsing text."""
    streamlit_input = state.get("user_input_from_streamlit", {}) or {}
    user_input = state.get("user_input", "")

    ingredients = streamlit_input.get("ingredients") or []
    extra_details = streamlit_input.get("extra_details")

    if ingredients:
        return {
            **state,
            "ingredients": ingredients,
            "extra_recipe_details": extra_details,
        }

    # fallback text parsing
    lines = [line.strip() for line in user_input.splitlines() if line.strip()]

    for line in lines:
        lower = line.lower()

        if lower.startswith("find recipes with "):
            ingredient_text = line[len("Find recipes with "):].strip().rstrip(".")
            ingredients = [
                item.strip()
                for item in re.split(r",| and ", ingredient_text)
                if item.strip()
            ]

        elif lower.startswith("extra details:"):
            extra_details = line.split(":", 1)[1].strip()

    return {
        **state,
        "ingredients": ingredients or None,
        "extra_recipe_details": extra_details,
    }


def extract_recipe_matches(text: str) -> list[str]:
    """Extract recipe titles from retrieved text."""
    recipes = []

    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        if (
            len(line) > 3
            and len(line) < 80
            and not line.lower().startswith(("ingredients", "instructions", "steps"))
            and not ":" in line
        ):
            recipes.append(line)

    return list(dict.fromkeys(recipes))[:5]  # unique + top 5


def search_recipes(state: StringCheeseState) -> StringCheeseState:
    """Search recipe/cookbook documents using the existing search_docs tool and strictly filter by requested ingredients."""
    warnings = list(state.get("warnings", []))

    user_input = state.get("user_input", "")
    ingredients = state.get("ingredients") or []

    extra_details = state.get("extra_recipe_details")

    query = user_input
    if ingredients:
        query = f"recipe with {', '.join(ingredients)}"

    if extra_details:
        query += f" {extra_details}"

    try:
        result = search_docs.invoke({"query": query})

        # If no ingredients were given, return normal result
        if not ingredients:
            return {
                **state,
                "retrieved_context": result,
                "response": result,
                "recipes": None,
                "warnings": warnings,
            }

        result_text = str(result)
        lower_result = result_text.lower()

        missing_ingredients = [
            ingredient
            for ingredient in ingredients
            if ingredient.lower() not in lower_result
        ]

        if missing_ingredients:
            warnings.append(
                "Some search results may not contain the requested ingredient(s): "
                + ", ".join(missing_ingredients)
            )

        # Split retrieved chunks if your retriever uses this separator
        sections = [
            section.strip()
            for section in result_text.split("\n\n---\n\n")
            if section.strip()
        ]

        filtered_sections = []

        def score_section(section: str) -> int:
            lower = section.lower()
            score = 0

            for ingredient in ingredients:
                ing = ingredient.lower()

                if ing in lower:
                    score += 5

                if f"_{ing}_" in lower or f" {ing} " in lower:
                    score += 10

            if extra_details and extra_details.lower() in lower:
                score += 3

            return score

        for section in sections:
            section_lower = section.lower()
            if all(ingredient.lower() in section_lower for ingredient in ingredients):
                filtered_sections.append(section)

        filtered_sections.sort(key=score_section, reverse=True)
        filtered_sections = filtered_sections[:3]

        if not filtered_sections:
            return {
                **state,
                "retrieved_context": None,
                "response": (
                    "No recipe matches were found that strictly contain: "
                    + ", ".join(ingredients)
                ),
                "recipes": None,
                "warnings": warnings,
            }

        filtered_result = "\n\n---\n\n".join(filtered_sections)

        return {
            **state,
            "retrieved_context": filtered_result,
            "response": filtered_result,
            "recipes": None,
            "warnings": warnings,
        }

    except Exception as e:
        warnings.append(f"Recipe search failed: {e}")
        return {
            **state,
            "retrieved_context": None,
            "response": None,
            "warnings": warnings,
        }
    

def general_chat_response(state: StringCheeseState) -> StringCheeseState:
    """Handle general dietary assistant responses."""
    warnings = list(state.get("warnings", []))
    user_query = state.get("user_input", "")

    streamlit_input = state.get("user_input_from_streamlit", {}) or {}

    plans = {
        "weight_plan": (
            streamlit_input.get("weight_plan_result")
            or streamlit_input.get("weightloss_plan")
            or state.get("weight_plan_result")
            or state.get("weightloss_plan")
        ),
        "calorie_counter": (
            streamlit_input.get("calorie_counter_result")
            or streamlit_input.get("calorie_counter_output")
            or state.get("calorie_counter_result")
        ),
        "macro_tracker": (
            streamlit_input.get("macro_tracker_result")
            or streamlit_input.get("macro_tracker_output")
            or state.get("macro_result")
        ),
        "dietary_plan": (
            streamlit_input.get("dietary_output")
            or state.get("plan")
        ),
    }

    if _llm is None:
        warnings.append("LLM not configured for general responses.")
        return {
            **state,
            "response": None,
            "warnings": warnings,
        }

    prompt = f"""You are String Cheese, a dietary planning and recipe assistant.

    Answer any questions the user has regarding their plans. Keep the answer practical, clear, and concise.
    Make it clear which information is based off user's current data and which is from your general knowledge.
    If you take information from the following plans, user weightloss plan, calorie counter output, or dietary plan make it clear
    If relevant to the current query display information from answers to previous queries but keep it very concise.

    User request: {user_query}
    Plans: {plans}


    """
    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()

        return {
            **state,
            "response": str(content).strip(),
            "warnings": warnings,
        }
    except Exception as e:
        warnings.append(f"General response failed: {e}")
        return {
            **state,
            "response": None,
            "warnings": warnings,
        }


def format_response(state: StringCheeseState) -> StringCheeseState:
    """Ensure a final response field exists for downstream display."""
    response = state.get("response")

    if response:
        return state

    if state.get("plan"):
        return {
            **state,
            "response": state["plan"],
        }

    if state.get("recipes"):
        return {
            **state,
            "response": "\n".join(f"- {recipe}" for recipe in state["recipes"]),
        }
    
    if state.get("weightloss_plan"):
        return {
            **state,
            "response": state["weightloss_plan"],
        }

    if state.get("retrieved_context"):
        return {
            **state,
            "response": state["retrieved_context"],
        }

    warnings = list(state.get("warnings", []))
    warnings.append("No final response was produced.")

    return {
        **state,
        "response": None,
        "warnings": warnings,
    }


def weightloss_plan(state: StringCheeseState) -> StringCheeseState:
    """Use provided user input to Generate weightloss plan"""
    warnings = list(state.get("warnings", []))

    streamlit_input = state.get("user_input_from_streamlit", {})

    user_input = streamlit_input

    try:
        result = weight_planning.invoke(user_input)
        return {
            **state,
            "weight_plan_result": result,
            "weightloss_plan": result,
            "weightloss_input": user_input,
            "response": result,
            "warnings": warnings,
        }
    except Exception as e:
        warnings.append(f"Weight Planning failed: {e}")
        return {
            **state,
            "retrieved_context": None,
            "response": None,
            "warnings": warnings,
        }


def calculate_calories(state: StringCheeseState) -> StringCheeseState:
    """Count calories and macros based on structured user input."""
    warnings = list(state.get("warnings", []))

    streamlit_input = state.get("user_input_from_streamlit", {})

    user_input = streamlit_input

    try:
        food_items = user_input.get("food_items", [])

        result = calorie_counter.invoke({
            "food_items": food_items
        })

        return {
            **state,
            "calorie_counter_result": result,
            "response": result,
            "warnings": warnings,
        }

    except Exception as e:
        warnings.append(f"Calorie Counting failed: {e}")
        return {
            **state,
            "response": None,
            "warnings": warnings,
        }

def calculate_macros(state: StringCheeseState) -> StringCheeseState:
    """Calculate macro recommendations based on target calories and goal."""
    warnings = list(state.get("warnings", []))

    user_input = state.get("user_input_from_streamlit", {}) or {}

    try:
        result = macro_tracker.invoke(user_input)

        return {
            **state,
            "macro_result": result,
            "response": result,
            "warnings": warnings,
        }

    except Exception as e:
        warnings.append(f"Macro Calculation failed: {e}")
        return {
            **state,
            "response": None,
            "warnings": warnings,
        }