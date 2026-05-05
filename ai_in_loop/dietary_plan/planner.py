"""Dietary plan generation logic for String Cheese."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ai_in_loop.llm import get_text

from .prompts import (
    DIETARY_PLAN_SYSTEM_PROMPT,
    build_dietary_plan_prompt,
    build_plan_analysis_prompt,
    build_plan_revision_prompt,
)


LOW_CALORIE_THRESHOLD = 1200

MEDICAL_CONTEXT_TERMS = [
    "diabetes",
    "pregnant",
    "pregnancy",
    "kidney disease",
    "renal",
    "eating disorder",
    "anorexia",
    "bulimia",
]

EXTREME_DIET_TERMS = [
    "starve",
    "starvation",
    "purge",
    "under 800 calories",
    "800 calories",
    "extreme weight loss",
    "lose weight as fast as possible",
]


@dataclass
class DietaryPlanContext:
    goal: str | None = None
    calorie_target: int | None = None
    dietary_preference: str | None = None
    allergies: list[str] | None = None
    plan_days: int = 3


def extract_dietary_context(user_input: str) -> DietaryPlanContext:
    """Extract simple dietary planning fields from free text."""
    text = user_input.strip()
    lower = text.lower()

    calorie_target = None
    calorie_patterns = [
        r"\b(\d{3,4})\s*cal(?:ories)?\b",
        r"\b(\d{3,4})\s*kcal\b",
    ]
    for pattern in calorie_patterns:
        match = re.search(pattern, lower)
        if match:
            try:
                calorie_target = int(match.group(1))
                break
            except ValueError:
                calorie_target = None

    dietary_preference = None
    preference_options = [
        "vegetarian",
        "vegan",
        "high-protein",
        "high protein",
        "keto",
        "low-carb",
        "low carb",
        "gluten-free",
        "gluten free",
        "dairy-free",
        "dairy free",
        "pescatarian",
    ]
    for option in preference_options:
        if option in lower:
            dietary_preference = option.replace(" ", "-")
            break

    allergies = _extract_allergies(lower)
    goal = _extract_goal(text)

    days = 3
    day_match = re.search(r"\b(\d+)\s*-\s*days?\b|\b(\d+)\s*days?\b", lower)
    if day_match:
        try:
            days = int(day_match.group(1) or day_match.group(2))
            days = max(1, min(days, 7))
        except ValueError:
            days = 3

    return DietaryPlanContext(
        goal=goal,
        calorie_target=calorie_target,
        dietary_preference=dietary_preference,
        allergies=allergies,
        plan_days=days,
    )


def _extract_goal(text: str) -> str | None:
    lower = text.lower()

    known_goals = [
        "lose weight",
        "weight loss",
        "gain weight",
        "build muscle",
        "muscle gain",
        "eat healthier",
        "healthy eating",
        "maintenance",
    ]
    for goal in known_goals:
        if goal in lower:
            return goal

    goal_match = re.search(r"goal:\s*(.+)", text, re.IGNORECASE)
    if goal_match:
        return goal_match.group(1).strip()

    return text.strip() or None


def _normalize_allergy(item: str) -> str:
    value = item.strip().lower()

    if "dairy" in value or "milk" in value or "lactose" in value:
        return "dairy"
    if "peanut" in value:
        return "peanuts"
    if "tree nut" in value or value == "nuts":
        return "tree nuts"
    if "gluten" in value or "wheat" in value:
        return "gluten"

    return value


def _extract_allergies(lower: str) -> list[str] | None:
    patterns = [
        r"(?:allergic to|allergy to)\s+([a-zA-Z,\s-]+)",
        r"(?:allergies:)\s*([a-zA-Z,\s-]+)",
        r"(?:foods to avoid:)\s*([a-zA-Z,\s-]+)",
        r"(?:avoid)\s+([a-zA-Z,\s-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            raw = match.group(1)
            items = [
                item.strip(" .,-")
                for item in re.split(r",|and", raw)
                if item.strip(" .,-")
            ]
            cleaned = [_normalize_allergy(item) for item in items if len(item) > 1]
            cleaned = list(dict.fromkeys(cleaned))
            return cleaned or None

    return None


def detect_plan_warnings(
    *,
    user_input: str,
    calorie_target: int | None,
) -> list[str]:
    """Detect safety and product warnings for dietary plan generation."""
    warnings: list[str] = []
    lower = user_input.lower()

    if calorie_target is not None and calorie_target < LOW_CALORIE_THRESHOLD:
        warnings.append(
            "The requested calorie target is very low. This app should avoid overly restrictive plans."
        )

    if any(term in lower for term in MEDICAL_CONTEXT_TERMS):
        warnings.append(
            "The request may involve a medical context. Only general wellness guidance should be provided."
        )

    if any(term in lower for term in EXTREME_DIET_TERMS):
        warnings.append(
            "The request may involve unsafe or extreme dieting language."
        )

    return warnings


def generate_dietary_plan(
    *,
    llm: BaseChatModel,
    user_input: str,
    goal: str | None = None,
    calorie_target: int | None = None,
    dietary_preference: str | None = None,
    allergies: list[str] | None = None,
    days: int = 3,
) -> str:
    """Generate a dietary plan using the shared app LLM."""
    prompt = build_dietary_plan_prompt(
        user_input=user_input,
        goal=goal,
        calorie_target=calorie_target,
        dietary_preference=dietary_preference,
        allergies=allergies,
        days=days,
    )

    response = llm.invoke(
        [
            SystemMessage(content=DIETARY_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return get_text(response)


def analyze_dietary_plan(
    *,
    llm: BaseChatModel,
    plan_text: str,
    user_goal: str | None = None,
) -> str:
    """Analyze an existing dietary plan."""
    prompt = build_plan_analysis_prompt(plan_text=plan_text, user_goal=user_goal)
    response = llm.invoke(
        [
            SystemMessage(content=DIETARY_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return get_text(response)


def revise_dietary_plan(
    *,
    llm: BaseChatModel,
    existing_plan: str,
    revision_request: str,
) -> str:
    """Revise an existing dietary plan."""
    prompt = build_plan_revision_prompt(
        existing_plan=existing_plan,
        revision_request=revision_request,
    )
    response = llm.invoke(
        [
            SystemMessage(content=DIETARY_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return get_text(response)