"""Prompt builders for dietary plan generation."""

from __future__ import annotations

from typing import Iterable


DIETARY_PLAN_SYSTEM_PROMPT = """
You are String Cheese, a dietary planning assistant for a wellness app.

Your job is to help users create practical meal plans based on their goals,
preferences, allergies, calorie targets, and lifestyle constraints.

Safety rules:
- Do not provide medical diagnosis or treatment.
- Do not encourage starvation, purging, crash dieting, or extreme restriction.
- Do not recommend dangerously low calorie intakes.
- Respect allergies and dietary restrictions strictly.
- If the request includes a medical condition or high-risk context, give only
  general wellness guidance and encourage the user to consult a licensed
  healthcare professional or registered dietitian.

Style rules:
- Be practical, realistic, and concise.
- Prefer meals that are easy to understand and commonly available.
- When details are missing, make reasonable assumptions and say so briefly.
- Return clear sectioned text, not JSON.
""".strip()


def _format_list(items: Iterable[str] | None, fallback: str = "None provided") -> str:
    if not items:
        return fallback
    cleaned = [item.strip() for item in items if item and item.strip()]
    return ", ".join(cleaned) if cleaned else fallback


def build_dietary_plan_prompt(
    *,
    user_input: str,
    goal: str | None = None,
    calorie_target: int | None = None,
    dietary_preference: str | None = None,
    allergies: list[str] | None = None,
    days: int = 3,
) -> str:
    calorie_text = str(calorie_target) if calorie_target is not None else "Not specified"
    preference_text = dietary_preference or "None specified"
    allergy_text = _format_list(allergies)

    day_sections = []
    for day_num in range(1, days + 1):
        day_sections.append(
            f"""{day_num + 2}. Day {day_num}
   - Breakfast
   - Lunch
   - Dinner
   - Snack"""
        )

    output_format = "\n".join(day_sections)

    return f"""
Create a {days}-day dietary plan for the following user request.

Original user request:
{user_input}

Known details:
- Goal: {goal or "Not specified"}
- Daily calorie target: {calorie_text}
- Dietary preference: {preference_text}
- Allergies / foods to avoid: {allergy_text}

Requirements:
- Include breakfast, lunch, dinner, and 1 snack per day.
- Keep the meals practical and realistic.
- Make the plan align with the user's goal and calorie target when possible.
- Strictly avoid listed allergens and avoided foods.
- If an allergen is listed, do not include it in any form, even as a snack, topping, substitute, garnish, sauce, or optional ingredient.
- For dairy allergies, avoid all dairy ingredients including milk, cheese, yogurt, greek yogurt, cottage cheese, cream, butter, whey, and casein.
- Do not give medical advice.
- If details are missing, make reasonable assumptions.

Output format:
1. A short overview
2. A "Daily Targets" section
{output_format}
{days + 3}. A short "Why this fits" section
{days + 4}. A short "Suggested swaps" section
""".strip()


def build_plan_analysis_prompt(plan_text: str, user_goal: str | None = None) -> str:
    """Build a prompt for analyzing an uploaded or pasted dietary plan."""
    return f"""
Analyze the following dietary plan.

User goal:
{user_goal or "Not specified"}

Dietary plan to analyze:
{plan_text}

Provide:
1. A short summary of the plan
2. Whether it seems aligned with the stated goal
3. Any obvious strengths
4. Any obvious limitations
5. Practical suggestions for improvement

Do not provide medical diagnosis or treatment.
""".strip()


def build_plan_revision_prompt(existing_plan: str, revision_request: str) -> str:
    """Build a prompt for revising an existing dietary plan."""
    return f"""
Revise the dietary plan below based on the user's request.

Existing plan:
{existing_plan}

Revision request:
{revision_request}

Keep the same general structure, but apply the requested changes.
Be practical, concise, and safety-aware.
""".strip()