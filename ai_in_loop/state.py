"""State schema for the String Cheese workflow.

This module defines the StringCheeseState TypedDict that flows through the
LangGraph workflow. Each node reads from and writes to fields in this state.
"""

from typing import Literal, TypedDict


class StringCheeseState(TypedDict, total=False):
    user_input: str
    user_input_from_streamlit: dict #this should be the user input that is collected from radio buttons etc
    #so when workflow is run the user info only relevant is here

    request_type: Literal[ "weightloss_plan", "calculate_calories", "calculate_macros", "dietary_plan","recipe_search", "general_chat_response"] | None

    prompt: str | None

    goal: str | None
    calorie_target: int | None
    dietary_preference: str | None
    allergies: list[str] | None
    ingredients: list[str] | None
    plan_days: int | None

    retrieved_context: str | None
    plan: str | None
    recipes: list[str] | None
    response: str | None

    weightloss_input: dict | None
    weightloss_plan: str | None


    calorie_and_macro_output: str | None

    warnings: list[str]