"""Tool definitions for the String Cheese application.

This module defines tools that can be used by the LLM via LangGraph tool
calling. Tools are defined using LangChain's @tool decorator for
model-agnostic compatibility.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from .config import Config

# Module-level config reference for search_docs tool
_search_config: Config | None = None


def set_search_config(cfg: Config) -> None:
    """Set config for the search_docs tool. Called at startup."""
    global _search_config
    _search_config = cfg


SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
}

SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "pi": math.pi,
    "e": math.e,
}


class SafeEvalError(Exception):
    """Raised when expression evaluation fails or is unsafe."""
    pass


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeEvalError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise SafeEvalError(f"Unknown variable: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise SafeEvalError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise SafeEvalError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in SAFE_OPERATORS:
                raise SafeEvalError(f"Unsupported comparison operator: {op_type.__name__}")
            right = _safe_eval_node(comparator)
            if not SAFE_OPERATORS[op_type](left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SafeEvalError("Only simple function calls are supported")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise SafeEvalError(f"Unknown function: {func_name}")
        func = SAFE_FUNCTIONS[func_name]
        if not callable(func):
            raise SafeEvalError(f"{func_name} is not a function")
        args = [_safe_eval_node(arg) for arg in node.args]
        return func(*args)

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval_node(elt) for elt in node.elts]

    raise SafeEvalError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> float | int | bool:
    """Safely evaluate a mathematical expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Invalid syntax: {e.msg}") from e

    return _safe_eval_node(tree.body)


@tool
def python_calc(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Use this tool for arithmetic or other math-based calculations.
    """
    try:
        result = safe_eval(expression)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except SafeEvalError as e:
        return f"Error: {e}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Result too large"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Unexpected error - {type(e).__name__}: {e}"


@tool
def search_docs(query: str) -> str:
    """Search dietary, recipe, or cookbook documents for relevant information.

    Use this tool to search documents stored in the configured resources
    directory, such as recipe files, cookbooks, or dietary reference text.

    Args:
        query: The search query describing what information is needed.

    Returns:
        Relevant document passages with source info, or a message if none found.
    """
    from .retriever import get_retriever

    if _search_config is None:
        return "Error: Search not configured."

    retriever = get_retriever(_search_config)
    if retriever is None:
        return "No documents available. The resources directory may be empty."

    results = retriever.invoke(query)
    if not results:
        return "No relevant documents found."

    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in results
    )

@tool
def weight_planning(
    sex: str,
    weight_lbs: float,
    height_in: float,
    age: int,
    activity_level: str,
    goal: str
) -> str:
    """
    Estimate calorie needs and provide a weight plan using RMR.

    Only use this tool when prompted about creating a weight plan. Furthermore, if the user does not provide the specific values to required arguments please ask the user for said information. 

    Uses custom RMR equations:
    - Male: 4.38*weight + 14.55*height - 5.08*age + 260
    - Female: 3.35*weight + 15.42*height - 2.31*age + 43

    Args:
        sex: "male" or "female"
        weight_lbs: Weight in pounds
        height_in: Height in inches
        age: Age in years
        activity_level: One of:
            - "sedentary"
            - "light"
            - "moderate"
            - "active"
            - "very_active"
        goal: "lose", "maintain", or "gain"

    Returns:
        RMR, maintenance calories, and recommended intake. And a small context paragraph relating to each result (RMR, maintenance calories, and recommended intake).
    """

    errors = ""
    
    # Input validation
    if not (90 <= weight_lbs <= 650):
        errors += "Error: Weight must be between 90 and 650 pounds. \n\n"
    if not (48 <= height_in <= 108):
        errors += "Error: Height must be between 48 and 108 inches. \n\n"
    if not (14 <= age <= 100):
        errors += "Error: Age must be between 14 and 100 years old. \n\n"
    
    if errors != "":
        return errors


    sex = sex.lower()
    goal = goal.lower()
    activity_level = activity_level.lower()

    # Calculate RMR
    if sex == "male":
        rmr = 4.38 * weight_lbs + 14.55 * height_in - 5.08 * age + 260
    elif sex == "female":
        rmr = 3.35 * weight_lbs + 15.42 * height_in - 2.31 * age + 43
    else:
        return "Error: Sex must be 'male' or 'female'."

    # Activity multipliers
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    maintenance_calories = rmr * activity_multipliers[activity_level]

    # Adjust based on goal
    if goal == "lose":
        target_calories = maintenance_calories - 500
        recommendation = "Caloric deficit for weight loss (~1 lb/week)."
    elif goal == "gain":
        target_calories = maintenance_calories + 500
        recommendation = "Caloric surplus for weight gain (~1 lb/week)."
    elif goal == "maintain":
        target_calories = maintenance_calories
        recommendation = "Maintain current weight."
    else:
        return "Error: Goal must be 'lose', 'maintain', or 'gain'."

    return (
        f"Weight Planning Summary: \n\n"
        f"RMR: {rmr:.2f} kcal/day  \n\n"
        f"Maintenance Calories: {maintenance_calories:.2f} kcal/day  \n\n"
        f"Target Calories: {target_calories:.2f} kcal/day  \n\n"
        f"Goal: {goal.capitalize()}  \n\n"
        f"Recommendation: {recommendation}"
    )

@tool
def calorie_counter(food_items: list[dict]) -> str:
    """
    Calculate total calories and macronutrients from food items.

    Use the information collected from provided recipes in documents or from user input to calculate the total calorie and macro intake for the day. 
    Only use this tool when prompted about counting calories or tracking macros. 
    Furthermore, if the user does not provide the specific values to required arguments please ask the user for said information.

    Args:
        food_items: A list of dictionaries, each containing:
            - "name": Food name (string)
            - "calories": kcal (int/float)
            - "protein": grams (int/float)
            - "carbs": grams (int/float)
            - "fats": grams (int/float)

    Example:
        [
            {"name": "Chicken Breast", "calories": 250, "protein": 40, "carbs": 0, "fats": 5},
            {"name": "Rice", "calories": 200, "protein": 4, "carbs": 45, "fats": 1}
        ]

    Returns:
        Total calories and macro breakdown.
    """

    if not food_items:
        return "Error: No food items provided."

    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fats = 0

    breakdown = []

    try:
        for item in food_items:
            name = item.get("name", "Unknown")

            calories = item.get("calories")
            protein = item.get("protein")
            carbs = item.get("carbs")
            fats = item.get("fats")

            # Validate inputs
            for field_name, value in {
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fats": fats
            }.items():
                if value is None or not isinstance(value, (int, float)):
                    return f"Error: Invalid {field_name} value for '{name}'."
                if value < 0:
                    return f"Error: {field_name} cannot be negative for '{name}'."

            total_calories += calories
            total_protein += protein
            total_carbs += carbs
            total_fats += fats

            breakdown.append(
                f"{name}: {calories} kcal | P:{protein}g C:{carbs}g F:{fats}g"
            )

        return (
            "Nutrition Summary:\n"
            + "\n".join(breakdown)
            + "\n\nTotals:\n"
            + f"Calories: {total_calories:.2f} kcal\n"
            + f"Protein: {total_protein:.2f} g\n"
            + f"Carbs: {total_carbs:.2f} g\n"
            + f"Fats: {total_fats:.2f} g"
        )

    except Exception as e:
        return f"Error: Unexpected issue - {type(e).__name__}: {e}"
    
@tool
def macro_tracker(
    target_calories: float,
    goal: str
) -> str:
    """
    Calculate daily macronutrient targets based on calorie goal.

    Use this tool alongside the weight_planning tool and calorie_counter tool determine how to split target calories into protein, carbs, and fats based on the user's goal (lose, maintain, gain).
    Only use this tool when prompted about tracking macros or creating a meal plan. 
    Furthermore, if the user does not provide the specific values to required arguments please ask the user for said information.

    Args:
        target_calories: Daily calorie target
        goal: "lose", "maintain", or "gain"

    Returns:
        Macro breakdown in grams and calories.
    """

    if not (1200 <= target_calories <= 12000):
        return "Error: Target calories must be above 1200 and below 12000 for a healthy diet."


    goal = goal.lower()

    # Macro splits
    if goal == "lose":
        protein_pct = 0.40
        carbs_pct = 0.30
        fats_pct = 0.30
    elif goal == "maintain":
        protein_pct = 0.30
        carbs_pct = 0.40
        fats_pct = 0.30
    elif goal == "gain":
        protein_pct = 0.25
        carbs_pct = 0.50
        fats_pct = 0.25
    else:
        return "Error: Goal must be 'lose', 'maintain', or 'gain'."

    # Calories per macro
    protein_cal = target_calories * protein_pct
    carbs_cal = target_calories * carbs_pct
    fats_cal = target_calories * fats_pct

    # Convert to grams
    protein_g = protein_cal / 4
    carbs_g = carbs_cal / 4
    fats_g = fats_cal / 9

    return (
        f"Macro Targets ({goal.capitalize()} Goal):\n\n"
        f"Calories: {target_calories:.2f} kcal\n\n"
        f"Protein: {protein_g:.2f} g ({protein_cal:.2f} kcal)\n\n"
        f"Carbs: {carbs_g:.2f} g ({carbs_cal:.2f} kcal)\n\n"
        f"Fats: {fats_g:.2f} g ({fats_cal:.2f} kcal)"
    )

