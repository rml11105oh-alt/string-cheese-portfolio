from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ai_in_loop.config import Config
from ai_in_loop.graph import build_app, get_graph_image

from pathlib import Path
from ai_in_loop.retriever import reset_retriever

from typing import Any
import re


# ── Cached workflow ──────────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    cfg = Config.from_env()
    return build_app(cfg), cfg


def run_workflow(user_input: str, user_input_from_streamlit:dict, request_type: str) -> dict:
    pipeline, _ = get_pipeline()
    initial_state = {
        "user_input": user_input,
        "user_input_from_streamlit": user_input_from_streamlit,
        "request_type": request_type,
        "warnings": [],
    }
    return pipeline.stream(initial_state, stream_mode="updates")
#    return pipeline.invoke(initial_state)


def collect_stream_result(user_input: str) -> dict:
    """Run the workflow stream and merge node outputs into one final result dict."""
    result = {}

    for chunk in run_workflow(user_input):
        for _, node_output in chunk.items():
            if isinstance(node_output, dict):
                result.update(node_output)

    return result


def render_result(result: dict) -> None:

    response = result.get("response")
    plan = result.get("plan")
    recipes = result.get("recipes")
    context = result.get("retrieved_context")
    warnings = result.get("warnings", [])

    if plan:
        st.subheader("Dietary Plan")
        st.markdown(plan)

    if recipes:
        st.subheader("Recipe Suggestions")
        for recipe in recipes:
            st.markdown(f"- {recipe}")

    if context:
        with st.expander("Retrieved Document Context", expanded=False):
            st.text(context)

    if response and response != plan:
        st.subheader("Response")
        st.markdown(response)

    if warnings:
        with st.expander("Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)


def format_weightloss_plan_with_llm(weightloss_plan: str) -> str:
    """Turn input into a fully formulated weightloss plan"""
    _, cfg = get_pipeline()

    from ai_in_loop.llm import get_llm, get_text
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(cfg)

    prompt = f"""Format the following weightloss plan: {weightloss_plan}

            Create a clean card with each of the newlines in the weightloss plan as a bullet point

            Create a new paragraph to expand upon what each bullet point means
            
            """.strip()
    
    response = llm.invoke(
        [
            SystemMessage(
                content="Format a weightloss plan"
            ),
            HumanMessage(content=prompt),
        ]
    )

    return get_text(response)

def save_uploaded_pdfs(uploaded_files) -> list[str]:
    """Save uploaded PDF files into resources/uploaded and reset retriever."""
    _, cfg = get_pipeline()
    resources_dir = Path(cfg.resources_dir) / "uploaded"
    resources_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue

        file_name = Path(uploaded_file.name).name
        file_path = resources_dir / file_name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(str(file_path))

    if saved_files:
        reset_retriever()

    return saved_files

def list_recipe_files() -> list[str]:
    """List available recipe/cookbook PDFs in the resources directory."""
    _, cfg = get_pipeline()
    resources_dir = Path(cfg.resources_dir)

    if not resources_dir.exists():
        return []

    pdfs = sorted(path.name for path in resources_dir.rglob("*.pdf"))
    return pdfs


def run_with_status(user_input: str, user_input_from_streamlit: dict, label: str, request_type: str) -> dict[str, Any]:
    """Run workflow with streamed status updates and merge node outputs."""
    result: dict[str, Any] = {}

    with st.status(label, expanded=True) as status:
        for chunk in run_workflow(user_input, user_input_from_streamlit, request_type):
            for node_name, node_output in chunk.items():
                st.write(f"Completed: **{node_name}**")
                if isinstance(node_output, dict):
                    result.update(node_output)

        status.update(label="Pipeline complete!", state="complete")

    return result


def format_recipe_match_with_llm(match: dict[str, str]) -> str:
    """Turn a raw retrieved recipe excerpt into a readable recipe card."""
    _, cfg = get_pipeline()

    from ai_in_loop.llm import get_llm, get_text
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(cfg)

    prompt = f"""You are formatting a retrieved recipe excerpt for a user.
           
            Source file:
            {match['source']}

            Recipe excerpt:
            {match['content']}

            Create a clean recipe card with exactly these sections:
            1. Recipe Title
            2. Short Overview
            3. Ingredients
            4. Directions
            5. Notes

            Rules:
            - If a recipe title is clearly present, use it exactly.
            - If no recipe title is clearly present, abstract a title from the PDF filename.
            - For Ingredients, copy ingredient lines exactly as written when they are present.
            - Preserve ingredient quantities and measurements from the 'Ingredients' section of the excerpt.
            - For Directions, copy or closely preserve the listed steps from the excerpt section 'Directions'.
            - Ignore FAQ sections, nutrition facts, review text, metadata, and unrelated material.
            - Do not invent missing ingredients, quantities, or steps.
            - Make the final output neat and readable in markdown.
            """.strip()

    response = llm.invoke(
        [
            SystemMessage(
                content="You format recipe excerpts into accurate recipe cards and preserve exact ingredient lines when available."
            ),
            HumanMessage(content=prompt),
        ]
    )
    return get_text(response)


def extract_recipe_matches(result: dict[str, object]) -> list[dict[str, str]]:
    """Group retrieved chunks by source and combine the most recipe-like excerpts."""
    context = result.get("retrieved_context") or result.get("response") or ""
    if not isinstance(context, str) or not context.strip():
        return []

    sections = [section.strip() for section in context.split("\n\n---\n\n") if section.strip()]
    grouped: dict[str, list[str]] = {}

    for section in sections:
        source_match = re.search(r"Source:\s*(.+)", section)
        content_match = re.search(r"Content:\s*(.+)", section, flags=re.DOTALL)

        source = source_match.group(1).strip() if source_match else "unknown"
        content = content_match.group(1).strip() if content_match else section.strip()

        grouped.setdefault(source, []).append(content)

    def score_chunk(text: str) -> int:
        lower = text.lower()
        score = 0

        if "ingredients" in lower:
            score += 5
        if "directions" in lower or "instructions" in lower or "step 1" in lower:
            score += 5
        if "frequently asked questions" in lower:
            score -= 6
        if "nutrition facts" in lower:
            score -= 5
        if "reviewed by dietitian" in lower:
            score -= 2
        if "updated on" in lower:
            score -= 2

        return score

    matches = []
    for i, (source, chunks) in enumerate(grouped.items(), start=1):
        # sort chunks so recipe-like ones come first
        ranked_chunks = sorted(chunks, key=score_chunk, reverse=True)

        # keep the best 2 chunks per source
        best_chunks = ranked_chunks[:2]
        combined_content = "\n\n".join(best_chunks)

        title = Path(source).stem.replace("_", " ").replace("-", " ").title()

        matches.append(
            {
                "title": title or f"Recipe Match {i}",
                "source": Path(source).name,
                "content": combined_content,
            }
        )

    # sort matches by how recipe-like their combined content is
    matches.sort(key=lambda m: score_chunk(m["content"]), reverse=True)

    return matches[:3]


st.set_page_config(
    page_title="String Cheese",
    page_icon="🧀",
    layout="wide",
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap" rel="stylesheet">
    <style>
    h1 {
        font-size: 67px !important;
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True,
)

st.markdown("""
    <style>
    div[data-testid="stAlert"] {
        background-color: #fffacd;
        border-radius: 10px;
    }
            
    div[data-testid="stAlert"] * {
    color: #676767 !important;
    }
    </style>
    """, unsafe_allow_html=True,
)

st.title("String Cheese 🧀")
st.caption("Dietary planning, recipe search, and general nutrition help.")

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(
        "String Cheese is a dietary assistant application that helps users "
        "generate meal plans, search recipes, search uploaded/reference documents, "
        "and ask general dietary questions."
    )
    st.divider()
    st.subheader("Upload Recipe / Cookbook PDFs")

    uploaded_pdfs = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Save Uploaded PDFs"):
        if not uploaded_pdfs:
            st.warning("Please choose at least one PDF file.")
        else:
            saved_paths = save_uploaded_pdfs(uploaded_pdfs)
            if saved_paths:
                st.success(f"Saved {len(saved_paths)} PDF file(s). Document search is now updated.")
                for path in saved_paths:
                    st.caption(f"Saved: {Path(path).name}")
            else:
                st.warning("No files were saved.")

    show_graph = st.checkbox("Show workflow graph", value=False)


# ── Optional graph display ───────────────────────────────────────────
if show_graph:
    _, cfg = get_pipeline()
    image_bytes = get_graph_image(cfg)
    if image_bytes:
        st.image(image_bytes, caption="String Cheese workflow graph")
    else:
        st.info("Workflow graph could not be generated. Install visualization support if needed.")


# ── Main feature tabs ────────────────────────────────────────────────
tab1, tab2 = st.tabs(
    ["Dietary Plan", "Recipe Library"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with tab1:
    if "weight_plan_result" not in st.session_state:
        st.session_state["weight_plan_result"] = None

    if "weight_plan_input" not in st.session_state:
        st.session_state["weight_plan_input"] = None

    if "calorie_counter_result" not in st.session_state:
        st.session_state["calorie_counter_result"] = None

    if "calorie_counter_input" not in st.session_state:
        st.session_state["calorie_counter_input"] = None
    
    if "macro_tracker_result" not in st.session_state:
        st.session_state["macro_tracker_result"] = None

    if "macro_tracker_input" not in st.session_state:
        st.session_state["macro_tracker_input"] = None
    
    st.header("Weight Planning")

    st.info(
        "🧀 **Weight Planning** helps estimate calorie needs and provide a weight plan using RMR"
    )
  
    #this is so the summary shows even after the next workflow is run
    if 'weightloss_output' not in st.session_state:
        st.session_state['weightloss_output'] = {} #if this section of the workflow has not run yet there will be no output associated with it and it will not render

    age = st.number_input("Age", value=None, placeholder=0,  min_value=0, step=10, )
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True,)
    gender = gender.lower()
    weight = st.number_input("Weight in lbs", value=None, placeholder=0, min_value=0, step=10)
    height = st.number_input("Height in inches",value=None, placeholder=0, min_value=0, step=10,)
    activity_level = st.radio(
        "Activity Level",
        ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
        horizontal=True,
    )
    activity_level = activity_level.lower()
    if activity_level == "very active":
        activity_level = "very_active"
    goal = st.radio(
        "Goal",
        ["Lose Weight", "Maintain Weight", "Gain Weight"],
            horizontal=True,
    )
    #setting it equal to expected input
    if goal == "Lose Weight":
        goal = "lose"
    if goal == "Maintain Weight":
        goal = "maintain"
    if goal == "Gain Weight":
        goal = "gain"

    weightloss_input = {"sex": gender, "weight_lbs": weight, "height_in": height, "age": age, "activity_level": activity_level, "goal": goal}

    if st.button("Generate Plan", key="weightloss_plan"):
        if any(v == None for v in (goal, height, weight, age)):
            st.warning("Please enter your information to generate your plan.")
            st.session_state['weightloss_output'] = {}

        else:
            prompt_lines = [
                f"Create a dietary plan using the dietary plan tool with the parameters for age, gender, weight, activity_level, and goal",
                f"age: {age}",
                f"gender: {gender}",
                f"weight: {weight}",
                f"activity_level: {activity_level}",
                f"goal: {goal}",

            ]

            prompt = "\n".join(line for line in prompt_lines if line)

            with st.spinner("Generating weightloss plan..."):
                result = run_with_status(prompt, weightloss_input, "Running dietary plan pipeline...", "weightloss_plan")
                st.session_state['weightloss_output'] = result
                #reset the session state for the chat here, so it only knows the previous queries for this specific time the plan is run
                
                st.session_state["weight_plan_result"] = result
                st.session_state["weight_plan_input"] = weightloss_input

            #result = format_weightloss_plan_with_llm(result) #if we want special formattings

    if st.session_state['weightloss_output'] != {}:
        render_result(st.session_state['weightloss_output'])
                    
        st.divider()

        st.write("Do you have any questions?")
        chat_prompt = st.text_input("Ask a question", key="weightloss_chat_input")

        if st.button("Ask", key="weightloss_questions"):
            if not chat_prompt.strip():
                st.warning("Please enter a question.")
            else:
                #Allow the llm to access the weightloss plan, sending it as the user input and state
                result = run_with_status(chat_prompt, {"weightloss_plan": st.session_state['weightloss_output']}, "Running chat pipeline...", "general_chat_response")
                
                st.session_state["messages"].append(result)

                render_result(result)

            
    st.header("Calorie Counter")


    st.info(
        "🧀 **Calorie Counter** calculates total calories and macronutrients from food items"
    )
  
    # Show summary even after the next workflow is run
    if 'calorie_output' not in st.session_state:
        st.session_state['calorie_output'] = {} 

    st.text("This tool will help you count your calories for the day. If you know the exact information for each food you have eaten select the calorie counter. Otherwise, select macro counter to calculate macro partitioning recommendations based on target calorie output from the Weight Planner")
    
 
    tool_to_use = st.radio("Please choose input type", ["Macro Tracker", "Calorie Counter"], horizontal=True,)

    if(tool_to_use=="Macro Tracker"):    

        target_calories = st.number_input("Target Calories", step=1, min_value=0, placeholder=1800, value = None)
 
        goal_input = st.radio("Goal", ["Lose", "Maintain", "Gain"], horizontal=True,)
        goal_input = goal_input.lower()

        macros_input = {"target_calories": target_calories, "goal": goal_input}


        if st.button("Count Macros", key="macro_count"):
            if target_calories == 0:
                st.warning("Please enter your target calories.")
            else:
                prompt_lines = [
                    f"Count macros based on user's target calories and goal",
                ]
                prompt = "\n".join(line for line in prompt_lines if line)
                    
                with st.spinner("Counting Macros..."):
                    result = run_with_status(prompt, macros_input, "Running recipe search pipeline...", "calculate_macros")
                    st.session_state['calorie_output'] = result
                    
                    st.session_state["macro_tracker_result"] = result
                    st.session_state["macro_tracker_input"] = macros_input

        if st.session_state['calorie_output'] != {}:
            render_result(st.session_state['calorie_output'])
                        
            st.divider()
            
            st.write("Do you have any questions?")
            chat_prompt = st.text_input("Ask a question")

            if st.button("Ask", key="calorie_questions"):
                if not chat_prompt.strip():
                    st.warning("Please enter a question.")
                else:
                    result = run_with_status(chat_prompt, {"calorie_counter_output": st.session_state['calorie_output'], "weightloss_plan": st.session_state['weightloss_output']}, "Running chat pipeline...", "general_chat_response")
                    
                    st.session_state["messages"].append(result)

                    render_result(result)


    else:
        st.write("Calorie Counter!")
        col1, col2, col3, col4, col5 = st.columns(5)

   
           # 1. Initialize the list of inputs in session state
        if 'rows' not in st.session_state:
            st.session_state.rows = [0]  # Start with one input line

        def add_row():
            # 2. Callback function to add a new index
            st.session_state.rows.append(len(st.session_state.rows))

       # food_items = {"food_items" : []}
        food_items_list = []
        # 4. Loop to render the inputs based on the state
        for i in st.session_state.rows:
            with col1:
                name = st.text_input("Name", key= f"name{i}")

            with col2:
                calories = st.number_input("Calories", step=1,  key= f"calpries{i}", min_value = 0, value = None, placeholder=0)

            with col3:
                protein = st.number_input("Protein", key= f"protein{i}", min_value = 0, value = None, placeholder=0)

            with col4:
                carbs = st.number_input("Carbs", key= f"carbs{i}", min_value = 0, value = None, placeholder=0)

            with col5:
                fats = st.number_input("SFats", key= f"fats{i}", min_value = 0, value = None, placeholder=0)
            
            food_item = {"name": name, "calories": calories, "protein": protein, "carbs": carbs, "fats": fats}

            if food_item["name"] != "": #ignore all blanks that are not filled in
                food_items_list.append(food_item)

        # 3. Use the plus button to trigger the callback
        st.button("➕", on_click=add_row)

        food_items = {"food_items" : food_items_list}


        if st.button("Count Calories", key="calorie_count"):
            if not food_items:
                st.warning("Please enter the foods you ate today")
            else:
                prompt_lines = [
                    f"Count calories remaining for the day with information from the food items and user's goal",
                    f"Foods Eaten: {food_items}" if food_items else None,
                ]
                prompt = "\n".join(line for line in prompt_lines if line)
                    
                with st.spinner("Counting Calories..."):
                    result = run_with_status(prompt, food_items, "Running recipe search pipeline...", "calculate_calories")
                    st.session_state['calorie_output'] = result

                    st.session_state["calorie_counter_result"] = result
                    st.session_state["calorie_counter_input"] = food_items

        if st.session_state['calorie_output'] != {}:
            render_result(st.session_state['calorie_output'])
                        
            st.divider()
            
            st.write("Do you have any questions?")
            chat_prompt = st.text_input("Ask a question", key="calorie_chat_input")

            if st.button("Ask", key="calorie_questions"):
                if not chat_prompt.strip():
                    st.warning("Please enter a question.")
                else:
                    result = run_with_status(chat_prompt, {"calorie_counter_output": st.session_state['calorie_output'], "weightloss_plan": st.session_state['weightloss_output']}, "Running chat pipeline...", "general_chat_response")
                    
                    st.session_state["messages"].append(result)

                    render_result(result)
   

    st.header("Generate a Dietary Plan")

    st.info(
        "🧀 **Generate Plan** creates a dietary plan from your manual inputs, "
        "Weight Planning targets, and/or today's calorie counter results."
    )

    if "dietary_output" not in st.session_state:
        st.session_state["dietary_output"] = {}

    st.write("Choose where your plan information should come from.")

    has_weight_plan = st.session_state["weight_plan_result"] is not None
    has_calorie_counter = st.session_state["calorie_counter_result"] is not None

    plan_source = st.radio(
        "Plan source",
        ["Enter my own goal and calorie target", "Use Weight Planning output"],
        disabled=not has_weight_plan,
    )

    use_weight_plan = plan_source == "Use Weight Planning output"

    use_calorie_counter = st.checkbox(
        "Adjust for foods already eaten today",
        value=False,
        disabled=not has_calorie_counter,
    )

    if not has_weight_plan:
        st.caption("Run Weight Planning first if you want to use its goal and calorie target.")

    if not has_calorie_counter:
        st.caption("Run Calorie Counter first if you want the plan adjusted for foods already eaten today.")

    st.divider()

    if use_weight_plan:
        goal = None
        calories = None
        st.info("Using Weight Planning output for both the goal and daily calorie target.")
    else:
        goal = st.text_input(
            "Goal",
            placeholder="e.g. lose weight, build muscle, eat healthier",
        )

        calories = st.number_input(
            "Calories Target",
            min_value=0,
            step=50,
            value=1800,
        )

    if use_calorie_counter:
        days = 1
        st.info(
            "Because you are using today's calorie counter output, "
            "the plan will be generated for 1 day."
        )
    else:
        days = st.number_input(
            "Number of Days",
            min_value=1,
            max_value=7,
            value=3,
            step=1,
        )

    dietary_preference = st.text_input(
        "Dietary Preference",
        placeholder="e.g. vegetarian, vegan, high-protein, gluten-free",
    )

    allergies = st.text_input(
        "Allergies / Foods to Avoid",
        placeholder="e.g. peanuts, shellfish, dairy",
    )

    extra_notes = st.text_area(
        "Extra Notes",
        placeholder="e.g. simple meals, low budget, quick prep, college student",
    )

    if st.button("Generate Plan", key="dietary_plan"):
        if use_weight_plan and not has_weight_plan:
            st.warning("Please run Weight Planning first.")
        elif use_calorie_counter and not has_calorie_counter:
            st.warning("Please run Calorie Counter first.")
        elif not use_weight_plan and not goal:
            st.warning("Please enter your goal.")
        else:
            dietary_plan_input = {
                "plan_source": plan_source,
                "goal": goal,
                "days": days,
                "calories_target": calories,
                "dietary_preference": dietary_preference,
                "allergies": allergies,
                "extra_notes": extra_notes,
            }

            if use_weight_plan:
                dietary_plan_input["weight_plan_result"] = st.session_state["weight_plan_result"]
                dietary_plan_input["weight_plan_input"] = st.session_state["weight_plan_input"]

            if use_calorie_counter:
                dietary_plan_input["calorie_counter_result"] = st.session_state["calorie_counter_result"]
                dietary_plan_input["calorie_counter_input"] = st.session_state["calorie_counter_input"]

            prompt_lines = [
                "Create a dietary plan using the following information.",
                f"Length: {days} day{'s' if days != 1 else ''}",
                f"Dietary preference: {dietary_preference}" if dietary_preference else None,
                f"Allergies or foods to avoid: {allergies}" if allergies else None,
                f"Extra notes: {extra_notes}" if extra_notes.strip() else None,
            ]

            if use_weight_plan:
                prompt_lines.append(
                    "Use the Weight Planning output for both the user's goal and daily calorie target."
                )
            else:
                prompt_lines.append(f"Goal: {goal}")
                prompt_lines.append(f"Calories target: {calories}")

            if use_calorie_counter:
                prompt_lines.append(
                    "Use the Calorie Counter output to account for foods already eaten today. "
                    "Generate a 1-day plan focused on the remaining meals or remaining nutrition needs."
                )

            prompt = "\n".join(line for line in prompt_lines if line)

            with st.spinner("Generating dietary plan..."):
                result = run_with_status(
                    prompt,
                    dietary_plan_input,
                    "Running dietary plan pipeline...",
                    "dietary_plan",
                )
                st.session_state["dietary_output"] = result

    if st.session_state["dietary_output"] != {}:
        render_result(st.session_state["dietary_output"])

        st.divider()

        st.write("Do you have any questions?")
        chat_prompt = st.text_input("Ask a question", key="dietary_chat")

        if st.button("Ask", key="dietary_questions"):
            if not chat_prompt.strip():
                st.warning("Please enter a question.")
            else:
                result = run_with_status(
                    chat_prompt,
                    {"dietary_output": st.session_state["dietary_output"]},
                    "Running chat pipeline...",
                    "general_chat_response",
                )

                st.session_state["messages"].append(result)

                render_result(result)



# ── Tab 2: Recipe Library ─────────────────────────────────────────────
with tab2:
    st.header("Recipe Library")

    st.subheader("Loaded Recipe PDFs")
    recipe_files = list_recipe_files()

    with st.expander("View loaded recipe PDFs", expanded=False):
        if recipe_files:
            for file_name in recipe_files:
                st.markdown(f"- {file_name}")
        else:
            st.info("No recipe PDFs are currently loaded. Upload one in the sidebar.")

    st.divider()

    st.subheader("Search Recipes")

    ingredients = st.text_input(
        "Ingredients",
        placeholder="e.g. chicken, spinach, pasta",
        key="recipe_ingredients",
    )
    extra_recipe_request = st.text_area(
        "Extra Details",
        placeholder="e.g. under 30 minutes, high protein, low sodium",
        key="recipe_extra_details",
    )

    if st.button("Search Recipes", key="search_recipes"):
        if not ingredients.strip():
            st.warning("Please enter at least one ingredient.")
        else:
            parsed_ingredients = [
                item.strip()
                for item in ingredients.split(",")
                if item.strip()
            ]

            recipe_search_input = {
                "ingredients": parsed_ingredients,
                "extra_details": extra_recipe_request.strip(),
            }

            prompt = f"Find recipes with {', '.join(parsed_ingredients)}."

            if extra_recipe_request.strip():
                prompt += f"\nExtra details: {extra_recipe_request.strip()}"

            result = run_with_status(
                prompt,
                recipe_search_input,
                "Running recipe search pipeline...",
                "recipe_search"
            )

            recipe_matches = extract_recipe_matches(result)

            warnings = result.get("warnings", [])
            for w in warnings:
                st.warning(w)

            if recipe_matches:
                st.subheader("Top Recipe Matches")
                for match in recipe_matches:
                    with st.expander(match["title"]):
                        st.caption(f"Source: {match['source']}")
                        cleaned_recipe = format_recipe_match_with_llm(match)
                        st.markdown(cleaned_recipe)
            else:
                render_result(result)