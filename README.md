# String Cheese: AI Dietary Assistant

String Cheese is an AI-powered dietary assistant that helps users plan meals, track nutrition, and search recipes from both built-in and user-uploaded cookbooks.

Built with Streamlit, LangGraph, and Google Gemini, this application combines deterministic nutrition tools with LLM-generated planning and PDF-based recipe retrieval.

---

## Team

- Freddy Melges  
- Gillian Donley  
- Raighen Ly  

---

## Overview

String Cheese functions as a “pocket dietitian,” providing:

- Personalized dietary plans
- Weight planning (lose, maintain, gain)
- Calorie tracking
- Macro tracking
- Recipe search from PDFs
- Ability to upload custom cookbook PDFs

The system routes user requests through a structured workflow using LangGraph, combining deterministic tools, retrieval-based search, and LLM-based generation.

---

## Use of AI Tools

This project was developed for the course **Creating with AI in the Loop**, where the goal was to design and build software while actively collaborating with AI tools throughout the development process.

AI tools were used in two main ways:

1. **As part of the application itself**  
   String Cheese uses a large language model through Google Gemini to generate dietary plans, format recipe search results, and provide natural-language responses. The app combines LLM-based generation with deterministic tools for calorie, macro, and weight-planning calculations.

2. **As a development assistant**  
   During development, AI tools were used to help brainstorm features, plan the application structure, debug code, improve documentation, and refine the LangGraph workflow. Human team members made the final design decisions, reviewed generated code, tested functionality, and integrated the system into the final project.

This project reflects an “AI in the loop” approach: AI supported both the user-facing experience and the development process, but the final implementation, evaluation, and project direction were guided by the team.

## Features

### Dietary Plan Generator

- Generates multi-day meal plans
- Uses:
  - Goal (lose/gain/maintain)
  - Target calories
  - Dietary preferences (vegan, high-protein, etc.)
  - Allergies
  - Custom notes
- Plans are generated via Gemini with built-in safety constraints

---

### Weight Planning Tool

- Calculates:
  - Resting Metabolic Rate (RMR)
  - Maintenance calories
  - Target calorie intake
- Supports weight loss, maintenance, and weight gain

---

### Calorie Counter

- Tracks total calories and macronutrients from user input
- Can be used for daily, weekly, or monthly tracking

---

### Macro Tracker

- Breaks calorie targets into:
  - Protein
  - Carbohydrates
  - Fats
- Adjusts distribution based on user goals

---

### Recipe Library and PDF Upload

- Includes preloaded recipe documents
- Users can upload cookbook PDFs
- Uploaded recipes are indexed and searchable

---

### Recipe Search (BM25 Retrieval)

- Search recipes by:
  - Ingredients
  - Additional constraints (e.g., "high protein", "quick meals")

**How it works:**

- PDFs are parsed using PyPDFLoader  
- Text is indexed using BM25 keyword search  
- Results are filtered by required ingredients  
- Matches are ranked and formatted using the LLM  

---

## Technical Architecture

### LangGraph Workflow

All requests pass through a structured pipeline:

    parse_input
    ↓
    route_request
    ↓
    [tool-specific node]
    ↓
    format_response


---

### Core Components

- Frontend: Streamlit (`app.py`)
- Workflow Engine: LangGraph
- LLM: Google Gemini (or mock fallback)
- Tools:
  - Weight planning
  - Calorie counter
  - Macro tracker
  - Recipe search
- Retriever:
  - BM25 keyword-based search
  - PDF and text ingestion

---

## Project Structure
    app.py # Streamlit UI

    ai_in_loop/
    ├── graph.py # LangGraph workflow
    ├── nodes.py # Workflow nodes
    ├── tools.py # Core tools
    ├── retriever.py # BM25 search implementation
    ├── llm.py # Gemini / mock model setup
    ├── config.py # Environment configuration
    ├── dietary_plan/ # Meal planning logic

    resources/
    └── uploaded/ # User-uploaded PDFs


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/string-cheese.git
cd string-cheese
```

### 2. Create a Virtual Environment
```
python -m venv .venv
```

Activate it:

macOS/Linux
```
source .venv/bin/activate
```

Windows (PowerShell)
```
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

Optional (for graph visualization):
```
pip install "langgraph[visualization]"
```

### 4. Configure Environment Variables

Copy the example file:

```
cp .env.example .env
```

Edit .env:
    USE_GEMINI=1
    GEMINI_API_KEY=your_api_key_here
    GEMINI_MODEL=gemini-2.5-flash

If no API key is provided, the app will use a mock LLM.

5. Run the Application

```
streamlit run app.py
```

Open the local URL shown in the terminal.

# Uploading Recipes
- Use the sidebar to upload PDF files
- Files are saved to:
    - resources/uploaded/
- The search index is refreshed automatically after upload

# How Recipe Search Works
1. PDFs are loaded and converted into text
2. Documents are indexed using BM25
3. Queries are built from:
    - Ingredients
    - Extra user input
4. Results are:
    - Filtered (must contain all ingredients)
    - Ranked
    - Formatted into readable recipes using the LLM

# Notes and Limitations
- Recipe search uses keyword-based retrieval (BM25), not semantic embeddings
- Large PDFs may return broad or noisy matches
- Generated dietary plans are not medical advice

# Future Improvements
- Add embedding-based semantic search
- Improve PDF chunking for better recipe extraction
- Add persistent user profiles
- Improve macro accuracy and validation
- Add automated testing