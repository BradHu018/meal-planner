from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from debug.state_utils import save_state
import os

MODEL_NAME = os.getenv(
    "MEAL_PLANNER_MODEL",
    "google_genai:gemini-3.5-flash-lite"
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langchain_google_genai.chat_models import GoogleAPIError

@retry(
    retry=retry_if_exception_type(GoogleAPIError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(
        multiplier=2,
        min=2,
        max=20,
    ),
    reraise=True,
)
def invoke_with_retry(model, messages):
    return model.invoke(messages)

from data.nutrition import (
    calculate_recipe_nutrition,
    scale_recipe_to_targets,
)
from data.prices import calculate_recipe_cost

from pydantic import BaseModel, Field 
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import RECIPE_PROMPT, TASTE_PROMPT, BALANCE_PROMPT, OPTIMIZER_PROMPT, REVISION_PROMPT, CRITIC_PROMPT

load_dotenv()

class Ingredient(BaseModel):
    name: str = Field(
        description=(
            "A clear common ingredient name. "
            "Include preparation details when useful, "
            "for example 'white rice cooked' or "
            "'grilled chicken breast'."
        )
    )

    grams: float = Field(
        gt=0,
        description="Amount of the ingredient in grams"
    )

class Recipe(BaseModel):
    name: str 
    cuisine: str 
    cooking_time: int 
    ingredients: list[Ingredient]

class RecipeList(BaseModel):
    recipes: list[Recipe]

class TasteEvaluation(BaseModel):
    recipe_name: str = Field(
        description="Exact name of the recipe being evaluated."
    )

    score: float = Field(
        ge=0,
        le=10,
        description="Taste compatibility score from 0 to 10."
    )

    reasoning: str = Field(
        description="Short explanation of why this recipe received the score"
    )

    matched_preferences: list[str] = Field(
        description="User preferences that this recipe matches."
    )

    concerns: list[str] = Field(
        description="Taste-related concerns or preference mismatches."
    )

class TasteAnalysis(BaseModel):
    evaluations: list[TasteEvaluation]

class BalanceEvaluation(BaseModel):
    recipe_name: str = Field(
        description="Exact name of the recipe being evaluated."
    )

    score: float = Field(
        ge=0,
        le=10,
        description="Meal balance score from 0 to 10."
    )

    reasoning: str = Field(
        description="Short explanation of why this recipe received the score"
    )

    strenghts: list[str] = Field(
        description="Positive qualities of this recipe for the weekly meal plan."
    )

    concerns: list[str] = Field(
        description="Potential balance or repetition concerns."
    )


class BalanceAnalysis(BaseModel):
    evaluations: list[BalanceEvaluation]

    overall_summary: str = Field(
        description="Overall summary of the candidate recipe pool."
    )

    repetition_concerns: list[str] = Field(
        description="Ingredeints, protein sources, cuisines, or meal styles that repeats often"
    )

    optimizer_suggestions: list[str] = Field(
        description="Suggestions to help the optimizer choosea variety set of meals."
    )


class SelectedMeal(BaseModel):
    recipe_name: str = Field(
        description="Exact name of a selected candidate recipe."
    )

    reason: str = Field(
        description="Short explanation of why this recipe was selected."
    )

class OptimizerSelection(BaseModel):
    selected_meals: list[SelectedMeal] = Field(
        description="Recipes selected for the weekly meal plan."
    )

    reasoning: str = Field(
        description="Short explanation of how the plan balances taste, cost, variety, and meal balance."
    )
class CriticReview(BaseModel):

    revision_problems: list[str] = Field(
        description=(
            "Meaningful problems that can actually be improved "
            "by swapping selected recipes with other existing candidates. "
            "Return an empty list if there are no revision-worthy problems."
        )
    )

    warnings: list[str] = Field(
        description=(
            "Weaknesses or limitations worth mentioning that cannot "
            "reasonably be fixed using the current candidate pool. "
            "Warnings should not cause rejection."
        )
    )

    suggestions: list[str] = Field(
        description=(
            "Specific recipe-selection changes that the revision agent "
            "can make using existing candidates."
        )
    )

    summary: str = Field(
        description=(
            "Short overall assessment of the selected weekly meal plan."
        )
    )

class RevisionSelection(BaseModel):
    selected_recipe_names: list[str] = Field(
        description=(
            "Exact names of the recipes selected for the revised weekly plan."
        )
    )

    changes_made: list[str] = Field(
        description=(
            "Short descriptions of which meals were replaced and why"
        )
    )

    reasoning: str = Field(
        description=(
            "Breif explanation of how the revised selection addresses the critic feedback"
        )
    )



# STATE
class MealPlanState(TypedDict):

    # user input
    preferences: dict
    pantry: list[str]
    weekly_budget: float
    nutrition_goals: dict 


    # planner
    planning_constraints: dict

    # recipes
    candidate_recipes: list[dict]

    enriched_recipes: list[dict]

    portioned_recipes: list[dict]

    #analysis 
    taste_analysis: dict
    budget_analysis: dict
    balance_analysis: dict

    #plan
    weekly_plan: list[dict]
    grocery_list: list[dict]
    estimated_total: float

    # critic 
    critic_feedback: str
    approved: bool
    revision_count: int

    # output 
    final_result: dict


# NODES
# gets the input data into structured data for other nodes to use
def planner_node(state: MealPlanState):
    constraints = {
        "preferred_cuisines":
            state["preferences"]["liked_cuisines"],

        "avoid":
            state["preferences"]["disliked_foods"],

        "max_cooking_time":
            state["preferences"]["max_cooking_time"],

        "meals_needed":
            state["preferences"]["meals_needed"],

        "budget":
            state["weekly_budget"],

        "pantry":
            state["pantry"],

        "nutrition_goals":
            state["nutrition_goals"],
    }

    return {
        "planning_constraints": constraints
    }


def recipe_generator_node(state: MealPlanState):

    constraints = state["planning_constraints"]

    model = init_chat_model(
        "MODEL_NAME"
    )

    recipe_model = model.with_structured_output(
        RecipeList
    )

    response = invoke_with_retry(
        recipe_model,
        [
            SystemMessage(
                content=RECIPE_PROMPT
            ),

            HumanMessage(
                content=f"""
Planning constraints:

{constraints}

Generate 10 candidate recipes.

Use common grocery ingredients and use clear,
specific ingredient names so they can be matched
against nutrition and grocery datasets.

All ingredient quantities must be in grams.
"""
        )
    ])

    recipes = [
        recipe.model_dump()
        for recipe in response.recipes
    ]

    return {
        "candidate_recipes": recipes
    }


# adds the calories and proteins to the generated recipe 
def nutrition_enrichment_node(state: MealPlanState):
    enriched = []

    for recipe in state["candidate_recipes"]:
        nutrition = calculate_recipe_nutrition(recipe)

        enriched_recipe = {
            # dictionary unpacking: copy paste key/value pairs from recipe into this new dict called nutritions
            **recipe, 
            "nutrition": nutrition,
        }

        enriched.append(enriched_recipe)
    return {
        "enriched_recipes": enriched
    }

# adjust recipe ingredient quantities to meet the nutrition goals
def portion_calculator_node(state: MealPlanState):
    target_calories = (
        state["nutrition_goals"]["meal_calories"]
    )

    target_protein = (
        state["nutrition_goals"]["meal_protein_g"]
    )

    portioned = []

    for recipe in state["enriched_recipes"]:
        adjusted = scale_recipe_to_targets (
            recipe, 
            target_calories,
            target_protein
        )

        portioned.append(adjusted)
    return {
        "portioned_recipes": portioned
    }

# evaluate the taste of the generated recipe based on user preferences and disliked foods
def taste_agent(state: MealPlanState):

    recipes = state["portioned_recipes"]
    preferences = state["preferences"]

    model = init_chat_model(
        "MODEL_NAME"
    )

    taste_model = model.with_structured_output(
        TasteAnalysis
    )

    response = invoke_with_retry(
    taste_model,
    [
        SystemMessage(
            content=TASTE_PROMPT
        ),
        HumanMessage(
            content=f"""
User preferences:

{preferences}

Candidate recipes:

{recipes}

Evaluate every candidate recipe for taste compatibility.

Make sure every recipe receives exactly one evaluation.
Use the exact recipe name provided in the candidate recipe data.
"""
        )
    ])

    print("\n=== TASTE ANALYSIS ===")

    for evaluation in response.evaluations:
        print(
            evaluation.recipe_name,
            "->",
            evaluation.score
        )

        print(evaluation.reasoning)
        print()

    return {
        "taste_analysis":
            response.model_dump()
    }

# calculate real cost based on the adjusted ingredient quantities
def budget_agent(state: MealPlanState):
    recipes = state["portioned_recipes"]

    costs = {}

    for recipe in recipes:
        cost = calculate_recipe_cost(
            recipe, 
            state["pantry"]
        )

        costs[recipe["name"]] = cost 
    return{
        "budget_analysis": {
            "recipe_cost": costs,
            "weekly_budget":
            state["weekly_budget"], 
        }
    }

# balance meal based on reptition and variety of ingredients, cuisines, and protein sources
def meal_balance_agent(state: MealPlanState):

    recipes = state["portioned_recipes"]
    goals = state["nutrition_goals"]

    model = init_chat_model(
        "MODEL_NAME"
    )

    balance_model = model.with_structured_output(
        BalanceAnalysis
    )

    response = invoke_with_retry(
    balance_model,
    [
        SystemMessage(
            content=BALANCE_PROMPT
        ),
        HumanMessage(
            content=f"""
Supplied meal constraints:

{goals}


Candidate recipes with already-calculated nutrition:

{recipes}


Evaluate every recipe exactly once.

Use the exact recipe names provided.

Also evaluate the candidate pool as a whole for
repetition and variety so the optimizer can later
select a balanced set of meals.
"""
        )
    ])

    print("\n=== BALANCE ANALYSIS ===")

    for evaluation in response.evaluations:

        print(
            evaluation.recipe_name,
            "->",
            evaluation.score
        )

        print(evaluation.reasoning)
        print()

    print("OVERALL:")
    print(response.overall_summary)

    return {
        "balance_analysis":
            response.model_dump()
    }


# builds a grocery list based on the selected recipes and the user's pantry
def build_grocery_list(selected_recipes, pantry):
    grocery_totals = {}

    pantry_normalized = {
        item.lower().strip()
        for item in pantry
    }

    for recipe in selected_recipes:

        for ingredient in recipe["ingredients"]:
            name = ingredient["name"]
            grams = ingredient["grams"]

            normalized_name = name.lower().strip()

            # v0.1 assumption:
            # if ingredient is in pantry, we do not purchase it
            if normalized_name in pantry_normalized:
                continue

            if normalized_name not in grocery_totals:
                grocery_totals[normalized_name] = {
                    "name": name,
                    "grams": 0
                }

            grocery_totals[normalized_name]["grams"] += grams

    grocery_list = []

    for ingredient in grocery_totals.values():
        grocery_list.append({
            "name": ingredient["name"],
            "grams": round(
                ingredient["grams"],
                1
            )
        })

    return grocery_list


# calculate the total cost of the weekly recipe
def calculate_selected_total(
    selected_recipes,
    budget_analysis
):
    recipe_costs = budget_analysis[
        "recipe_cost"
    ]

    total = 0

    for recipe in selected_recipes:
        recipe_name = recipe["name"]

        cost_info = recipe_costs.get(
            recipe_name
        )

        if cost_info is None:
            continue

        total += cost_info[
            "estimated_cost"
        ]

    return round(total, 2)

# optimizes the weekly recipes based on the taste, cost, and balance analysis
def optimizer_node(state: MealPlanState):

    recipes = state["portioned_recipes"]

    taste_analysis = state[
        "taste_analysis"
    ]

    budget_analysis = state[
        "budget_analysis"
    ]

    balance_analysis = state[
        "balance_analysis"
    ]

    meals_needed = state[
        "preferences"
    ]["meals_needed"]

    weekly_budget = state[
        "weekly_budget"
    ]

    model = init_chat_model(
        "MODEL_NAME"
    )

    optimizer_model = (
        model.with_structured_output(
            OptimizerSelection
        )
    )

    response = invoke_with_retry(
    optimizer_model,
    [
        SystemMessage(
            content=OPTIMIZER_PROMPT
        ),
        HumanMessage(
            content=f"""
Number of meals required:
{meals_needed}

Weekly budget:
{weekly_budget}


Candidate recipes:

{recipes}


Taste analysis:

{taste_analysis}


Budget analysis:

{budget_analysis}


Balance analysis:

{balance_analysis}


Select exactly {meals_needed} unique recipes.

Use only exact recipe names from the candidate recipe list.
"""
        )
    ])

    selected_names = [
        meal.recipe_name
        for meal in response.selected_meals
    ]

    print(
        "\n=== OPTIMIZER SELECTION ==="
    )

    for meal in response.selected_meals:
        print(
            meal.recipe_name,
            "->",
            meal.reason
        )

    # Create lookup table:
    #
    # recipe name -> complete recipe dict
    recipe_lookup = {
        recipe["name"]: recipe
        for recipe in recipes
    }

    selected_recipes = []

    for name in selected_names:

        if name not in recipe_lookup:
            continue

        selected_recipes.append(
            recipe_lookup[name]
        )

    # Safety check:
    # Gemini must return exactly the requested
    # number of valid recipes.
    if len(selected_recipes) != meals_needed:
        raise ValueError(
            f"Optimizer selected "
            f"{len(selected_recipes)} valid meals, "
            f"but {meals_needed} were required."
        )

    grocery_list = build_grocery_list(
        selected_recipes,
        state["pantry"]
    )

    estimated_total = (
        calculate_selected_total(
            selected_recipes,
            budget_analysis
        )
    )

    weekly_plan = []

    for index, recipe in enumerate(
        selected_recipes,
        start=1
    ):
        weekly_plan.append({
            "meal_number": index,
            **recipe
        })

    return {
        "weekly_plan": weekly_plan,
        "grocery_list": grocery_list,
        "estimated_total":
            estimated_total
    }


# in the case that the food is not provided per 100g this gets the missing prices of those items
def get_selected_missing_prices(
    weekly_plan,
    budget_analysis
):
    recipe_costs = budget_analysis[
        "recipe_cost"
    ]

    missing = set()

    for meal in weekly_plan:

        recipe_name = meal["name"]

        cost_info = recipe_costs.get(
            recipe_name
        )

        if cost_info is None:
            continue

        for ingredient in cost_info.get(
            "missing_prices",
            []
        ):
            missing.add(ingredient)

    return sorted(missing)

def save_pre_critic_state_node(
    state: MealPlanState
):
    save_state(
        state,
        "pre_critic_state.json"
    )
    return {}


def critic_node(state: MealPlanState):

    hard_problems = []
    warnings = []

    weekly_plan = state["weekly_plan"]

    expected_meals = (
        state["preferences"]["meals_needed"]
    )

    # 1. Correct number of meals

    if len(weekly_plan) != expected_meals:

        hard_problems.append(
            f"Expected {expected_meals} meals, "
            f"but received {len(weekly_plan)}."
        )

    # 2. Duplicate recipes

    selected_names = [
        meal["name"]
        for meal in weekly_plan
    ]

    if len(selected_names) != len(
        set(selected_names)
    ):

        hard_problems.append(
            "Weekly plan contains duplicate recipes."
        )

    # 3. Valid candidate recipes

    valid_recipe_names = {
        recipe["name"]
        for recipe in state["portioned_recipes"]
    }

    invalid_recipes = [
        name
        for name in selected_names
        if name not in valid_recipe_names
    ]

    if invalid_recipes:

        hard_problems.append(
            "Weekly plan contains recipes that "
            f"were not valid candidates: "
            f"{invalid_recipes}"
        )

    # 4. Budget

    if (
        state["estimated_total"]
        > state["weekly_budget"]
    ):

        hard_problems.append(
            f"Weekly plan exceeds budget: "
            f"${state['estimated_total']:.2f} "
            f"> ${state['weekly_budget']:.2f}."
        )

    # 5. Missing price information

    missing_prices = (
        get_selected_missing_prices(
            weekly_plan,
            state["budget_analysis"]
        )
    )

    if missing_prices:

        warnings.append(
            "The grocery price estimate is incomplete because "
            "prices are missing for: "
            + ", ".join(missing_prices)
        )

    # 6. Gemini soft-quality review

    model = init_chat_model(
        MODEL_NAME
    )

    critic_model = model.with_structured_output(
        CriticReview
    )

    response = invoke_with_retry(
    critic_model,
    [
        SystemMessage(
            content=CRITIC_PROMPT
        ),
        HumanMessage(
            content=f"""
Selected weekly plan:

{weekly_plan}


User preferences:

{state["preferences"]}


All candidate recipes:

{state["portioned_recipes"]}


Taste analysis:

{state["taste_analysis"]}


Meal-balance analysis:

{state["balance_analysis"]}


Budget analysis:

{state["budget_analysis"]}


Deterministic hard problems already found:

{hard_problems}


Deterministic warnings already found:

{warnings}


Review the weekly plan for meaningful quality problems.

Only put something in revision_problems if it can realistically
be improved by replacing selected recipes with other recipes from
the CURRENT candidate pool.

Problems caused by limitations of the entire candidate pool should
go into warnings instead.

Do not suggest:
- changing ingredient quantities
- adding ingredients
- generating new recipes
- retrieving new recipes
- recalculating nutrition
- recalculating prices
"""
        )
    ])

    # 7. Get Gemini results

    revision_problems = (
        response.revision_problems
    )

    all_warnings = (
        warnings
        + response.warnings
    )

    # 8. Final approval

    approved = (
        len(hard_problems) == 0
        and len(revision_problems) == 0
    )

    # 9. Build critic feedback

    feedback_parts = []

    if hard_problems:

        feedback_parts.append(
            "Hard constraint problems:\n- "
            + "\n- ".join(
                hard_problems
            )
        )

    if revision_problems:

        feedback_parts.append(
            "Revision-worthy problems:\n- "
            + "\n- ".join(
                revision_problems
            )
        )

    if response.suggestions:

        feedback_parts.append(
            "Revision suggestions:\n- "
            + "\n- ".join(
                response.suggestions
            )
        )

    if all_warnings:

        feedback_parts.append(
            "Warnings:\n- "
            + "\n- ".join(
                all_warnings
            )
        )

    critic_feedback = "\n\n".join(
        feedback_parts
    )

    # 10. Debug output

    print("\n=== CRITIC ===")

    print(
        "Approved:",
        approved
    )

    print("\nSummary:")
    print(
        response.summary
    )

    if critic_feedback:

        print("\nFeedback:")
        print(
            critic_feedback
        )

    # 11. Update LangGraph state

    return {
        "approved": approved,
        "critic_feedback":
            critic_feedback
    }

def revision_node(state: MealPlanState):

    current_plan = state["weekly_plan"]
    candidate_recipes = state["portioned_recipes"]

    critic_feedback = state["critic_feedback"]

    meals_needed = (
        state["preferences"]["meals_needed"]
    )

    weekly_budget = state["weekly_budget"]

    model = init_chat_model(
        MODEL_NAME
    )

    revision_model = model.with_structured_output(
        RevisionSelection
    )

    response = invoke_with_retry(
        revision_model,
        [
            SystemMessage(
                content=REVISION_PROMPT
            ),

            HumanMessage(
                content=f"""
Current weekly plan:

{current_plan}


Critic feedback:

{critic_feedback}


Required number of meals:

{meals_needed}


Weekly budget:

{weekly_budget}


Available candidate recipes:

{candidate_recipes}


Taste analysis:

{state["taste_analysis"]}


Balance analysis:

{state["balance_analysis"]}


Budget analysis:

{state["budget_analysis"]}


Revise the current meal selection using only
the available candidate recipes.

Preserve as many good existing selections as possible.

Return exactly {meals_needed} unique recipe names.
"""
            )
        ]
    )

    # ================================
    # Debug output
    # ================================

    print("\n=== REVISION ===")

    print("\nCritic feedback:")
    print(critic_feedback)

    print("\nChanges made:")

    for change in response.changes_made:
        print("-", change)

    print("\nReasoning:")
    print(response.reasoning)

    print("\nRevised recipes:")

    for name in response.selected_recipe_names:
        print("-", name)

    # ================================
    # Build recipe lookup
    # ================================

    recipe_lookup = {
        recipe["name"]: recipe
        for recipe in candidate_recipes
    }

    # ================================
    # Validate Gemini selection
    # ================================

    selected_recipes = []

    seen = set()

    for name in response.selected_recipe_names:

        # Gemini returned an unknown recipe
        if name not in recipe_lookup:
            continue

        # Gemini returned a duplicate
        if name in seen:
            continue

        seen.add(name)

        selected_recipes.append(
            recipe_lookup[name]
        )

    # Revision must still produce
    # exactly the required number of meals
    if len(selected_recipes) != meals_needed:

        raise ValueError(
            f"Revision selected "
            f"{len(selected_recipes)} valid unique meals, "
            f"but {meals_needed} were required."
        )

    # ================================
    # Rebuild weekly plan
    # ================================

    weekly_plan = []

    for index, recipe in enumerate(
        selected_recipes,
        start=1
    ):

        weekly_plan.append({
            "meal_number": index,
            **recipe
        })

    # ================================
    # Rebuild grocery list
    # ================================

    grocery_list = build_grocery_list(
        selected_recipes,
        state["pantry"]
    )

    # ================================
    # Recalculate cost deterministically
    # ================================

    estimated_total = (
        calculate_selected_total(
            selected_recipes,
            state["budget_analysis"]
        )
    )

    print(
        "\nRevised estimated total:",
        estimated_total
    )

    # Update state

    return {
        "weekly_plan": weekly_plan,
        "grocery_list": grocery_list,
        "estimated_total": estimated_total,

        "revision_count":
            state["revision_count"] + 1,
    }


def finalize_node(state):

    return {
        "final_result": {

            "weekly_plan":
                state["weekly_plan"],

            "grocery_list":
                state["grocery_list"],

            "estimated_total":
                state["estimated_total"],

            "weekly_budget":
                state["weekly_budget"],

            "nutrition_goals":
                state["nutrition_goals"],

            "approved":
                state["approved"],

            "critic_feedback":
                state["critic_feedback"],
        }
    }

# CONDITIONAL EDGE


def route_after_critic(state: MealPlanState):

    if state["approved"]:
        return "finalize"

    if state["revision_count"] >= 2:
        return "finalize"

    return "revision"


# =========================
# GRAPH
# =========================

builder = StateGraph(MealPlanState)

builder.add_node("planner", planner_node)
builder.add_node("recipe_generator", recipe_generator_node)
builder.add_node("nutrition_enrichment", nutrition_enrichment_node)
builder.add_node("portion_calculator", portion_calculator_node)


builder.add_node("taste", taste_agent)
builder.add_node("budget", budget_agent)
builder.add_node("balance", meal_balance_agent)

builder.add_node("optimizer", optimizer_node)
builder.add_node("critic", critic_node)
builder.add_node("revision", revision_node)
builder.add_node("finalize", finalize_node)
builder.add_node("save_pre_critic_state", save_pre_critic_state_node)


# edges
builder.add_edge(START, "planner")

builder.add_edge("planner", "recipe_generator")

builder.add_edge("recipe_generator", "nutrition_enrichment")

builder.add_edge("nutrition_enrichment", "portion_calculator")

# parallel
builder.add_edge("portion_calculator", "taste")
builder.add_edge("portion_calculator", "budget")
builder.add_edge("portion_calculator", "balance")

# join
builder.add_edge(
    ["taste", "budget", "balance"],
    "optimizer"
)

# builder.add_edge("optimizer", "critic")
builder.add_edge(
    "optimizer",
    "save_pre_critic_state"
)

builder.add_edge(
    "save_pre_critic_state",
    "critic"
)

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "revision": "revision",
        "finalize": "finalize",
    }
)

builder.add_edge("revision", "critic")
builder.add_edge("finalize", END)


graph = builder.compile()


png_data = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)