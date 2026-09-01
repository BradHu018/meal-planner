from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from data.nutrition import NUTRITION_DATA
from data.prices import PRICE_DATA

from data.nutrition import (
    calculate_recipe_nutrition,
    scale_recipe_to_targets,
)

from data.prices import calculate_recipe_cost
from pydantic import BaseModel, Field 
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import RECIPE_PROMPT, TASTE_PROMPT, BALANCE_PROMPT, OPTIMIZER_PROMPT

load_dotenv()

class Ingredient(BaseModel):
    name: str = Field(
        description = "ingredient name exactly as provided in the allowed ingredient list."
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

    model = init_chat_model("google_genai:gemini-3.6-flash")

    recipe_model = model.with_structured_output(RecipeList)

    SUPPORTED_INGREDIENTS = sorted(
        set(NUTRITION_DATA.keys())
        & set(PRICE_DATA.keys())
    )

    response = recipe_model.invoke([
        SystemMessage(content=RECIPE_PROMPT
    ),
        HumanMessage(
            content=f"""
Planning constraints: 
{constraints}

Allowed ingredients:

{SUPPORTED_INGREDIENTS}

Generate 10 candidate recipes using only these allowed ingredients
"""
        )
    ])

    recipes = [
        recipe.model_dump() 
        for recipe in response.recipes
    ]

    print("\n GENERATED RECIPES")

    for recipe in recipes:
        print(recipe["name"])
        print(recipe["ingredients"])
        print()

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
    

def taste_agent(state: MealPlanState):
    recipes = state["portioned_recipes"]

    # LLM analyzes taste compatability 
    
    return {
        "taste_analysis": {}
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

def meal_balance_agent(state: MealPlanState):
    recipes = state["portioned_recipes"]

    goals = state["nutrition_goals"]

    # LLM gets:
    #
    # recipe nutrition
    # supplied goals
    # recipe ingredients
    #
    # and evaluates the meal options

    return {
        "balance_analysis": {}
    }



def optimizer_node(state: MealPlanState):
      # Reads:
    #
    # portioned_recipes
    # taste_analysis
    # budget_analysis
    # balance_analysis
    #
    # LLM selects meals.

    return {
        "weekly_plan": [],
        "grocery_list": [],
        "estimated_total": 0
    }


def critic_node(state: MealPlanState):
    problems = []

    if (
        state["estimated_total"]
        > state["weekly_budget"]
    ):
        problems.append(
            "Weekly plan exceeds budget."
        )

    expected_meals = (
        state["preferences"]["meals_needed"]
    )

    if len(state["weekly_plan"]) != expected_meals:
        problems.append(
            "Incorrect number of meals."
        )

    # Later:
    # LLM checks softer concerns like
    # variety and preference match.

    approved = len(problems) == 0

    return {
        "approved": approved,
        "critic_feedback": "\n".join(problems)
    }


def revision_node(state: MealPlanState):

    # LLM receives:
    #
    # existing plan
    # candidate recipes
    # critic feedback
    #
    # Then substitutes recipes.

    return {
        "weekly_plan": [],
        "grocery_list": [],
        "estimated_total": 0,
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
builder.add_edge("taste", "optimizer")
builder.add_edge("budget", "optimizer")
builder.add_edge("balance", "optimizer")

builder.add_edge("optimizer", "critic")

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