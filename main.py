from workflow import graph


initial_state = {

    # USER INPUT
    "preferences": {
        "liked_cuisines": [
            "Korean",
            "Chinese"
        ],

        "disliked_foods": [
            "olives"
        ],

        "max_cooking_time": 30,

        "meals_needed": 7,
    },

    "pantry": [
        "rice",
        "soy sauce"
    ],

    "weekly_budget": 70,

    # Treat these as supplied/demo constraints.
    # v0.1 does NOT calculate a person's needs
    # from body measurements.
    "nutrition_goals": {
        "meal_calories": 600,
        "meal_protein_g": 30,
    },


    # =========================
    # EMPTY GRAPH STATE
    # =========================

    "planning_constraints": {},

    "candidate_recipes": [],

    "enriched_recipes": [],

    "portioned_recipes": [],

    "taste_analysis": {},
    "budget_analysis": {},
    "balance_analysis": {},

    "weekly_plan": [],
    "grocery_list": [],
    "estimated_total": 0,

    "critic_feedback": "",
    "approved": False,
    "revision_count": 0,

    "final_result": {},
}


result = graph.invoke(initial_state)


print(result["final_result"])