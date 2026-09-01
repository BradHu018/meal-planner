import pandas as pd 

food_df = pd.read_csv("data/cnf/food_name.csv")
nutrient_name_df = pd.read_csv("data/cnf/nutrient_name.csv")
nutrient_amount_df = pd.read_csv("data/cnf/nutrient_amount.csv")

PROTEIN_CODE = 203 
CALORIE_CODE = 208


# print("FOOD COLUMNS: ")
# print(food_df.columns)

# print("\nNUTRIENT NAME COLUMNS:")
# print(nutrient_name_df.columns)

# print("\nNUTRIENT AMOUNT COLUMNS:")
# print(nutrient_amount_df.columns)

# print("\nFOOD SAMPLE:")
# print(food_df.head())

# print("\nNUTRIENT SAMPLE:")
# print(nutrient_name_df.head())

# NUTRITION_DATA = {
#     "chicken breast": {
#         "calories_per_100g": 165,
#         "protein_per_100g": 31,
#     },

#     "rice": {
#         "calories_per_100g": 130,
#         "protein_per_100g": 2.7,
#     },

#     "broccoli": {
#         "calories_per_100g": 35,
#         "protein_per_100g": 2.4,
#     },

#     "tofu": {
#         "calories_per_100g": 144,
#         "protein_per_100g": 17,
#     },
# }

def get_nutrition(food_name):
    return NUTRITION_DATA.get(food_name)

def calculate_ingredient_nutrition(food_name, grams):

    data = get_nutrition(food_name)

    if data is None:
        return None

    multiplier = grams / 100

    return {
        "calories": data["calories_per_100g"] * multiplier,
        "protein_g": data["protein_per_100g"] * multiplier,
    }

def calculate_recipe_nutrition(recipe):

    total_calories = 0
    total_protein = 0

    for ingredient in recipe["ingredients"]:

        nutrition = calculate_ingredient_nutrition(
            ingredient["name"],
            ingredient["grams"]
        )

        if nutrition is None:
            continue

        total_calories += nutrition["calories"]
        total_protein += nutrition["protein_g"]

    return {
        "calories": round(total_calories, 1),
        "protein_g": round(total_protein, 1),
    }

# optimizer to find out the most appropriate scale to match the user's goal for calories and protein
def scale_recipe_to_targets(
    recipe,
    target_calories,
    target_protein
):
    best_recipe = None
    best_score = float("inf")

    scale_factors = [
        0.75,
        0.85,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
    ]

    for scale in scale_factors:

        scaled_ingredients = []

        for ingredient in recipe["ingredients"]:
            scaled_ingredients.append({
                "name": ingredient["name"],
                "grams": round(
                    ingredient["grams"] * scale,
                    1
                )
            })

        scaled_recipe = {
            **recipe,
            "ingredients": scaled_ingredients,
        }

        nutrition = calculate_recipe_nutrition(
            scaled_recipe
        )

        calorie_error = (
            abs(nutrition["calories"] - target_calories)
            / target_calories
        )

        protein_error = (
            abs(nutrition["protein_g"] - target_protein)
            / target_protein
        )

        score = calorie_error + protein_error

        if score < best_score:
            best_score = score

            best_recipe = {
                **scaled_recipe,
                "nutrition": nutrition,
                "scale_factor": scale,
            }

    return best_recipe