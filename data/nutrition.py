import pandas as pd 
import re 
from difflib import SequenceMatcher 
from functools import lru_cache 


food_df = pd.read_csv("data/cnf/food_name.csv")
nutrient_name_df = pd.read_csv("data/cnf/nutrient_name.csv")
nutrient_amount_df = pd.read_csv("data/cnf/nutrient_amount.csv")

PROTEIN_CODE = 203 
CALORIE_CODE = 208

def search_food(name):
    results = food_df[
        food_df["Food_Description_EN"]
        .str.contains(name, case=False, na=False)
    ]
    return results[
        ["Food_Code", "Food_Description_EN"]
    ]

# Words that usually indicate the CNF row is a prepared product
# rather than the basic ingredient we want.
UNDESIRED_TERMS = {
    "alcohol": 100,
    "sake": 100,
    "babyfood": 80,
    "soup": 70,
    "deli meat": 60,
    "frozen entree": 70,
    "fast foods": 70,
    "stuffed": 50,
    "stuffing": 50,
    "breaded": 70,
    "batter dipped": 70,
    "flour coated": 70,
    "with sauce": 50,
    "pudding": 70,
    "cereal": 60,
    "snacks": 60,
    "candies": 80,
    "dessert": 70,
    "beverage": 80,
    "juice": 60,
}


PREFERRED_TERMS = {
    "cooked": 10,
    "boiled": 8,
    "roasted": 8,
    "grilled": 8,
    "plain": 8,
    "regular": 5,
    "meat only": 8,
    "skinless": 5,
    "boneless": 5,
}

def normalize_text(text):
    """
    Converts text into a cleaner format for comparison.

    Example:
    'Chicken, Broiler, Breast'
        ->
    'chicken broiler breast'
    """

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

@lru_cache(maxsize=500)
# this function will pick the closest matching food based on description and matching name to the ingreident that the user wants by 
# ranking by the score of the matching words and the similarity of the description to the query
def resolve_food(food_name):

    query = normalize_text(food_name)
    query_words = set(query.split())

    candidates = []

    for _, row in food_df.iterrows():

        description = row["Food_Description_EN"]

        normalized_description = normalize_text(
            description
        )

        description_words = set(
            normalized_description.split()
        )

        # -----------------------------------
        # 1. All query words must be present
        # -----------------------------------

        if not query_words.issubset(description_words):
            continue

        score = 0

        # -----------------------------------
        # 2. Reward query word matches
        # -----------------------------------

        score += len(query_words) * 20

        # Exact phrase is useful,
        # but should NOT dominate everything.
        if query in normalized_description:
            score += 10

        # -----------------------------------
        # 3. Reward similarity
        # -----------------------------------

        similarity = SequenceMatcher(
            None,
            query,
            normalized_description
        ).ratio()

        score += similarity * 15

        # -----------------------------------
        # 4. Reward normal ingredient forms
        # -----------------------------------

        for term, bonus in PREFERRED_TERMS.items():

            normalized_term = normalize_text(term)

            if normalized_term in normalized_description:
                score += bonus

        # -----------------------------------
        # 5. Strongly penalize composite /
        #    processed foods
        # -----------------------------------

        for term, penalty in UNDESIRED_TERMS.items():

            normalized_term = normalize_text(term)

            # Don't penalize it if the user
            # explicitly asked for it
            if (
                normalized_term in normalized_description
                and normalized_term not in query
            ):
                score -= penalty

        # -----------------------------------
        # 6. Prefer simpler descriptions
        # -----------------------------------

        extra_words = (
            len(description_words)
            - len(query_words)
        )

        score -= extra_words * 0.5

        candidates.append({
            "food_code": int(row["Food_Code"]),
            "description": description,
            "score": round(score, 2),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[0]


def get_nutrition(food_name):
    resolved = resolve_food(food_name)

    if resolved is None:
        return None

    food_code = resolved["food_code"]

    food_nutrients = nutrient_amount_df[
        nutrient_amount_df["Food_Code"] == food_code
    ]

    protein_row = food_nutrients[
        food_nutrients["Nutrient_Code"] == PROTEIN_CODE
    ]

    calorie_row = food_nutrients[
        food_nutrients["Nutrient_Code"] == CALORIE_CODE
    ]

    if protein_row.empty or calorie_row.empty:
        return None

    protein = protein_row["Nutrient_Amount"].iloc[0]
    calories = calorie_row["Nutrient_Amount"].iloc[0]

    return {
        "calories_per_100g": float(calories),
        "protein_per_100g": float(protein),
        "resolved_food": resolved["description"],
        "food_code": food_code,
    }

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


if __name__ == "__main__":

    foods = [
        "chicken breast",
        "rice",
        "broccoli",
        "tofu",
        "egg",
        "salmon",
        "potato",
        "carrot",
        "beef",
        "spinach",
    ]

    for food in foods:
        print(food, "->", resolve_food(food))