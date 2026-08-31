PRICE_DATA = {
    "chicken breast": {
        "price_per_100g": 2.00
    },

    "rice": {
        "price_per_100g": 0.50
    },

    "broccoli": {
        "price_per_100g": 0.80
    },

    "tofu": {
        "price_per_100g": 1.00
    },
}

def get_price(food_name):
    return PRICE_DATA.get(food_name)

def calculate_ingredient_cost(food_name, grams):

    data = get_price(food_name)

    if data is None:
        return 0

    multiplier = grams / 100

    return data["price_per_100g"] * multiplier


def calculate_recipe_cost(recipe, pantry):

    total = 0

    for ingredient in recipe["ingredients"]:

        name = ingredient["name"]

        # v0.1 simplifying assumption:
        # if it's in pantry, assume the user
        # already has enough
        if name in pantry:
            continue

        total += calculate_ingredient_cost(
            name,
            ingredient["grams"]
        )

    return round(total, 2)