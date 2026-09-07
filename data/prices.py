import pandas as pd 
import re 
from difflib import SequenceMatcher
from functools import lru_cache

PRICE_FILE = "data/cnf/food_price.csv"

price_df = pd.read_csv(
    PRICE_FILE, 
    skiprows=9
)

# remove dollar units row 

price_df = price_df.iloc[1:]
price_df = price_df.dropna(
    how="all"
).reset_index(drop=True)

data_columns = [
    column 
    for column in price_df.columns 
    if column != "Products"
]

for column in data_columns:
    price_df[column] = pd.to_numeric(
        price_df[column],
        errors="coerce"
    )

latest_month = data_columns[-1]


def normalize_text(text):
    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# remove plural in words
def normalize_word(word):

    if len(word) > 3 and word.endswith("s"):
        return word[:-1]

    return word


# remove trailing numbers on products
def clean_product_name(name):

    name = str(name)

    # Remove trailing StatsCan footnote number
    name = re.sub(
        r"\s+\d+\s*$",
        "",
        name
    )

    return name.strip()

# Actually clean the dataframe
price_df["Products"] = (
    price_df["Products"]
    .apply(clean_product_name)
)


def convert_to_price_per_100g(product_name, price):

    name = normalize_text(product_name)

    # Example:
    # Chicken breasts, per kilogram
    if "per kilogram" in name:
        return price / 10

    # Example:
    # Bacon, 500 grams
    gram_match = re.search(
        r"(\d+(?:\.\d+)?) grams?",
        name
    )

    if gram_match:

        grams = float(
            gram_match.group(1)
        )

        return price / (grams / 100)

    # Example:
    # Rice, 2 kilograms
    kg_match = re.search(
        r"(\d+(?:\.\d+)?) kilograms?",
        name
    )

    if kg_match:

        kilograms = float(
            kg_match.group(1)
        )

        grams = kilograms * 1000

        return price / (grams / 100)

    return None


@lru_cache(maxsize=500)
def resolve_price_product(food_name):

    query = normalize_text(food_name)

    candidates = []

    query_words = {
        normalize_word(word)
        for word in query.split()
    }

    for _, row in price_df.iterrows():

        product = row["Products"]

        if pd.isna(product):
            continue

        price = row[latest_month]

        if pd.isna(price):
            continue

        normalized_product = normalize_text(
            product
        )

        product_words = {
            normalize_word(word)
            for word in normalized_product.split()
        }

        similarity = SequenceMatcher(
            None,
            query,
            normalized_product
        ).ratio()

        score = similarity * 30

        matching_words = len(
            query_words & product_words
        )

        score += matching_words * 20

        if query in normalized_product:
            score += 25

        candidates.append({
            "product": product,
            "price": float(price),
            "month": latest_month,
            "score": round(score, 2),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[0]

def get_price(food_name):

    result = resolve_price_product(
        food_name
    )

    if result is None:
        return None

    price_per_100g = (
        convert_to_price_per_100g(
            result["product"],
            result["price"]
        )
    )

    if price_per_100g is None:
        return None

    return {
        "price_per_100g":
            round(price_per_100g, 4),

        "resolved_product":
            result["product"],

        "original_price":
            result["price"],

        "month":
            result["month"],
    }

def calculate_ingredient_cost(food_name, grams):

    data = get_price(food_name)

    if data is None:
        return None

    multiplier = grams / 100

    return data["price_per_100g"] * multiplier


def calculate_recipe_cost(recipe, pantry):

    total = 0
    missing_prices = []

    for ingredient in recipe["ingredients"]:

        name = ingredient["name"]

        if name in pantry:
            continue

        cost = calculate_ingredient_cost(
            name,
            ingredient["grams"]
        )

        if cost is None:
            missing_prices.append(name)
            continue

        total += cost

    return {
        "estimated_cost": round(total, 2),
        "missing_prices": missing_prices,
    }


if __name__  == "__main__":
    # print("COLUMNS:")
    # print(price_df.columns)

    # print("\nLATEST MONTH:")
    # print(latest_month)

    # print("\nPRODUCTS + LATEST PRICES:")
    # print(
    #     price_df[
    #         ["Products", latest_month]
    #     ].head(30)
    # )

    # foods = [
    #     "chicken breast",
    #     "salmon",
    #     "bacon",
    #     "shrimp",
    #     "rice",
    # ]

    # for food in foods:
    #     print(food, "->", get_price(food))

    print(resolve_price_product("broccoli"))
    print(get_price("broccoli"))
    
    test_recipe = {
        "name": "Test Chicken Bowl",
        "ingredients": [
            {"name": "chicken breast", "grams": 200},
            {"name": "rice", "grams": 150},
            {"name": "broccoli", "grams": 100},
        ]
    }

    print(
        calculate_recipe_cost(
            test_recipe,
            pantry=[]
        )
    )
  