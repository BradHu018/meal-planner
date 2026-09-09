import ast
import pandas as pd


INPUT_FILE = "data/recipes/RAW_recipes.csv"
OUTPUT_FILE = "data/recipes/recipes_rag.csv"

SAMPLE_SIZE = 5000


# 1. Load raw recipes
df = pd.read_csv(INPUT_FILE)

print("Original recipes:", len(df))


# 2. Keep only useful RAG columns

df = df[
    [
        "id",
        "name",
        "minutes",
        "tags",
        "description",
        "ingredients",
        "steps",
    ]
].copy()


# 3. Remove unusable recipes

df = df.dropna(
    subset=[
        "id",
        "name",
        "minutes",
        "ingredients",
    ]
)


# Remove duplicate recipe names
df = df.drop_duplicates(
    subset=["name"]
)


# 4. Remove weird cooking times

df = df[
    (df["minutes"] > 0)
    & (df["minutes"] <= 180)
]

# 5. Convert string lists
#    into actual Python lists
def parse_list(value):

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return []


df["ingredients"] = (
    df["ingredients"].apply(parse_list)
)

df["steps"] = (
    df["steps"].apply(parse_list)
)

df["tags"] = (
    df["tags"].apply(parse_list)
)


# 6. Remove recipes that failed
#    ingredient parsing

df = df[
    df["ingredients"].map(len) > 0
]


# 7. Fill missing descriptions

df["description"] = (
    df["description"].fillna("")
)


# 8. Random sample for RAG v1

sample_size = min(
    SAMPLE_SIZE,
    len(df)
)

df = df.sample(
    n=sample_size,
    random_state=42
)


# 9. Save cleaned dataset

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved {len(df)} recipes "
    f"to {OUTPUT_FILE}"
)

print("\nCleaned columns:")
print(df.columns.tolist())

print("\nExample recipe:")
print(df.iloc[0])