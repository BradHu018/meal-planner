"""Original Food.com cleaning rules, with reproducible sample options."""
import argparse
import ast
from pathlib import Path
import pandas as pd


def parse_list(value):

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return []


def clean_recipes(input_file):
    # 1. Load raw recipes
    df = pd.read_csv(input_file)

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


    return df


def prepare(input_file, output_file, sample_size=5000, seed=42):
    output = Path(output_file)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    df = clean_recipes(input_file)
    if len(df) < sample_size:
        raise ValueError(f"Only {len(df)} cleaned recipes; {sample_size} requested")
    sample = df.sample(n=sample_size, random_state=seed)
    sample.to_csv(output, index=False)
    print(f"Saved {len(sample)} recipes to {output}")
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/recipes/RAW_recipes.csv")
    parser.add_argument("--output", default="data/recipes/recipes_rag.csv")
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output, args.size, args.seed)
