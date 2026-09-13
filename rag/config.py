"""Corpus selection shared by ingestion, retrieval, and offline evaluation.

Set MEAL_PLANNER_CORPUS=20k before starting a process for the isolated experiment.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = os.getenv('MEAL_PLANNER_CORPUS', '5k')
if CORPUS not in {'5k', '20k'}:
    raise ValueError('MEAL_PLANNER_CORPUS must be 5k or 20k')
RECIPE_FILE = ROOT / 'data/recipes' / ('recipes_rag.csv' if CORPUS == '5k' else 'recipes_rag_20k.csv')
VECTOR_DB_PATH = ROOT / ('chroma_db' if CORPUS == '5k' else 'chroma_db_20k')
COLLECTION_NAME = 'meal_planner_recipes' if CORPUS == '5k' else 'meal_planner_recipes_20k'
