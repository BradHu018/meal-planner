import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from data.recipes.prepare_recipes import clean_recipes, prepare


class CorpusExperimentTests(unittest.TestCase):
    def test_profiles_select_matching_isolated_paths(self):
        profiles = []
        for profile in ['5k', '20k']:
            result = subprocess.check_output([
                sys.executable, '-B', '-c',
                'from rag.config import RECIPE_FILE,VECTOR_DB_PATH,COLLECTION_NAME; '
                'import json; print(json.dumps([str(RECIPE_FILE),str(VECTOR_DB_PATH),COLLECTION_NAME]))'
            ], env={**os.environ, 'MEAL_PLANNER_CORPUS': profile}, text=True)
            profiles.append(json.loads(result))
        for left, right in zip(*profiles):
            self.assertNotEqual(left, right)
        self.assertTrue(profiles[0][0].endswith('/recipes_rag.csv'))
        self.assertTrue(profiles[1][0].endswith('/recipes_rag_20k.csv'))

    def test_original_cleaning_and_reproducible_nested_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows = [dict(id=i, name=f'recipe {i}', minutes=20, tags="['main-dish']",
                         description='', ingredients="['rice']", steps="['cook']") for i in range(20)]
            rows += [{**rows[0], 'id': 100}, {**rows[1], 'name': 'too long', 'minutes': 181},
                     {**rows[2], 'name': 'zero', 'minutes': 0},
                     {**rows[3], 'name': 'invalid', 'ingredients': 'not a list'}]
            raw = folder / 'raw.csv'
            pd.DataFrame(rows).to_csv(raw, index=False)
            clean = clean_recipes(raw)
            self.assertEqual(len(clean), 20)
            small = prepare(raw, folder / 'small.csv', 5)
            large = prepare(raw, folder / 'large.csv', 10)
            repeated = prepare(raw, folder / 'repeat.csv', 5)
            self.assertEqual(small['id'].tolist(), repeated['id'].tolist())
            self.assertTrue(set(small['id']).issubset(set(large['id'])))
            with self.assertRaises(FileExistsError):
                prepare(raw, folder / 'small.csv', 5)

    def test_invalid_profile_fails(self):
        result = subprocess.run([sys.executable, '-B', '-c', 'import rag.config'],
                                env={**os.environ, 'MEAL_PLANNER_CORPUS': 'typo'}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
