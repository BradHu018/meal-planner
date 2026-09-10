"""Metric regression tests, without Chroma or Gemini calls."""
import unittest
from unittest.mock import patch
from debug import rag_evaluation as evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='1', name='Chinese tofu dinner', minutes=20,
                        description='Quick dinner', ingredients=['tofu', 'soy sauce'],
                        tags=['chinese', 'main-dish', 'vegetarian'])
        self.recipe = {**self.row, 'recipe_id': '1', 'content': 'reference'}
        self.case = dict(id='test', category='test', query='Chinese tofu dinner',
                         expected_tag_groups=[['chinese'], ['main-dish']],
                         expected_ingredient_groups=[['tofu']], max_minutes=30)
        self.metadata = {'1': self.row}

    def test_precision_fixed_denominator_and_unknown_id(self):
        self.assertEqual(evaluation.precision_at_5([self.recipe], self.metadata, self.case), .2)
        self.assertEqual(evaluation.precision_at_5([], self.metadata, self.case), 0)
        self.assertEqual(evaluation.precision_at_5([{'recipe_id': 'bad'}], self.metadata, self.case), 0)
        self.assertIsNone(evaluation.precision_at_5([self.recipe], self.metadata, {'query': 'nice'}))

    def test_relevance_requires_all_groups_and_hard_constraints(self):
        self.assertTrue(evaluation.relevant(self.row, self.case))
        for changes in [dict(tags=['main-dish']), dict(ingredients=['rice']), dict(minutes=31)]:
            self.assertFalse(evaluation.relevant({**self.row, **changes}, self.case))
        self.assertFalse(evaluation.relevant(self.row, {**self.case, 'forbidden_terms': ['soy sauce']}))
        self.assertFalse(evaluation.matches('eggplant', 'egg'))
        self.assertTrue(evaluation.matches('roasted peanuts', 'peanut'))

    def test_small_pool_uses_real_filter_and_records_gate(self):
        with patch.object(evaluation, 'recipe_retriever_node', return_value={'retrieved_recipes': [self.recipe]}):
            result = evaluation.evaluate_case(self.case, self.metadata)
        self.assertEqual(result['filtered_count'], 1)
        self.assertFalse(result['enough_candidates'])
        self.assertIn('Only 1', result['filter_error'])
        self.assertEqual(result['filtered_precision_at_5'], .2)
        self.assertEqual(result['time_violations'], 0)

    def test_empty_pool_and_missing_metadata(self):
        with patch.object(evaluation, 'recipe_retriever_node', return_value={'retrieved_recipes': []}):
            result = evaluation.evaluate_case(self.case, self.metadata)
        self.assertEqual(result['usable_candidate_rate'], 0)
        self.assertEqual(result['filtered_count'], 0)
        with patch.object(evaluation, 'recipe_retriever_node', return_value={'retrieved_recipes': [self.recipe]}):
            result = evaluation.evaluate_case(self.case, {})
        self.assertEqual(result['invalid_source_ids'], 1)
        self.assertEqual(result['filtered_precision_at_5'], 0)

    def test_duplicates_and_diversity(self):
        stats = evaluation.name_diversity([self.recipe, {**self.recipe, 'name': 'CHINESE tofu dinner!'}])
        self.assertEqual(stats['duplicate_name_rate'], .5)
        self.assertEqual(stats['near_duplicate_name_rate'], 0)
        self.assertEqual(evaluation.cuisine_diversity([self.recipe], self.metadata)['distinct_cuisine_tags'], 1)


if __name__ == '__main__':
    unittest.main()
