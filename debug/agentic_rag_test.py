"""Bounded agentic retrieval tests using real graph edges and mocked LLM calls."""
import copy
import unittest
from unittest.mock import patch

import workflow as w
from main import initial_state
from debug.rag_integration_test import source
from debug import rag_evaluation as evaluation


class AgenticRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.state = copy.deepcopy(initial_state)
        self.state['preferences']['meals_needed'] = 1
        self.state['preferences']['disliked_foods'] = ['soy sauce']
        self.state['preferences']['max_cooking_time'] = 5
        self.good = source('good', minutes=5, ingredients=['tofu'])
        self.bad = source('bad', minutes=6, ingredients=['soy sauce'])

    def run_graph(self, pools, grades, adapter_id='good'):
        trace = []
        responses = iter(grades)
        def answer(schema, messages):
            if schema is w.RetrievalGrade:
                return next(responses)
            if schema is w.RewrittenQuery:
                return w.RewrittenQuery(query='quick tofu main dish')
            if schema is w.RecipeAdaptationList:
                return w.RecipeAdaptationList(recipes=[w.RecipeAdaptation(
                    source_recipe_id=adapter_id, cuisine='Chinese',
                    ingredients=[w.Ingredient(name='tofu', grams=150)])])
            raise AssertionError(f'Unexpected model schema: {schema}')
        with patch.object(w, 'retrieve_recipes', side_effect=pools) as retrieve, patch.object(
            w, 'init_chat_model'
        ) as model, patch.object(w, 'invoke_with_retry', side_effect=answer):
            model.return_value.with_structured_output.side_effect = lambda schema: schema
            state = copy.deepcopy(self.state)
            error = None
            try:
                for update in w.graph.stream(state, stream_mode='updates', interrupt_after=['nutrition_enrichment']):
                    for node, values in update.items():
                        if node != '__interrupt__':
                            trace.append(node)
                            state.update(values or {})
            except w.RetrievalFailure as exc:
                error = str(exc)
            return state, trace, error, retrieve.call_count

    def test_good_pool_goes_directly_to_adapter_and_keeps_source(self):
        state, nodes, error, calls = self.run_graph([[self.good]], [w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=True, reason='Aligned')])
        self.assertIsNone(error)
        self.assertNotIn('query_rewriter', nodes)
        self.assertEqual(calls, 1)
        self.assertEqual(state['retrieval_attempts'], 1)
        self.assertEqual(state['enriched_recipes'][0]['source_recipe_id'], 'good')

    def test_low_pool_rewrites_returns_to_retriever_and_succeeds(self):
        state, nodes, error, calls = self.run_graph([[self.bad], [self.good]], [w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=True, reason='Aligned')])
        self.assertIsNone(error)
        self.assertEqual(calls, 2)
        self.assertEqual(nodes[nodes.index('query_rewriter') + 1], 'recipe_retriever')
        self.assertEqual(state['retrieval_attempts'], 2)
        self.assertEqual(len(state['retrieval_query_history']), 2)
        self.assertEqual([r['recipe_id'] for r in state['filtered_recipes']], ['good'])
        self.assertIn('recipe_adapter', nodes)
        self.assertEqual(state['planning_constraints']['avoid'], ['soy sauce'])
        self.assertEqual(state['planning_constraints']['max_cooking_time'], 5)

    def test_semantically_poor_pool_rewrites(self):
        state, nodes, error, calls = self.run_graph([[self.good], [self.good]], [
            w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=False, reason='Off topic', rewrite_focus='Main dishes'),
            w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=True, reason='Aligned'),
        ])
        self.assertIsNone(error)
        self.assertEqual(nodes.count('query_rewriter'), 1)
        self.assertEqual(calls, 2)

    def test_sparse_pool_fails_after_exactly_three_attempts(self):
        state, nodes, error, calls = self.run_graph([[self.bad]] * 3, [])
        self.assertIn('after 3 attempts', error)
        self.assertIn('Hard constraints were not relaxed', error)
        self.assertEqual(calls, 3)
        self.assertEqual(nodes.count('query_rewriter'), 2)
        self.assertNotIn('recipe_adapter', nodes)
        self.assertEqual(state['filtered_recipes'], [])
        self.assertEqual(state['retrieval_attempts'], 3)

    def test_semantic_failure_also_stops_at_limit(self):
        state, nodes, error, calls = self.run_graph([[self.good]] * 3, [
            w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=False, reason='Not relevant') for _ in range(3)
        ])
        self.assertIn('Not relevant', error)
        self.assertEqual(calls, 3)
        self.assertNotIn('recipe_adapter', nodes)

    def test_rewriter_only_updates_query_and_retains_constraint_text(self):
        state = {**self.state, **w.planner_node(self.state), 'retrieval_attempts': 1,
                 'retrieval_query': 'Korean tofu dinner', 'retrieval_query_history': ['Korean tofu dinner'],
                 'retrieval_feedback': {'reason': 'Too sparse'}}
        before = copy.deepcopy(state)
        with patch.object(w, 'init_chat_model'), patch.object(w, 'invoke_with_retry', return_value=w.RewrittenQuery(query='tofu supper')):
            update = w.query_rewriter_node(state)
        self.assertEqual(state, before)
        self.assertEqual(set(update), {'retrieval_query'})
        self.assertIn('5 minutes', update['retrieval_query'])
        self.assertIn('Exclude ingredients: soy sauce', update['retrieval_query'])
        self.assertIn('Original intent: Korean tofu dinner', update['retrieval_query'])

    def test_later_worse_pools_restore_earlier_sufficient_sources(self):
        later = source('later', minutes=5, ingredients=['tofu'])
        state, nodes, error, calls = self.run_graph(
            [[self.good], [later], [later]], [
                w.RetrievalGrade(sufficient=True, top_alignment=3, meal_suitability=4,
                                 diversity=3, reason='Acceptable but improvable'),
                w.RetrievalGrade(sufficient=True, top_alignment=2, meal_suitability=4,
                                 diversity=3, reason='Mixed, even though model said yes'),
                w.RetrievalGrade(sufficient=False, top_alignment=1, meal_suitability=2,
                                 diversity=2, reason='Worse'),
            ])
        self.assertIsNone(error)
        self.assertEqual(calls, 3)
        self.assertEqual(state['best_retrieval']['attempt'], 1)
        self.assertEqual(state['retrieved_recipes'][0]['recipe_id'], 'good')
        self.assertEqual(state['enriched_recipes'][0]['source_recipe_id'], 'good')
        self.assertEqual(state['retrieval_query'], state['retrieval_query_history'][0])

    def test_quality_floor_rejects_unsupported_positive_verdict(self):
        state, nodes, error, calls = self.run_graph([[self.good]] * 3, [
            w.RetrievalGrade(sufficient=True, top_alignment=2, meal_suitability=4,
                             diversity=4, reason='Many recipes but mixed alignment')
            for _ in range(3)
        ])
        self.assertIsNotNone(error)
        self.assertNotIn('recipe_adapter', nodes)
        self.assertEqual(state['best_retrieval']['attempt'], 1)  # ties keep earlier

    def test_better_later_pool_is_selected(self):
        later = source('later', minutes=5, ingredients=['tofu'])
        state, nodes, error, calls = self.run_graph([[self.good], [later]], [
            w.RetrievalGrade(sufficient=True, top_alignment=3, meal_suitability=3,
                             diversity=2, reason='Acceptable'),
            w.RetrievalGrade(sufficient=True, top_alignment=4, meal_suitability=4,
                             diversity=3, reason='Strong'),
        ], adapter_id='later')
        self.assertIsNone(error)
        self.assertEqual(state['best_retrieval']['attempt'], 2)
        self.assertEqual(state['enriched_recipes'][0]['source_recipe_id'], 'later')

    def test_rewrite_guard_rejects_new_restrictions(self):
        original = 'Korean tofu dinner without soy sauce'
        for proposal in ['soy-free Korean tofu', 'tofu under 60 minutes',
                         'vegan tofu dinner', 'tofu without peanuts', 'low sodium tofu']:
            self.assertEqual(w.safe_rewrite_phrasing(proposal, original), original)
        self.assertEqual(w.safe_rewrite_phrasing('Korean bean curd supper', original),
                         'Korean bean curd supper')

    def test_count_gate_never_calls_llm(self):
        state = {**w.planner_node(self.state), 'filtered_recipes': [], 'retrieved_recipes': [], 'retrieval_query': 'test', 'retrieval_attempts': 1}
        with patch.object(w, 'init_chat_model') as model:
            result = w.retrieval_grader_node(state)
        model.assert_not_called()
        self.assertFalse(result['retrieval_sufficient'])

    def test_adapter_rejects_source_from_previous_attempt(self):
        with self.assertRaisesRegex(ValueError, 'unknown source_recipe_id'):
            self.run_graph([[self.good]], [w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=True, reason='Good')], adapter_id='old')

    def test_evaluation_labels_do_not_enter_production_state(self):
        case = dict(id='hidden', category='test', query='tofu dinner', expected_terms=['SECRET_LABEL'],
                    expected_tag_groups=[['SECRET_TAG']], max_minutes=5, forbidden_terms=['soy sauce'])
        def driver(state):
            self.assertNotIn('SECRET', str(state))
            state.update(retrieved_recipes=[], filtered_recipes=[], retrieval_attempts=3,
                         retrieval_sufficient=False, retrieval_query_history=['tofu dinner'])
            return [], 'Sparse corpus'
        with patch.object(evaluation, 'run_agentic_retrieval', side_effect=driver):
            result = evaluation.evaluate_case(case, {}, mode='agentic')
        self.assertEqual(result['retrieval_failure'], 'Sparse corpus')
        self.assertEqual(result['filtered_precision_at_5'], 0)

    def test_max_retrieval_and_existing_revision_loops_can_finalize(self):
        counts = {'grades': 0, 'critics': 0}
        def answer(schema, messages):
            if schema is w.RewrittenQuery:
                return w.RewrittenQuery(query='tofu dinner')
            if schema is w.RetrievalGrade:
                counts['grades'] += 1
                return w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=counts['grades'] == 3, reason='Quality check')
            if schema is w.RecipeAdaptationList:
                return w.RecipeAdaptationList(recipes=[w.RecipeAdaptation(source_recipe_id='good', cuisine='Chinese', ingredients=[w.Ingredient(name='tofu', grams=150)])])
            if schema is w.TasteAnalysis:
                return w.TasteAnalysis(evaluations=[])
            if schema is w.BalanceAnalysis:
                return w.BalanceAnalysis(evaluations=[], overall_summary='Test', repetition_concerns=[], optimizer_suggestions=[])
            if schema is w.OptimizerSelection:
                return w.OptimizerSelection(selected_meals=[w.SelectedMeal(recipe_name=self.good['name'], reason='Test')], reasoning='Test')
            if schema is w.CriticReview:
                counts['critics'] += 1
                return w.CriticReview(revision_problems=['Test issue'] if counts['critics'] < 3 else [], warnings=[], suggestions=[], summary='Test')
            if schema is w.RevisionSelection:
                return w.RevisionSelection(selected_recipe_names=[self.good['name']], changes_made=[], reasoning='Test')
            raise AssertionError(schema)
        with patch.object(w, 'retrieve_recipes', return_value=[self.good]), patch.object(w, 'init_chat_model') as model, patch.object(w, 'invoke_with_retry', side_effect=answer), patch.object(w, 'save_state'):
            model.return_value.with_structured_output.side_effect = lambda schema: schema
            result = w.graph.invoke(copy.deepcopy(self.state))
        self.assertEqual(result['retrieval_attempts'], 3)
        self.assertEqual(result['revision_count'], 2)
        self.assertTrue(result['final_result']['approved'])
        self.assertEqual(result['final_result']['weekly_plan'][0]['source_recipe_id'], 'good')


if __name__ == '__main__':
    unittest.main()
