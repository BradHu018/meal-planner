"""Offline regression tests: python -m unittest debug.rag_integration_test -v."""
import copy
import unittest
from unittest.mock import patch, Mock

import workflow as w
from main import initial_state
from rag import retriever
from langchain_core.documents import Document


def source(recipe_id="1", minutes=20, ingredients=None):
    return dict(recipe_id=recipe_id, name=f"Recipe {recipe_id}", minutes=minutes,
                ingredients=ingredients or ["rice", "tofu"], content="Rice and tofu dinner")


class RagIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.state = copy.deepcopy(initial_state)
        self.state["preferences"]["meals_needed"] = 1
        self.state.update(w.planner_node(self.state))

    def test_filter_hard_constraints_and_duplicates(self):
        self.state["retrieved_recipes"] = [
            source(), source(), source("2", 31),
            source("3", ingredients=["Black OLIVE"]),
            source("4", 30), source("5", 0),
            {**source("6"), "ingredients": []},
        ]
        result = w.deterministic_retrieval_filter_node(self.state)
        self.assertEqual([r["recipe_id"] for r in result["filtered_recipes"]], ["1", "4"])
        self.assertTrue(w.contains_disliked_food(["peanut-butter"], ["peanuts"]))
        self.assertFalse(w.contains_disliked_food(["eggplant"], ["egg"]))

    def test_insufficient_results_fail_explicitly(self):
        self.state["retrieved_recipes"] = []
        with self.assertRaisesRegex(ValueError, "Only 0"):
            w.deterministic_retrieval_filter_node(self.state)

    def test_retriever_preserves_document_and_structured_ingredients(self):
        doc = Document(page_content="Reference", metadata=dict(recipe_id="123", name="Tofu", minutes=20))
        store = Mock()
        store.similarity_search.return_value = [doc]
        with patch.object(retriever, "get_vector_store", return_value=store), patch.object(
            retriever, "recipe_ingredients", return_value={"123": ["tofu"]}
        ):
            result = retriever.retrieve_recipes("tofu dinner", k=50)
        store.similarity_search.assert_called_once_with("tofu dinner", k=50)
        self.assertEqual(result[0]["ingredients"], ["tofu"])
        self.assertEqual(result[0]["recipe_id"], "123")
        self.assertEqual(result[0]["content"], "Reference")

    def response(self, **updates):
        recipe = dict(source_recipe_id="1", cuisine="Chinese",
                      ingredients=[dict(name="tofu", grams=150)])
        recipe.update(updates)
        return w.RecipeAdaptationList(recipes=[w.RecipeAdaptation(**recipe)])

    def test_adapter_rejects_invalid_sources_and_constraints(self):
        self.state["filtered_recipes"] = [source()]
        for response in [self.response(source_recipe_id="unknown"),
                         w.RecipeAdaptationList(recipes=[self.response().recipes[0]] * 2),
                         self.response(ingredients=[dict(name="olives", grams=10)]),
                         w.RecipeAdaptationList(recipes=[])]:
            with self.subTest(response=response), patch.object(w, "init_chat_model"), patch.object(
                w, "invoke_with_retry", return_value=response
            ), self.assertRaises(ValueError):
                w.recipe_adapter_node(self.state)

    def test_adapter_restores_canonical_fields_without_model_copies(self):
        canonical = {**source(), "name": "food.com   title (original)", "minutes": 23}
        self.state["filtered_recipes"] = [canonical]
        properties = w.RecipeAdaptation.model_json_schema()["properties"]
        self.assertNotIn("name", properties)
        self.assertNotIn("cooking_time", properties)
        with patch.object(w, "init_chat_model"), patch.object(
            w, "invoke_with_retry", return_value=self.response()
        ):
            recipe = w.recipe_adapter_node(self.state)["candidate_recipes"][0]
        self.assertEqual(recipe["name"], canonical["name"])
        self.assertEqual(recipe["cooking_time"], 23)
        self.assertEqual(recipe["source_recipe_id"], "1")
        self.assertEqual(recipe["ingredients"][0]["grams"], 150)

    def test_adapter_rejects_duplicate_ids_at_correct_count(self):
        self.state["filtered_recipes"] = [source(), source("2")]
        response = w.RecipeAdaptationList(recipes=[self.response().recipes[0]] * 2)
        with patch.object(w, "init_chat_model"), patch.object(
            w, "invoke_with_retry", return_value=response
        ), self.assertRaisesRegex(ValueError, "unique source recipes"):
            w.recipe_adapter_node(self.state)

    def test_adapter_rejects_colliding_canonical_names(self):
        self.state["filtered_recipes"] = [source(), {**source("2"), "name": "Recipe 1"}]
        response = w.RecipeAdaptationList(recipes=[
            self.response().recipes[0], self.response(source_recipe_id="2").recipes[0]
        ])
        with patch.object(w, "init_chat_model"), patch.object(
            w, "invoke_with_retry", return_value=response
        ), self.assertRaisesRegex(ValueError, "unique canonical names"):
            w.recipe_adapter_node(self.state)

    def test_compiled_graph_through_real_cnf_and_portioning(self):
        # Only external retrieval/model calls are mocked; execute the actual graph.
        with patch.object(w, "retrieve_recipes", return_value=[source()]) as retrieve, patch.object(
            w, "init_chat_model"
        ), patch.object(w, "invoke_with_retry", return_value=self.response()):
            updates = list(w.graph.stream(self.state, stream_mode="updates",
                                          interrupt_after=["portion_calculator"]))
        updates = [update for update in updates if "__interrupt__" not in update]
        nodes = [next(iter(update)) for update in updates]
        self.assertEqual(nodes, ["planner", "retrieval_query", "recipe_retriever",
                               "deterministic_retrieval_filter", "recipe_adapter",
                               "nutrition_enrichment", "portion_calculator"])
        self.assertEqual(retrieve.call_args.kwargs["k"], 50)
        recipes = updates[-1]["portion_calculator"]["portioned_recipes"]
        self.assertEqual(recipes[0]["source_recipe_id"], "1")
        self.assertGreater(recipes[0]["nutrition"]["calories"], 0)


if __name__ == "__main__":
    unittest.main()
