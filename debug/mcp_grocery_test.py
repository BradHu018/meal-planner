"""Offline MCP and selected-grocery pricing regression tests."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from decimal import Decimal
import os
import unittest
from unittest.mock import patch

from mcp import Client

from data import grocery_pricing as pricing
from main import initial_state
from mcp_grocery.provider import GroceryProvider, GroceryProviderError, VynnProvider
from mcp_grocery.client import GroceryMCPClient
from mcp_grocery.schemas import GroceryProduct, GroceryQuote, GrocerySearchResponse
from mcp_grocery.server import create_server
import workflow as w
from debug.rag_integration_test import source


def quote(
    product_id: str = "tofu-454",
    product_name: str = "Firm tofu 454 g",
    quantity: str = "454",
    unit: str = "g",
    price: str = "3.49",
    currency: str = "CAD",
) -> GroceryQuote:
    return GroceryQuote(
        product_id=product_id,
        product_name=product_name,
        package_quantity=Decimal(quantity),
        package_unit=unit,
        price=Decimal(price),
        currency=currency,
        provider="test-provider",
        source="test source",
        retrieval_timestamp=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )


class FakeMCPClient:
    def __init__(self, quotes: list[GroceryQuote] | None = None, reason: str | None = None):
        self.quotes = quotes or []
        self.reason = reason
        self.calls: list[str] = []

    def lookup(self, ingredient: str):
        self.calls.append(ingredient)
        products = [
            GroceryProduct(
                product_id=item.product_id,
                product_name=item.product_name,
                package_quantity=item.package_quantity,
                package_unit=item.package_unit,
                provider=item.provider,
                source=item.source,
            )
            for item in self.quotes
        ]
        return GrocerySearchResponse(
            ingredient=ingredient, products=products, unresolved_reason=self.reason
        ), self.quotes


class FakeProvider(GroceryProvider):
    name = "fake-provider"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.searches: list[str] = []

    def search_products(self, ingredient: str) -> GrocerySearchResponse:
        self.searches.append(ingredient)
        if self.fail:
            raise GroceryProviderError("provider_request_failed")
        return GrocerySearchResponse(
            ingredient=ingredient,
            products=[GroceryProduct(
                product_id="fake-1", product_name="Fake tofu 454 g",
                package_quantity=Decimal("454"), package_unit="g",
                provider=self.name, source="fake source",
            )],
        )

    def get_quote(self, product_id: str) -> GroceryQuote:
        if self.fail:
            raise GroceryProviderError("provider_request_failed")
        return quote(product_id="fake-1", product_name="Fake tofu 454 g")


class GroceryPricingTests(unittest.TestCase):
    def grocery(self, name="tofu", grams=700):
        return [{"name": name, "grams": grams}]

    def test_statscan_hit_does_not_call_mcp_and_records_source(self):
        client = FakeMCPClient([quote()])
        local = {"price_per_100g": 0.50, "resolved_product": "Tofu, 350 grams", "month": "2026-08"}
        with patch.object(pricing, "get_price", return_value=local):
            result = pricing.price_selected_grocery_list(self.grocery(), client)
        self.assertEqual(client.calls, [])
        self.assertEqual(result["pricing_audit"][0]["pricing_source"], "StatsCan")
        self.assertEqual(result["estimated_total"], 3.5)
        self.assertTrue(result["estimated_total_complete"])

    def test_statscan_miss_calls_mcp_with_structured_provenance(self):
        client = FakeMCPClient([quote()])
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(self.grocery(), client)
        item = result["pricing_audit"][0]
        self.assertEqual(client.calls, ["tofu"])
        self.assertEqual(item["pricing_source"], "MCP")
        self.assertEqual(item["external_product_id"], "tofu-454")
        self.assertEqual(item["matched_product"], "Firm tofu 454 g")
        self.assertEqual(item["normalized_package_grams"], 454.0)
        self.assertEqual(item["retrieval_timestamp"], "2026-09-15T00:00:00+00:00")

    def test_duplicate_grocery_ingredients_are_aggregated(self):
        recipes = [
            {"ingredients": [{"name": "Tofu", "grams": 300}]},
            {"ingredients": [{"name": "tofu", "grams": 400}]},
        ]
        self.assertEqual(w.build_grocery_list(recipes, []), [{"name": "Tofu", "grams": 700}])

    def test_package_rounding_and_checkout_cost_are_deterministic(self):
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(self.grocery(), FakeMCPClient([quote()]))
        item = result["pricing_audit"][0]
        self.assertEqual(item["packages_required"], 2)
        self.assertEqual(item["checkout_cost"], 6.98)
        self.assertEqual(result["estimated_total"], 6.98)
        self.assertEqual(result["consumption_total"], 5.38)

    def test_unsupported_package_unit_remains_missing(self):
        bad_quote = quote(quantity="12", unit="count")
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(self.grocery(), FakeMCPClient([bad_quote]))
        item = result["pricing_audit"][0]
        self.assertIsNone(item["checkout_cost"])
        self.assertEqual(item["unresolved_reason"], "package_unit_not_convertible_to_grams")
        self.assertEqual(result["missing_prices"], ["tofu"])
        self.assertFalse(result["estimated_total_complete"])

    def test_no_external_match_remains_missing_without_zero_price(self):
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(self.grocery(), FakeMCPClient(reason="provider_no_match"))
        item = result["pricing_audit"][0]
        self.assertEqual(item["unresolved_reason"], "provider_no_match")
        self.assertIsNone(item["checkout_cost"])
        self.assertEqual(result["estimated_total"], 0.0)
        self.assertEqual(result["missing_prices"], ["tofu"])

    def test_mcp_failure_is_graceful(self):
        class FailingClient:
            def lookup(self, ingredient):
                raise RuntimeError("stdio failed")
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(self.grocery(), FailingClient())
        self.assertEqual(result["pricing_audit"][0]["unresolved_reason"], "mcp_connection_failed")
        self.assertFalse(result["estimated_total_complete"])

    def test_unsupported_currency_is_not_converted_or_guessed(self):
        usd_quote = quote(currency="USD")
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(self.grocery(), FakeMCPClient([usd_quote]))
        self.assertEqual(result["pricing_audit"][0]["unresolved_reason"], "unsupported_currency")

    def test_runtime_without_provider_key_is_explicitly_unresolved(self):
        with patch.dict(os.environ, {}, clear=True):
            search, quotes = GroceryMCPClient().lookup("tofu")
        self.assertEqual(quotes, [])
        self.assertEqual(search.unresolved_reason, "provider_not_configured")


class MCPServerTests(unittest.TestCase):
    def test_server_exposes_only_structured_product_and_quote_tools(self):
        provider = FakeProvider()

        async def run():
            async with Client(create_server(provider), raise_exceptions=True) as client:
                tools = await client.list_tools()
                self.assertEqual({tool.name for tool in tools.tools}, {
                    "search_grocery_products", "get_grocery_quote"
                })
                search = await client.call_tool("search_grocery_products", {"ingredient": "tofu"})
                self.assertFalse(search.is_error)
                parsed_search = GrocerySearchResponse.model_validate(search.structured_content)
                self.assertEqual(parsed_search.products[0].product_id, "fake-1")
                fetched_quote = await client.call_tool("get_grocery_quote", {"product_id": "fake-1"})
                parsed_quote = GroceryQuote.model_validate(fetched_quote.structured_content)
                self.assertEqual(parsed_quote.price, Decimal("3.49"))

        asyncio.run(run())
        self.assertEqual(provider.searches, ["tofu"])


class MCPSubprocessStdioTests(unittest.TestCase):
    """Exercise the production client boundary against a real child process."""

    def stdio_client(self) -> GroceryMCPClient:
        return GroceryMCPClient(
            server_module="debug.mcp_stdio_fixture_server",
            require_api_key=False,
        )

    def test_real_stdio_client_initializes_lists_tools_and_returns_quote(self):
        client = self.stdio_client()
        search, quotes = client.lookup("tofu")

        self.assertEqual(client.last_protocol_version, "2025-11-25")
        self.assertEqual(set(client.last_discovered_tools), {
            "search_grocery_products", "get_grocery_quote",
        })
        self.assertIsNone(search.unresolved_reason)
        self.assertEqual(search.products[0].product_id, "tofu-454")
        self.assertEqual(quotes[0].product_name, "Firm tofu 454 g")
        self.assertEqual(quotes[0].price, Decimal("3.49"))
        self.assertEqual(quotes[0].currency, "CAD")

    def test_statscan_miss_uses_real_stdio_server_then_python_package_math(self):
        with patch.object(pricing, "get_price", return_value=None):
            result = pricing.price_selected_grocery_list(
                [{"name": "tofu", "grams": 700}], self.stdio_client()
            )

        item = result["pricing_audit"][0]
        self.assertEqual(item["pricing_source"], "MCP")
        self.assertEqual(item["external_product_id"], "tofu-454")
        self.assertEqual(item["normalized_package_grams"], 454.0)
        self.assertEqual(item["packages_required"], 2)
        self.assertEqual(item["checkout_cost"], 6.98)
        self.assertTrue(result["estimated_total_complete"])


class VynnProviderTests(unittest.TestCase):
    def test_documented_search_and_price_calls_are_mocked(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.calls = []
                self.responses = [
                    Response({"products": [{
                        "product_id": "vynn-tofu", "product_name": "Firm tofu 454 g",
                        "package_quantity": "454", "package_unit": "g",
                    }]}),
                    Response({"observations": [{
                        "price": "3.49", "currency": "CAD", "observed_at": "2026-09-15T00:00:00Z",
                    }]}),
                ]

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return self.responses.pop(0)

        session = Session()
        provider = VynnProvider("test-key", session=session)
        search = provider.search_products("tofu")
        fetched = provider.get_quote(search.products[0].product_id)
        self.assertEqual(fetched.product_id, "vynn-tofu")
        self.assertEqual(fetched.price, Decimal("3.49"))
        self.assertEqual(session.calls[0][0], "https://vynn.ai/v1/products/search")
        self.assertEqual(session.calls[1][0], "https://vynn.ai/v1/prices/history")
        self.assertEqual(session.calls[0][1]["headers"]["Authorization"], "Bearer test-key")

    def test_server_turns_provider_failure_into_unresolved_data(self):
        async def run():
            async with Client(create_server(FakeProvider(fail=True)), raise_exceptions=True) as client:
                result = await client.call_tool("search_grocery_products", {"ingredient": "tofu"})
                parsed = GrocerySearchResponse.model_validate(result.structured_content)
                self.assertEqual(parsed.unresolved_reason, "provider_request_failed")
        asyncio.run(run())


class FullGraphMCPFallbackTests(unittest.TestCase):
    def test_full_graph_finalizes_with_mocked_mcp_fallback(self):
        state = copy.deepcopy(initial_state)
        state["preferences"]["meals_needed"] = 1
        state["preferences"]["max_cooking_time"] = 5
        good = source("mcp", minutes=5, ingredients=["tofu"])
        calls = {"critic": 0}

        def response(schema, messages):
            if schema is w.RetrievalGrade:
                return w.RetrievalGrade(top_alignment=4, meal_suitability=4, diversity=3, sufficient=True, reason="Good")
            if schema is w.RecipeAdaptationList:
                return w.RecipeAdaptationList(recipes=[w.RecipeAdaptation(
                    source_recipe_id="mcp", cuisine="Chinese",
                    ingredients=[w.Ingredient(name="tofu", grams=700)],
                )])
            if schema is w.TasteAnalysis:
                return w.TasteAnalysis(evaluations=[])
            if schema is w.BalanceAnalysis:
                return w.BalanceAnalysis(evaluations=[], overall_summary="Test", repetition_concerns=[], optimizer_suggestions=[])
            if schema is w.OptimizerSelection:
                return w.OptimizerSelection(selected_meals=[w.SelectedMeal(recipe_name=good["name"], reason="Test")], reasoning="Test")
            if schema is w.CriticReview:
                calls["critic"] += 1
                return w.CriticReview(revision_problems=[], warnings=[], suggestions=[], summary="Approved")
            raise AssertionError(schema)

        def selected_price(groceries):
            with patch.object(pricing, "get_price", return_value=None):
                return pricing.price_selected_grocery_list(groceries, FakeMCPClient([quote()]))

        with patch.object(w, "retrieve_recipes", return_value=[good]), patch.object(
            w, "init_chat_model"
        ) as model, patch.object(w, "invoke_with_retry", side_effect=response), patch.object(
            w, "price_selected_grocery_list", side_effect=selected_price
        ), patch.object(w, "save_state"):
            model.return_value.with_structured_output.side_effect = lambda schema: schema
            result = w.graph.invoke(state)

        self.assertTrue(result["final_result"]["approved"])
        self.assertTrue(result["final_result"]["estimated_total_complete"])
        self.assertEqual(result["final_result"]["pricing_audit"][0]["pricing_source"], "MCP")
        self.assertEqual(result["final_result"]["estimated_total"], 6.98)


if __name__ == "__main__":
    unittest.main()
