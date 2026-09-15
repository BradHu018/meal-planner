"""Deterministic MCP server process used only by stdio integration tests."""

from decimal import Decimal
from datetime import datetime, timezone
import anyio

from mcp_grocery.provider import GroceryProvider
from mcp_grocery.schemas import GroceryProduct, GroceryQuote, GrocerySearchResponse
from mcp_grocery.server import create_server


class FixtureProvider(GroceryProvider):
    name = "stdio-test-provider"

    def search_products(self, ingredient: str) -> GrocerySearchResponse:
        return GrocerySearchResponse(
            ingredient=ingredient,
            products=[GroceryProduct(
                product_id="tofu-454",
                product_name="Firm tofu 454 g",
                package_quantity=Decimal("454"),
                package_unit="g",
                provider=self.name,
                source="deterministic test fixture",
            )],
        )

    def get_quote(self, product_id: str) -> GroceryQuote:
        return GroceryQuote(
            product_id=product_id,
            product_name="Firm tofu 454 g",
            package_quantity=Decimal("454"),
            package_unit="g",
            price=Decimal("3.49"),
            currency="CAD",
            provider=self.name,
            source="deterministic test fixture",
            retrieval_timestamp=datetime(2026, 9, 15, tzinfo=timezone.utc),
        )


mcp = create_server(FixtureProvider())


if __name__ == "__main__":
    from mcp_grocery.stdio import run_stdio_server

    anyio.run(run_stdio_server, mcp)
