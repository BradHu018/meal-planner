"""Small MCP server exposing factual grocery products and quotes."""

from __future__ import annotations

from mcp.server import MCPServer
import anyio

try:  # Supports both normal package imports and the official ``mcp run`` CLI.
    from mcp_grocery.provider import GroceryProvider, GroceryProviderError, VynnProvider
    from mcp_grocery.schemas import GroceryQuote, GrocerySearchResponse
except ModuleNotFoundError:  # pragma: no cover - exercised by the stdio CLI process
    from provider import GroceryProvider, GroceryProviderError, VynnProvider
    from schemas import GroceryQuote, GrocerySearchResponse


def create_server(provider: GroceryProvider | None = None) -> MCPServer:
    provider = provider or VynnProvider.from_environment()
    server = MCPServer(
        name="meal-planner-grocery-pricing",
        version="1.0.0",
        instructions="Returns factual Canadian grocery product and package-price observations only.",
    )

    @server.tool(structured_output=True)
    async def search_grocery_products(ingredient: str) -> GrocerySearchResponse:
        """Search grocery products for one requested ingredient."""
        try:
            return provider.search_products(ingredient)
        except GroceryProviderError as exc:
            return GrocerySearchResponse(
                ingredient=ingredient,
                products=[],
                unresolved_reason=str(exc),
            )

    @server.tool(structured_output=True)
    async def get_grocery_quote(product_id: str) -> GroceryQuote:
        """Get the latest observed package quote for one searched product ID."""
        try:
            return provider.get_quote(product_id)
        except GroceryProviderError as exc:
            return GroceryQuote(
                product_id=product_id,
                product_name="Unknown product",
                provider=provider.name,
                source=provider.name,
                unresolved_reason=str(exc),
            )

    return server


mcp = create_server()


if __name__ == "__main__":
    from mcp_grocery.stdio import run_stdio_server

    anyio.run(run_stdio_server, mcp)
