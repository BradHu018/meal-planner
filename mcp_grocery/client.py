"""Application-side MCP client for the local grocery server."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

from mcp import Client

from .schemas import GroceryQuote, GrocerySearchResponse
from .stdio import stdio_client_transport


class GroceryMCPClient:
    """Call the grocery server through stdio, never by importing its provider."""

    def __init__(self, project_root: Path | None = None, server_module: str = "mcp_grocery.server", require_api_key: bool = True) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.server_module = server_module
        self.require_api_key = require_api_key
        # Kept for connection diagnostics and the real-transport integration
        # test. These values are populated only after a successful handshake.
        self.last_protocol_version: str | None = None
        self.last_discovered_tools: tuple[str, ...] = ()

    def lookup(self, ingredient: str) -> tuple[GrocerySearchResponse, list[GroceryQuote]]:
        if self.require_api_key and not os.getenv("VYNN_API_KEY"):
            return (
                GrocerySearchResponse(
                    ingredient=ingredient,
                    products=[],
                    unresolved_reason="provider_not_configured",
                ),
                [],
            )
        try:
            return asyncio.run(self._lookup(ingredient))
        except Exception:
            return (
                GrocerySearchResponse(
                    ingredient=ingredient,
                    products=[],
                    unresolved_reason="mcp_connection_failed",
                ),
                [],
            )

    async def _lookup(self, ingredient: str) -> tuple[GrocerySearchResponse, list[GroceryQuote]]:
        transport = stdio_client_transport(sys.executable, ["-m", self.server_module], self.project_root)
        async with Client(transport, mode="legacy", read_timeout_seconds=15) as client:
            self.last_protocol_version = client.protocol_version
            tools = await client.list_tools()
            self.last_discovered_tools = tuple(tool.name for tool in tools.tools)
            search_result = await client.call_tool(
                "search_grocery_products", {"ingredient": ingredient}
            )
            if search_result.is_error or not search_result.structured_content:
                return GrocerySearchResponse(
                    ingredient=ingredient,
                    products=[],
                    unresolved_reason="mcp_search_failed",
                ), []
            search = GrocerySearchResponse.model_validate(search_result.structured_content)
            quotes: list[GroceryQuote] = []
            for product in search.products:
                quote_result = await client.call_tool(
                    "get_grocery_quote", {"product_id": product.product_id}
                )
                if quote_result.is_error or not quote_result.structured_content:
                    continue
                quotes.append(GroceryQuote.model_validate(quote_result.structured_content))
            return search, quotes
