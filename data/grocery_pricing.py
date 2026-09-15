"""Deterministic selected-grocery pricing with an MCP fallback.

Statistics Canada remains the primary source. The MCP client is contacted only
after the local resolver cannot produce a usable unit price.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from typing import Protocol

from data.prices import get_price
from mcp_grocery.client import GroceryMCPClient
from mcp_grocery.schemas import GroceryQuote, GrocerySearchResponse


class GroceryLookupClient(Protocol):
    def lookup(self, ingredient: str) -> tuple[GrocerySearchResponse, list[GroceryQuote]]:
        """Return product candidates and their quotes through MCP."""


GRAM_UNITS = {"g": Decimal("1"), "gram": Decimal("1"), "grams": Decimal("1")}
KILOGRAM_UNITS = {"kg": Decimal("1000"), "kilogram": Decimal("1000"), "kilograms": Decimal("1000")}
SAFE_GRAM_UNITS = GRAM_UNITS | KILOGRAM_UNITS


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01")))


def package_grams(quantity: Decimal | None, unit: str | None) -> Decimal | None:
    """Convert only explicit metric mass package labels to grams."""
    if quantity is None or not unit:
        return None
    multiplier = SAFE_GRAM_UNITS.get(unit.strip().lower())
    return quantity * multiplier if multiplier is not None else None


def _unresolved(ingredient: str, required_grams: Decimal, reason: str) -> dict:
    return {
        "ingredient": ingredient,
        "required_grams": float(required_grams),
        "pricing_source": None,
        "matched_product": None,
        "external_product_id": None,
        "package_quantity": None,
        "package_unit": None,
        "normalized_package_grams": None,
        "package_price": None,
        "packages_required": None,
        "checkout_cost": None,
        "consumption_cost": None,
        "currency": None,
        "retrieval_timestamp": None,
        "unresolved_reason": reason,
    }


def _statscan_audit(ingredient: str, required_grams: Decimal, price_data: dict) -> dict | None:
    unit_price = Decimal(str(price_data["price_per_100g"]))
    if unit_price <= 0:
        return None
    consumption_cost = (required_grams / Decimal("100")) * unit_price
    return {
        "ingredient": ingredient,
        "required_grams": float(required_grams),
        "pricing_source": "StatsCan",
        "matched_product": price_data["resolved_product"],
        "external_product_id": None,
        "package_quantity": None,
        "package_unit": None,
        "normalized_package_grams": None,
        "package_price": None,
        "packages_required": None,
        # StatsCan supplies a normalised unit price, not a sellable package.
        "checkout_cost": _money(consumption_cost),
        "consumption_cost": _money(consumption_cost),
        "currency": "CAD",
        "retrieval_timestamp": None,
        "statscan_month": price_data["month"],
        "unresolved_reason": None,
    }


def _mcp_audit(ingredient: str, required_grams: Decimal, quote: GroceryQuote) -> dict | None:
    normalized_grams = package_grams(quote.package_quantity, quote.package_unit)
    if normalized_grams is None:
        return None
    if quote.price is None or quote.price <= 0:
        return None
    if quote.currency != "CAD":
        return None
    packages = int((required_grams / normalized_grams).to_integral_value(rounding=ROUND_CEILING))
    if packages <= 0:
        return None
    checkout_cost = quote.price * packages
    consumption_cost = quote.price * required_grams / normalized_grams
    return {
        "ingredient": ingredient,
        "required_grams": float(required_grams),
        "pricing_source": "MCP",
        "matched_product": quote.product_name,
        "external_product_id": quote.product_id,
        "package_quantity": float(quote.package_quantity),
        "package_unit": quote.package_unit,
        "normalized_package_grams": float(normalized_grams),
        "package_price": _money(quote.price),
        "packages_required": packages,
        "checkout_cost": _money(checkout_cost),
        "consumption_cost": _money(consumption_cost),
        "currency": quote.currency,
        "retrieval_timestamp": quote.retrieval_timestamp.isoformat() if quote.retrieval_timestamp else None,
        "unresolved_reason": None,
    }


def price_selected_grocery_list(
    grocery_list: list[dict],
    mcp_client: GroceryLookupClient | None = None,
) -> dict:
    """Price aggregated groceries, first locally then through the MCP boundary.

    The function is deliberately synchronous because the existing LangGraph
    nodes are synchronous. The client encapsulates its stdio/async lifecycle.
    """
    audit: list[dict] = []
    client = mcp_client

    for grocery in grocery_list:
        ingredient = str(grocery["name"])
        required_grams = Decimal(str(grocery["grams"]))
        if required_grams <= 0:
            audit.append(_unresolved(ingredient, required_grams, "invalid_required_quantity"))
            continue

        local = get_price(ingredient)
        local_audit = _statscan_audit(ingredient, required_grams, local) if local else None
        if local_audit is not None:
            audit.append(local_audit)
            continue

        if client is None:
            client = GroceryMCPClient()
        try:
            search, quotes = client.lookup(ingredient)
        except Exception:
            audit.append(_unresolved(ingredient, required_grams, "mcp_connection_failed"))
            continue
        for quote in quotes:
            candidate = _mcp_audit(ingredient, required_grams, quote)
            if candidate is not None:
                audit.append(candidate)
                break
        else:
            if search.unresolved_reason:
                reason = search.unresolved_reason
            elif not quotes:
                reason = "mcp_no_quote"
            elif next((quote.unresolved_reason for quote in quotes if quote.unresolved_reason), None):
                reason = next(quote.unresolved_reason for quote in quotes if quote.unresolved_reason)
            elif any(quote.currency not in (None, "CAD") for quote in quotes):
                reason = "unsupported_currency"
            elif any(package_grams(quote.package_quantity, quote.package_unit) is None for quote in quotes):
                reason = "package_unit_not_convertible_to_grams"
            else:
                reason = "mcp_quote_not_usable"
            audit.append(_unresolved(ingredient, required_grams, reason))

    resolved = [item for item in audit if item["unresolved_reason"] is None]
    missing = [item["ingredient"] for item in audit if item["unresolved_reason"] is not None]
    checkout_total = sum(
        (Decimal(str(item["checkout_cost"])) for item in resolved),
        start=Decimal("0"),
    )
    consumption_total = sum(
        (Decimal(str(item["consumption_cost"])) for item in resolved),
        start=Decimal("0"),
    )
    return {
        "pricing_audit": audit,
        "missing_prices": sorted(missing),
        "estimated_total": _money(checkout_total),
        "consumption_total": _money(consumption_total),
        "estimated_total_complete": not missing,
        "resolved_count": len(resolved),
        "unresolved_count": len(missing),
    }
