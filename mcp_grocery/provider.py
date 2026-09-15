"""External grocery-data providers.

Providers only retrieve product and price observations. They never calculate a
meal-plan cost or infer a unit conversion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
from typing import Any

import requests

try:  # ``mcp run path/to/server.py`` loads the file outside its package.
    from .schemas import GroceryProduct, GroceryQuote, GrocerySearchResponse
except ImportError:  # pragma: no cover - exercised by the stdio CLI process
    from schemas import GroceryProduct, GroceryQuote, GrocerySearchResponse


class GroceryProviderError(RuntimeError):
    """A provider could not return a trustworthy response."""


class GroceryProvider(ABC):
    name: str

    @abstractmethod
    def search_products(self, ingredient: str) -> GrocerySearchResponse:
        """Return factual product candidates in provider ranking order."""

    @abstractmethod
    def get_quote(self, product_id: str) -> GroceryQuote:
        """Return the most recent observed quote for a searched product."""


class UnconfiguredProvider(GroceryProvider):
    """Normal runtime behavior when no provider credential is configured."""

    name = "unconfigured"

    def search_products(self, ingredient: str) -> GrocerySearchResponse:
        return GrocerySearchResponse(
            ingredient=ingredient,
            products=[],
            unresolved_reason="provider_not_configured",
        )

    def get_quote(self, product_id: str) -> GroceryQuote:
        return GroceryQuote(
            product_id=product_id,
            product_name="Unknown product",
            provider=self.name,
            source=self.name,
            unresolved_reason="provider_not_configured",
        )


class VynnProvider(GroceryProvider):
    """Adapter for Vynn's documented Canadian grocery-price API.

    Vynn documents search separately from its observed price history. Search
    responses are cached for the server session so a later quote call can keep
    the package metadata associated with the selected product ID.
    """

    name = "vynn"
    base_url = "https://vynn.ai"

    def __init__(
        self,
        api_key: str,
        province: str = "ON",
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("A Vynn API key is required")
        self.api_key = api_key
        self.province = province.upper()
        self.session = session or requests.Session()
        self._searched_products: dict[str, GroceryProduct] = {}

    @classmethod
    def from_environment(cls) -> GroceryProvider:
        api_key = os.getenv("VYNN_API_KEY")
        if not api_key:
            return UnconfiguredProvider()
        return cls(api_key=api_key, province=os.getenv("VYNN_PROVINCE", "ON"))

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise GroceryProviderError("provider_request_failed") from exc
        if not isinstance(payload, dict):
            raise GroceryProviderError("provider_invalid_response")
        if payload.get("error"):
            raise GroceryProviderError("provider_error_response")
        return payload

    @staticmethod
    def _first_present(item: dict[str, Any], *names: str) -> Any:
        for name in names:
            value = item.get(name)
            if value is not None:
                return value
        return None

    def _product_from_payload(self, item: dict[str, Any]) -> GroceryProduct | None:
        product_id = self._first_present(item, "product_id", "id", "sku")
        product_name = self._first_present(item, "product_name", "name", "title")
        package_quantity = self._first_present(
            item, "package_quantity", "package_size", "size", "quantity"
        )
        package_unit = self._first_present(item, "package_unit", "unit", "size_unit")
        if not product_id or not product_name:
            return None
        try:
            return GroceryProduct(
                product_id=str(product_id),
                product_name=str(product_name),
                package_quantity=package_quantity,
                package_unit=package_unit,
                provider=self.name,
                source="Vynn Canadian grocery price API",
            )
        except (TypeError, ValueError):
            return None

    def search_products(self, ingredient: str) -> GrocerySearchResponse:
        payload = self._request(
            "/v1/products/search",
            {"q": ingredient, "province": self.province, "limit": 10},
        )
        raw_products = self._first_present(payload, "products", "results", "data")
        if not isinstance(raw_products, list):
            raise GroceryProviderError("provider_invalid_search_response")
        products = [
            product
            for item in raw_products
            if isinstance(item, dict)
            if (product := self._product_from_payload(item)) is not None
        ]
        self._searched_products.update({product.product_id: product for product in products})
        return GrocerySearchResponse(
            ingredient=ingredient,
            products=products,
            unresolved_reason=None if products else "provider_no_match",
        )

    def get_quote(self, product_id: str) -> GroceryQuote:
        product = self._searched_products.get(product_id)
        if product is None:
            return GroceryQuote(
                product_id=product_id,
                product_name="Unknown product",
                provider=self.name,
                source="Vynn Canadian grocery price API",
                unresolved_reason="product_not_in_server_search_session",
            )
        payload = self._request(
            "/v1/prices/history",
            {"product_id": product_id, "province": self.province, "period": "7d"},
        )
        observations = self._first_present(payload, "observations", "prices", "data")
        if not isinstance(observations, list) or not observations:
            return GroceryQuote(
                **product.model_dump(),
                unresolved_reason="provider_no_price_observation",
            )
        latest = next((row for row in reversed(observations) if isinstance(row, dict)), None)
        if latest is None:
            raise GroceryProviderError("provider_invalid_price_response")
        price = self._first_present(latest, "price", "amount", "current_price")
        observed_at = self._first_present(latest, "observed_at", "timestamp", "date")
        try:
            timestamp = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00")) if observed_at else datetime.now(timezone.utc)
            return GroceryQuote(
                product_id=product.product_id,
                product_name=product.product_name,
                package_quantity=product.package_quantity,
                package_unit=product.package_unit,
                price=price,
                currency=str(self._first_present(latest, "currency") or "CAD"),
                provider=self.name,
                source="Vynn Canadian grocery price API",
                retrieval_timestamp=timestamp,
                unresolved_reason=None,
            )
        except (TypeError, ValueError):
            return GroceryQuote(
                **product.model_dump(),
                unresolved_reason="provider_invalid_price_response",
            )
