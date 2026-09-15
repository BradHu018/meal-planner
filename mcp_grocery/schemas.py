"""Validated data passed across the grocery MCP boundary."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator


class GroceryProduct(BaseModel):
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    package_quantity: Decimal | None = Field(default=None, gt=0)
    package_unit: str | None = None
    provider: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("package_unit")
    @classmethod
    def normalise_package_unit(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else None


class GrocerySearchResponse(BaseModel):
    ingredient: str = Field(min_length=1)
    products: list[GroceryProduct] = Field(default_factory=list)
    unresolved_reason: str | None = None


class GroceryQuote(BaseModel):
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    package_quantity: Decimal | None = Field(default=None, gt=0)
    package_unit: str | None = None
    price: Decimal | None = Field(default=None, gt=0)
    currency: str | None = None
    provider: str = Field(min_length=1)
    source: str = Field(min_length=1)
    retrieval_timestamp: datetime | None = None
    unresolved_reason: str | None = None

    @field_validator("package_unit")
    @classmethod
    def normalise_quote_unit(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else None

    @field_validator("currency")
    @classmethod
    def normalise_currency(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None
