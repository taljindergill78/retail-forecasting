"""
Phase C — Step 2: Strict Pydantic input/output schema for the serving API.

Separating schema from app.py keeps routing logic clean.
These models are the single source of truth for what /predict accepts and returns.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class StoreWeekRow(BaseModel):
    """
    One store-department-week observation.

    Required fields match the MLflow model's saved input schema exactly.
    Optional fields (lag and rolling features) are None when historical data
    is unavailable (e.g. the very first weeks of a new store).
    """

    # ── Core identifiers ──────────────────────────────────────────────────────
    store_id: int = Field(..., description="Store number (1–45)")
    dept_id: int = Field(..., description="Department number")
    store_type: str = Field(..., description='Store type: "A", "B", or "C"')
    store_size: int = Field(..., description="Store floor area in square feet")

    # ── Calendar ──────────────────────────────────────────────────────────────
    week_of_year: int = Field(..., ge=1, le=53, description="ISO week number (1–53)")
    month: int = Field(..., ge=1, le=12, description="Month (1–12)")
    year: int = Field(..., description="Calendar year, e.g. 2010")
    day_of_year: int = Field(..., ge=1, le=366, description="Day of year (1–366)")

    # ── Holiday flags ─────────────────────────────────────────────────────────
    isholiday: bool = Field(..., description="True if the week contains a public holiday")
    is_holiday_prev1: int = Field(..., ge=0, le=1, description="1 if the previous week was a holiday")
    is_holiday_next1: int = Field(..., ge=0, le=1, description="1 if the next week is a holiday")
    within_7days_after_holiday: int = Field(..., ge=0, le=1, description="1 if within 7 days after a holiday")
    within_7days_before_holiday: int = Field(..., ge=0, le=1, description="1 if within 7 days before a holiday")

    # ── Economic indicators ───────────────────────────────────────────────────
    temperature: float = Field(..., description="Average regional temperature (°F)")
    fuel_price: float = Field(..., description="Regional fuel price (USD/gallon)")
    cpi: float = Field(..., description="Consumer Price Index")
    unemployment: float = Field(..., description="Regional unemployment rate (%)")

    # ── Markdowns ─────────────────────────────────────────────────────────────
    markdown1: float = Field(..., description="Promotional markdown type 1 (0.0 if none)")
    markdown2: float = Field(..., description="Promotional markdown type 2 (0.0 if none)")
    markdown3: float = Field(..., description="Promotional markdown type 3 (0.0 if none)")
    markdown4: float = Field(..., description="Promotional markdown type 4 (0.0 if none)")
    markdown5: float = Field(..., description="Promotional markdown type 5 (0.0 if none)")
    markdown_total: float = Field(..., description="Sum of markdown1 through markdown5")
    has_markdown: int = Field(..., ge=0, le=1, description="1 if markdown_total > 0, else 0")

    # ── Sales lag features (optional) ─────────────────────────────────────────
    lag_1: Optional[float] = Field(None, description="Weekly sales 1 week ago")
    lag_2: Optional[float] = Field(None, description="Weekly sales 2 weeks ago")
    lag_4: Optional[float] = Field(None, description="Weekly sales 4 weeks ago")
    lag_52: Optional[float] = Field(None, description="Weekly sales 52 weeks ago (same week last year)")

    # ── Sales rolling statistics (optional) ───────────────────────────────────
    rolling_mean_4: Optional[float] = Field(None, description="4-week rolling mean of sales")
    rolling_mean_8: Optional[float] = Field(None, description="8-week rolling mean of sales")
    rolling_std_4: Optional[float] = Field(None, description="4-week rolling std of sales")
    rolling_std_8: Optional[float] = Field(None, description="8-week rolling std of sales")

    # ── Markdown total lag features (optional) ────────────────────────────────
    markdown_total_lag_1: Optional[float] = Field(None)
    markdown_total_lag_2: Optional[float] = Field(None)
    markdown_total_lag_4: Optional[float] = Field(None)
    markdown_total_lag_52: Optional[float] = Field(None)
    markdown_total_rolling_mean_4: Optional[float] = Field(None)
    markdown_total_rolling_mean_8: Optional[float] = Field(None)

    # ── Fuel price lag features (optional) ────────────────────────────────────
    fuel_price_lag_1: Optional[float] = Field(None)
    fuel_price_lag_2: Optional[float] = Field(None)
    fuel_price_lag_4: Optional[float] = Field(None)
    fuel_price_lag_52: Optional[float] = Field(None)
    fuel_price_rolling_mean_4: Optional[float] = Field(None)
    fuel_price_rolling_mean_8: Optional[float] = Field(None)

    # ── Temperature lag features (optional) ───────────────────────────────────
    temperature_lag_1: Optional[float] = Field(None)
    temperature_lag_2: Optional[float] = Field(None)
    temperature_lag_4: Optional[float] = Field(None)
    temperature_lag_52: Optional[float] = Field(None)
    temperature_rolling_mean_4: Optional[float] = Field(None)
    temperature_rolling_mean_8: Optional[float] = Field(None)


class PredictRequest(BaseModel):
    """
    Request body for POST /predict.

    Step 1 (simple version):
        rows: List[dict[str, Any]]  ← any dict, no validation

    Step 2 (this version):
        rows: List[StoreWeekRow]    ← every field named, typed, and required/optional declared
    """

    rows: List[StoreWeekRow] = Field(
        ...,
        min_length=1,
        description="One or more store-week rows to predict. Each row must supply all required fields.",
    )


class PredictResponse(BaseModel):
    """Response body from POST /predict."""

    predictions: List[float] = Field(
        ...,
        description="Predicted weekly sales for each input row, in the same order as the request.",
    )
