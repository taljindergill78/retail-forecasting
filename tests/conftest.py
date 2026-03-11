"""
Pytest fixtures shared across tests.

Used by tests/serve/test_schema.py and any future serve tests.
"""
import pytest


@pytest.fixture
def valid_store_week_row():
    """
    One valid store-week row matching StoreWeekRow schema (all required fields).
    Optional fields (lags, rolling) are omitted; Pydantic allows None for those.
    """
    return {
        "store_id": 1,
        "dept_id": 1,
        "store_type": "A",
        "store_size": 151315,
        "week_of_year": 5,
        "month": 2,
        "year": 2010,
        "day_of_year": 36,
        "isholiday": False,
        "is_holiday_prev1": 0,
        "is_holiday_next1": 0,
        "within_7days_after_holiday": 0,
        "within_7days_before_holiday": 0,
        "temperature": 42.31,
        "fuel_price": 2.572,
        "cpi": 211.0963,
        "unemployment": 8.106,
        "markdown1": 0.0,
        "markdown2": 0.0,
        "markdown3": 0.0,
        "markdown4": 0.0,
        "markdown5": 0.0,
        "markdown_total": 0.0,
        "has_markdown": 0,
    }
