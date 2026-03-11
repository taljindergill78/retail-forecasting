"""
Unit tests for the serving API schema (src.serve.schema).

These tests validate that:
- StoreWeekRow accepts valid input and rejects invalid types/ranges.
- PredictRequest and PredictResponse behave as the API contract expects.

When you change schema.py (e.g. add a field, change a constraint), run:
  pytest tests/serve/test_schema.py -v
If a change breaks the contract, one or more tests will fail — that's how
unit tests "detect" regressions: they encode expected behaviour and fail
when the code no longer matches it.
"""
import pytest
from pydantic import ValidationError

from src.serve.schema import PredictRequest, PredictResponse, StoreWeekRow


# ── StoreWeekRow: valid and invalid inputs ───────────────────────────────────


def test_store_week_row_accepts_valid_row(valid_store_week_row):
    """A dict with all required fields and correct types becomes a valid StoreWeekRow."""
    row = StoreWeekRow(**valid_store_week_row)
    assert row.store_id == 1
    assert row.store_type == "A"
    assert row.week_of_year == 5
    assert row.month == 2
    assert row.markdown_total == 0.0


def test_store_week_row_rejects_wrong_type_for_store_id(valid_store_week_row):
    """store_id must be int; string or float should raise ValidationError."""
    valid_store_week_row["store_id"] = "not_a_number"
    with pytest.raises(ValidationError):
        StoreWeekRow(**valid_store_week_row)


def test_store_week_row_rejects_week_of_year_out_of_range(valid_store_week_row):
    """week_of_year must be 1–53 (ge=1, le=53)."""
    valid_store_week_row["week_of_year"] = 0
    with pytest.raises(ValidationError):
        StoreWeekRow(**valid_store_week_row)

    valid_store_week_row["week_of_year"] = 54
    with pytest.raises(ValidationError):
        StoreWeekRow(**valid_store_week_row)


def test_store_week_row_rejects_invalid_store_type(valid_store_week_row):
    """store_type is a string; we expect 'A', 'B', or 'C' (no constraint in schema, but must be str)."""
    valid_store_week_row["store_type"] = 123  # wrong type
    with pytest.raises(ValidationError):
        StoreWeekRow(**valid_store_week_row)


def test_store_week_row_accepts_optional_lags_as_none(valid_store_week_row):
    """Optional fields (lag_1, lag_52, etc.) can be omitted or None."""
    row = StoreWeekRow(**valid_store_week_row)
    assert row.lag_1 is None
    assert row.lag_52 is None


# ── PredictRequest ────────────────────────────────────────────────────────────


def test_predict_request_accepts_one_row(valid_store_week_row):
    """PredictRequest must have at least one row (min_length=1)."""
    req = PredictRequest(rows=[StoreWeekRow(**valid_store_week_row)])
    assert len(req.rows) == 1
    assert req.rows[0].store_id == 1


def test_predict_request_rejects_empty_rows():
    """PredictRequest.rows cannot be empty."""
    with pytest.raises(ValidationError):
        PredictRequest(rows=[])


def test_predict_request_accepts_multiple_rows(valid_store_week_row):
    """PredictRequest can have multiple rows (batch prediction)."""
    row2 = {**valid_store_week_row, "store_id": 2, "dept_id": 2}
    req = PredictRequest(
        rows=[
            StoreWeekRow(**valid_store_week_row),
            StoreWeekRow(**row2),
        ]
    )
    assert len(req.rows) == 2
    assert req.rows[0].store_id == 1
    assert req.rows[1].store_id == 2


# ── PredictResponse ──────────────────────────────────────────────────────────


def test_predict_response_accepts_list_of_floats():
    """PredictResponse.predictions is a list of floats (one per input row)."""
    resp = PredictResponse(predictions=[32184.21, 15000.0])
    assert resp.predictions == [32184.21, 15000.0]


def test_predict_response_accepts_empty_predictions_list():
    """PredictResponse.predictions can be empty (schema does not enforce min_length)."""
    resp = PredictResponse(predictions=[])
    assert resp.predictions == []


def test_predict_response_rejects_wrong_type():
    """predictions must be list of numbers, not strings."""
    with pytest.raises(ValidationError):
        PredictResponse(predictions=["not", "numbers"])
