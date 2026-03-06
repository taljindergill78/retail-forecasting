"""
scripts/smoke_test_serving.py
─────────────────────────────────────────────────────────────────────────────
Smoke test for the retail-forecasting FastAPI serving container.

Checks three things:
  1. GET  /health       → 200, model_loaded=true
  2. POST /predict      → 200, prediction in a sensible range  (known good row)
  3. POST /predict      → 422, validation error on a bad payload

Usage
─────
  # Against local Docker container (default http://localhost:8080):
  python scripts/smoke_test_serving.py

  # Against a specific base URL (e.g. ECS ALB):
  python scripts/smoke_test_serving.py --base-url http://my-alb.us-west-2.elb.amazonaws.com

Exit codes
──────────
  0  — all checks passed
  1  — one or more checks failed (details printed to stdout)

This script is intentionally self-contained (stdlib + requests only) so it
can run in any CI environment without extra dependencies.
"""

import argparse
import json
import sys

import requests


# ── Known-good payload (Store 1, Dept 1, 2010-02-05) ─────────────────────────
# This exact row was verified during Step 2 & 3 testing.
# Expected prediction is approximately $32,184 (±10%).
KNOWN_GOOD_ROW = {
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
    "markdown1": 0,
    "markdown2": 0,
    "markdown3": 0,
    "markdown4": 0,
    "markdown5": 0,
    "markdown_total": 0,
    "has_markdown": 0,
}

EXPECTED_PREDICTION_MIN = 10_000
EXPECTED_PREDICTION_MAX = 200_000


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return passed


def run_smoke_tests(base_url: str) -> bool:
    base_url = base_url.rstrip("/")
    print(f"\nSmoke test → {base_url}\n")
    all_passed = True

    # ── Check 1: GET /health ──────────────────────────────────────────────────
    try:
        r = requests.get(f"{base_url}/health", timeout=15)
        body = r.json()
        passed = r.status_code == 200 and body.get("model_loaded") is True
        _check(
            "GET /health → 200 + model_loaded=true",
            passed,
            f"status={r.status_code}  body={json.dumps(body)}",
        )
    except Exception as exc:
        _check("GET /health → 200 + model_loaded=true", False, str(exc))
        passed = False
    all_passed = all_passed and passed

    # ── Check 2: POST /predict with known-good row ────────────────────────────
    try:
        r = requests.post(
            f"{base_url}/predict",
            json={"rows": [KNOWN_GOOD_ROW]},
            timeout=30,
        )
        body = r.json()
        preds = body.get("predictions", [])
        pred_val = preds[0] if preds else None
        in_range = (
            pred_val is not None
            and EXPECTED_PREDICTION_MIN <= pred_val <= EXPECTED_PREDICTION_MAX
        )
        passed = r.status_code == 200 and in_range
        _check(
            f"POST /predict → 200 + prediction in [{EXPECTED_PREDICTION_MIN:,}–{EXPECTED_PREDICTION_MAX:,}]",
            passed,
            f"status={r.status_code}  prediction={pred_val}",
        )
    except Exception as exc:
        _check("POST /predict → 200 + prediction in range", False, str(exc))
        passed = False
    all_passed = all_passed and passed

    # ── Check 3: POST /predict with bad payload → 422 ────────────────────────
    try:
        r = requests.post(
            f"{base_url}/predict",
            json={"rows": [{"store_id": "not_a_number", "dept_id": 1}]},
            timeout=15,
        )
        passed = r.status_code == 422
        _check(
            "POST /predict with bad payload → 422 Unprocessable Entity",
            passed,
            f"status={r.status_code}",
        )
    except Exception as exc:
        _check("POST /predict with bad payload → 422", False, str(exc))
        passed = False
    all_passed = all_passed and passed

    print()
    if all_passed:
        print("All checks passed.")
    else:
        print("One or more checks FAILED. See details above.")
    print()
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the serving API.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the serving API (default: http://localhost:8080)",
    )
    args = parser.parse_args()
    success = run_smoke_tests(args.base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
