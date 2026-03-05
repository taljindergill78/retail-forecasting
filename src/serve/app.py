import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.config import get_mlflow_tracking_uri
from src.serve.schema import PredictRequest, PredictResponse


DEFAULT_MODEL_NAME = "RetailSalesForecaster"
DEFAULT_MODEL_STAGE = "Production"

_model = None                   # MLflow pyfunc wrapper (registry access, health, URI tracking)
_model_uri = None
_feature_order: list = []       # exact column order the model was trained with
_cat_cols: Set[str] = set()     # columns the model treats as categorical features


def _resolve_feature_metadata() -> None:
    """
    Read the loaded model's own metadata to determine:
      1. _feature_order — the exact column order the model was trained with.
         Tree models validate that incoming feature names and order match
         training exactly, so we must reorder the DataFrame before predicting.
      2. _cat_cols — which columns were treated as categorical features.
         These must be `category` dtype; all other object columns are numeric
         columns whose None values became `object` dtype and need float64.

    Strategy (tried in order, most specific to least):

    A. Booster feature_types (XGBoost, LightGBM, CatBoost)
       Each framework stores the type of every feature in the trained model.
       For XGBoost: booster.feature_types → list of 'q' / 'c' / 'i' per column.
       We mark any column whose type is 'c' (categorical) as a cat_col.
       This is ground truth — it comes directly from training, not from MLflow.

    B. MLflow input schema (fallback when framework metadata is unavailable)
       MLflow serialises string-typed features as DataType.string.
       We use this as a proxy for categorical columns when (A) is unavailable.

    This function does not hardcode any column name. If the Production model is
    replaced with one that has different categorical features, the metadata read
    at startup will reflect the new model automatically.
    """
    global _feature_order, _cat_cols

    if _model is None:
        return

    impl = _model._model_impl
    resolved = False

    # ── A: framework-native feature_types ────────────────────────────────────
    # Try to get the booster/model object and its feature metadata.
    # We check for the attribute name used by each framework rather than
    # importing the framework package, so this stays import-free and generic.
    booster = None
    for attr in ("xgb_model", "lgb_model", "cb_model"):
        candidate = getattr(impl, attr, None)
        if candidate is not None:
            booster = candidate
            break

    if booster is not None:
        try:
            # XGBoost: XGBRegressor / XGBClassifier
            if hasattr(booster, "get_booster"):
                b = booster.get_booster()
                names = b.feature_names      # ['store_id', 'store_type', ...]
                types = b.feature_types      # ['q', 'c', 'q', 'c', ...]  or None
                if names:
                    _feature_order = list(names)
                if names and types:
                    _cat_cols = {n for n, t in zip(names, types) if t == "c"}
                    resolved = True
            # LightGBM: LGBMRegressor / LGBMClassifier
            elif hasattr(booster, "booster_"):
                lgb_booster = booster.booster_()
                names = lgb_booster.feature_name()
                if names:
                    _feature_order = list(names)
                # LightGBM categorical features are stored in pandas_categorical
                if hasattr(booster, "_catfeature"):
                    _cat_cols = set(booster._catfeature)
                    resolved = True
        except Exception:
            pass  # fall through to strategy B

    # ── B: MLflow input schema fallback ──────────────────────────────────────
    if not resolved:
        try:
            from mlflow.types import DataType
            schema = _model.metadata.get_input_schema()
            if schema:
                if not _feature_order:
                    _feature_order = [spec.name for spec in schema.inputs]
                if not _cat_cols:
                    _cat_cols = {
                        spec.name
                        for spec in schema.inputs
                        if hasattr(spec, "type") and spec.type == DataType.string
                    }
        except Exception:
            pass


def _load_model() -> None:
    """
    Load the MLflow Production model once at startup, then resolve its
    feature metadata so predict() can prepare DataFrames correctly.
    """
    global _model, _model_uri

    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    model_uri_env = os.getenv("MLFLOW_MODEL_URI")
    model_uri = model_uri_env if model_uri_env else f"models:/{DEFAULT_MODEL_NAME}/{DEFAULT_MODEL_STAGE}"

    try:
        _model = mlflow.pyfunc.load_model(model_uri)
        _model_uri = model_uri
        _resolve_feature_metadata()
    except Exception as exc:
        raise RuntimeError(f"Failed to load MLflow model from '{model_uri}': {exc}") from exc


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix pandas dtype issues and column order before calling any ML model.

    Three problems that arise when building a DataFrame from Pydantic model_dump():

    Problem 1 — Categorical columns need `category` dtype
        Tree models (XGBoost, LightGBM, CatBoost) require columns that were
        trained as categorical features to have pandas `category` dtype.
        We know which columns are categorical from _cat_cols, which is
        populated at startup from the model's own booster feature_types
        (the ground truth stored by the framework during training).
        Fix: cast those columns to `category`.

    Problem 2 — None in optional numeric columns → object dtype
        When an optional column (e.g. lag_52) has value None for every row in
        the batch, pandas cannot infer a numeric type and defaults to `object`.
        object dtype is rejected by all tree models.
        Fix: convert remaining object columns to float64 via pd.to_numeric
        (None becomes NaN, which tree models handle as a missing value).

    Problem 3 — Column order mismatch
        Pydantic's model_dump() preserves the schema.py field definition order,
        which may differ from the training DataFrame column order. Tree models
        validate that incoming feature names and order match training exactly.
        Fix: reorder DataFrame columns to match _feature_order, which is
        populated at startup from the model's booster metadata.

    No column names are hardcoded here. All three fixes use metadata read from
    the trained model at startup via _resolve_feature_metadata().
    """
    for col in list(df.select_dtypes(include="object").columns):
        if col in _cat_cols:
            # Categorical feature (from booster feature_types) → category dtype
            df[col] = df[col].astype("category")
        else:
            # Numeric column with None values → NaN → float64
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Also cast any integer-typed columns that are in _cat_cols to category.
    # (e.g. dept_id: trained as categorical but MLflow schema stores it as int)
    for col in _cat_cols:
        if col in df.columns and df[col].dtype != "category":
            df[col] = df[col].astype("category")

    # Reorder columns to match the exact training order from the model's metadata.
    if _feature_order:
        df = df[[c for c in _feature_order if c in df.columns]]

    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Retail Sales Forecaster API", version="0.2.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui() -> HTMLResponse:
    """
    Serve the interactive prediction form.

    Returns the HTML page at src/serve/static/index.html.
    The page calls POST /predict via JavaScript fetch() — no page reload needed.
    """
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
def health() -> dict:
    """
    Simple health endpoint for load balancers and smoke tests.

    Returns JSON with status and a flag indicating whether a model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_uri": _model_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict endpoint (Step 2 version).

    Pipeline:
      1. Pydantic has already validated every field in request.rows.
      2. Build a pandas DataFrame from the typed rows.
      3. Fix dtype issues (None → NaN for numerics, string cols → category).
      4. Call _model._model_impl.predict(df) — the flavour-specific wrapper —
         which bypasses MLflow's schema enforcement layer (which cannot handle
         category dtype) while still using the correct predict logic for
         whatever model flavour is in Production (XGBoost, LightGBM, etc.).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([row.model_dump() for row in request.rows])
        df = _fix_dtypes(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not build input DataFrame: {exc}") from exc

    try:
        # _model._model_impl is the flavour-specific wrapper loaded by MLflow
        # (e.g. _XGBModelWrapper, _LGBMModelWrapper, _SKLearnModelWrapper).
        # Calling predict() on it bypasses MLflow's schema enforcement — which
        # breaks on category dtype — while each flavour wrapper handles the
        # data correctly for its own framework.
        predictions = _model._model_impl.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    preds_list = list(map(float, predictions.tolist())) if hasattr(predictions, "tolist") else list(predictions)
    return PredictResponse(predictions=preds_list)
