"""
Model evaluation script: loads train outputs, computes test metrics and visualizations,
and attaches evaluation artifacts to the same MLflow run as train.py (via data/mlflow_run_id.txt).

Run as part of the DVC pipeline after train:
  dvc repro evaluate   # or dvc repro
"""

import json
import sys
from pathlib import Path

# Add project root so "src" is importable when run as: python src/model/evaluate_models.py
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_params, get_data_dir, get_splits_dir, get_reports_dir, get_mlflow_tracking_uri
from src.model.evaluate import evaluate_predictions_with_wmae, evaluate_by_segments

REPORTS_DIR = get_reports_dir()
FIGURES_DIR = REPORTS_DIR / "figures"


def create_evaluation_report(all_metrics, comparison_path, report_path):
    """
    Build evaluation report CSV: one row per model with test metrics.
    Optionally merge in validation metrics from model_comparison.csv if present.
    """
    rows = []
    for model_name, metrics in sorted(all_metrics.items(), key=lambda x: x[1]["rmse"]):
        row = {
            "model_name": model_name,
            "test_rmse": round(metrics["rmse"], 4),
            "test_mae": round(metrics["mae"], 4),
            "test_wmae": round(metrics["wmae"], 4),
            "test_mape": round(metrics["mape"], 4),
            "test_r_squared": round(metrics["r_squared"], 4),
            "n_samples": metrics["n_samples"],
        }
        rows.append(row)
    report_df = pd.DataFrame(rows)

    # Optionally add validation metrics from train's model_comparison.csv
    if Path(comparison_path).exists():
        try:
            comp_df = pd.read_csv(comparison_path)
            if "model_name" in comp_df.columns:
                comp_df = comp_df[["model_name", "rmse", "wmae"]].rename(
                    columns={"rmse": "val_rmse", "wmae": "val_wmae"}
                )
                report_df = report_df.merge(comp_df, on="model_name", how="left")
        except Exception:
            pass

    report_df.to_csv(report_path, index=False)
    return report_df


def main():
    """Main evaluation function."""
    params = load_params()
    eval_params = params.get("evaluation", {})
    holiday_weight = eval_params.get("holiday_weight", 5.0)
    n_plot_samples = eval_params.get("n_plot_samples", 5)

    # If train.py wrote a run_id, we'll attach evaluation artifacts to that MLflow run
    data_dir = get_data_dir()
    mlflow_run_id = None
    run_id_file = data_dir / "mlflow_run_id.txt"
    if run_id_file.exists():
        try:
            mlflow_run_id = run_id_file.read_text().strip()
        except Exception:
            pass

    print("=" * 60)
    print("📊 Model Evaluation & Visualization (All Models)")
    print("=" * 60)

    # Load data (paths overridable via SPLITS_DIR, DATA_DIR on SageMaker)
    splits_dir = get_splits_dir()
    print("\n📥 Loading data...")
    try:
        test_df = pd.read_csv(splits_dir / "test_features.csv")
        test_df["week_date"] = pd.to_datetime(test_df["week_date"])
        print(f"  Test data: {len(test_df):,} rows")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Actuals for merge: need store_id, dept_id, week_date, weekly_sales, isholiday
    actual_cols = ["store_id", "dept_id", "week_date", "weekly_sales"]
    if "isholiday" in test_df.columns:
        actual_cols.append("isholiday")
    actual_df = test_df[actual_cols].copy()

    # Load train outputs
    comparison_path = data_dir / "model_comparison.csv"
    with open(data_dir / "final_test_results.json") as f:
        final_results = json.load(f)
    best_model_name = final_results.get("model", "")

    # Load all test predictions from data/test_predictions_all
    pred_dir = data_dir / "test_predictions_all"
    if not pred_dir.exists():
        print("  ERROR: data/test_predictions_all not found. Run train stage first.")
        sys.exit(1)
    metadata_path = pred_dir / "metadata.json"
    with open(metadata_path) as f:
        meta = json.load(f)
    model_names = meta.get("models", [])

    all_metrics = {}
    all_predictions = {}
    for name in model_names:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        csv_path = pred_dir / f"{safe_name}.csv"
        if not csv_path.exists():
            continue
        pred_df = pd.read_csv(csv_path)
        pred_df["week_date"] = pd.to_datetime(pred_df["week_date"])
        metrics = evaluate_predictions_with_wmae(actual_df, pred_df, holiday_weight=holiday_weight)
        all_metrics[name] = metrics
        all_predictions[name] = pred_df

    if not all_metrics:
        print("  ERROR: No prediction CSVs found in data/test_predictions_all")
        sys.exit(1)

    # Best model's predictions for figures and segments
    best_pred_df = all_predictions.get(best_model_name)
    if best_pred_df is None:
        best_model_name = list(all_predictions.keys())[0]
        best_pred_df = all_predictions[best_model_name]

    # Create report
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_evaluation_report.csv"
    print("\n📝 Creating evaluation report...")
    create_evaluation_report(all_metrics, comparison_path, report_path)

    # Segmented evaluation (holiday vs non-holiday) for best model
    print("\n📊 Creating segmented evaluation...")
    if "isholiday" in actual_df.columns:
        segmented_results = evaluate_by_segments(
            actual_df, best_pred_df, segment_col="isholiday", holiday_weight=holiday_weight
        )
    else:
        segmented_results = pd.DataFrame()

    # Figures: actual vs predicted scatter (best model)
    print("\n🎨 Creating figures...")
    merged = pd.merge(
        actual_df, best_pred_df, on=["store_id", "dept_id", "week_date"], how="inner"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["weekly_sales"], merged["predicted_sales"], alpha=0.3, s=5)
    max_val = max(merged["weekly_sales"].max(), merged["predicted_sales"].max())
    ax.plot([0, max_val], [0, max_val], "k--", label="Perfect prediction")
    ax.set_xlabel("Actual weekly sales ($)")
    ax.set_ylabel("Predicted weekly sales ($)")
    ax.set_title(f"Actual vs Predicted ({best_model_name})")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Segment bar chart (holiday vs non-holiday RMSE)
    if not segmented_results.empty and "segment" in segmented_results.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        seg = segmented_results.copy()
        seg["segment"] = seg["segment"].astype(str)
        ax.bar(seg["segment"], seg["rmse"], color=["#2ecc71", "#3498db"])
        ax.set_ylabel("RMSE ($)")
        ax.set_xlabel("Segment (isholiday)")
        ax.set_title(f"Test RMSE by segment ({best_model_name})")
        fig.savefig(FIGURES_DIR / "rmse_by_segment.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Attach evaluation artifacts to the same MLflow run as train.
    # Use same tracking URI as train (env MLFLOW_TRACKING_URI on SageMaker, else local SQLite).
    if mlflow_run_id:
        try:
            import mlflow
            tracking_uri = get_mlflow_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Retail_Forecasting_Models")
            mlflow.start_run(run_id=mlflow_run_id)
            if FIGURES_DIR.exists():
                mlflow.log_artifacts(str(FIGURES_DIR), artifact_path="evaluation/figures")
            if report_path.exists():
                mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        except Exception as e:
            print(f"  MLflow logging failed: {e}")
        finally:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print(f"\nVisualizations: {FIGURES_DIR}")
    print(f"Report: reports/model_evaluation_report.csv")
    print("\nModel performance (test):")
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1]["rmse"]):
        marker = " " if name != best_model_name else " (best)"
        print(f"  {name}{marker}: RMSE ${m['rmse']:,.2f}, WMAE ${m['wmae']:,.2f}, R2 {m['r_squared']:.4f}")


if __name__ == "__main__":
    main()
