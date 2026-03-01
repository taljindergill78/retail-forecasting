"""
Run all baseline models and compare results with segmented evaluation.

This script:
1. Loads train and validation splits
2. Runs each baseline model (including new stronger baselines)
3. Evaluates predictions with WMAE and standard metrics
4. Performs segmented evaluation (by store_type, holiday, volume quartiles, top depts)
5. Compares results and saves to files
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.model.baseline import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
    seasonal_average_forecast,
    hybrid_fallback_forecast
)
from src.config import load_params, get_splits_dir, get_data_dir
from src.model.evaluate import evaluate_predictions_with_wmae, evaluate_by_segments


def create_segments(val_df):
    """Create segment columns for evaluation."""
    val_df = val_df.copy()
    
    # Volume quartiles based on mean sales per (store, dept)
    store_dept_means = val_df.groupby(['store_id', 'dept_id'])['weekly_sales'].mean()
    quartiles = store_dept_means.quantile([0.25, 0.5, 0.75])
    
    def assign_quartile(row):
        mean_sales = store_dept_means.get((row['store_id'], row['dept_id']), 0)
        if mean_sales <= quartiles[0.25]:
            return 'Q1 (Low)'
        elif mean_sales <= quartiles[0.5]:
            return 'Q2 (Medium-Low)'
        elif mean_sales <= quartiles[0.75]:
            return 'Q3 (Medium-High)'
        else:
            return 'Q4 (High)'
    
    val_df['volume_quartile'] = val_df.apply(assign_quartile, axis=1)
    
    # Top departments (top 10 by total sales)
    dept_totals = val_df.groupby('dept_id')['weekly_sales'].sum().sort_values(ascending=False)
    top_depts = set(dept_totals.head(10).index)
    val_df['is_top_dept'] = val_df['dept_id'].apply(lambda x: 'Top 10' if x in top_depts else 'Other')
    
    return val_df


def main():
    params = load_params()
    holiday_weight = params.get("evaluation", {}).get("holiday_weight", 5.0)

    print("=" * 60)
    print("📊 Baseline Model Evaluation (Enhanced)")
    print("=" * 60)

    # Step 1: Load data (paths overridable via SPLITS_DIR on SageMaker)
    splits_dir = get_splits_dir()
    data_dir = get_data_dir()
    print("\n📥 Loading data...")
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    
    # Convert dates to datetime
    train_df['week_date'] = pd.to_datetime(train_df['week_date'])
    val_df['week_date'] = pd.to_datetime(val_df['week_date'])
    
    print(f"  ✅ Training: {len(train_df):,} rows")
    print(f"  ✅ Validation: {len(val_df):,} rows")
    
    # Create segments
    val_df = create_segments(val_df)
    
    # Get unique forecast dates from validation set
    forecast_dates = sorted(val_df['week_date'].unique())
    print(f"  ✅ Forecast horizon: {len(forecast_dates)} weeks")
    
    # Step 2: Run each baseline
    results = {}
    segmented_results = {}
    
    baselines = [
        ('Naive', naive_forecast),
        ('Seasonal Naive', seasonal_naive_forecast),
        ('Moving Average (4 weeks)', lambda t, d: moving_average_forecast(t, d, window_size=4)),
        ('Seasonal Average (k=3)', lambda t, d: seasonal_average_forecast(t, d, k=3)),
        ('Hybrid Fallback', hybrid_fallback_forecast),
    ]
    
    for baseline_name, baseline_func in baselines:
        print(f"\n🔮 Running {baseline_name}...")
        try:
            pred = baseline_func(train_df, forecast_dates)
            
            # Overall metrics with WMAE
            metrics = evaluate_predictions_with_wmae(val_df, pred, holiday_weight=holiday_weight)
            results[baseline_name] = metrics
            
            print(f"  RMSE: ${metrics['rmse']:,.2f}")
            print(f"  MAE: ${metrics['mae']:,.2f}")
            print(f"  WMAE: ${metrics['wmae']:,.2f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  R²: {metrics['r_squared']:.4f}")
            
            # Segmented evaluation
            segments = {}
            for segment_col in ['store_type', 'isholiday', 'volume_quartile', 'is_top_dept']:
                seg_df = evaluate_by_segments(val_df, pred, segment_col, holiday_weight=holiday_weight)
                segments[segment_col] = seg_df
                seg_df.to_csv(data_dir / f'baseline_segments_{baseline_name.replace(" ", "_")}_{segment_col}.csv', index=False)
            
            segmented_results[baseline_name] = segments
            
        except Exception as e:
            print(f"  ❌ Error running {baseline_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Compare overall results
    print("\n" + "=" * 60)
    print("📊 Baseline Comparison (Overall)")
    print("=" * 60)
    
    comparison_df = pd.DataFrame(results).T
    # Reorder columns for better readability
    comparison_df = comparison_df[['rmse', 'mae', 'wmae', 'mape', 'r_squared', 'n_samples']]
    print(comparison_df.to_string())
    
    # Step 4: Save overall results (overridable via DATA_DIR on SageMaker)
    output_file = data_dir / 'baseline_results.csv'
    comparison_df.to_csv(output_file)
    print(f"\n💾 Overall results saved to: {output_file}")
    
    # Step 5: Find best baseline by RMSE and WMAE
    print("\n" + "=" * 60)
    print("🏆 Best Baselines")
    print("=" * 60)
    
    best_rmse = comparison_df['rmse'].idxmin()
    best_wmae = comparison_df['wmae'].idxmin()
    
    print(f"\n  Best by RMSE: {best_rmse}")
    print(f"    RMSE: ${comparison_df.loc[best_rmse, 'rmse']:,.2f}")
    print(f"    WMAE: ${comparison_df.loc[best_rmse, 'wmae']:,.2f}")
    
    print(f"\n  Best by WMAE: {best_wmae}")
    print(f"    RMSE: ${comparison_df.loc[best_wmae, 'rmse']:,.2f}")
    print(f"    WMAE: ${comparison_df.loc[best_wmae, 'wmae']:,.2f}")
    
    # Step 6: Segmented analysis summary
    print("\n" + "=" * 60)
    print("📊 Segmented Analysis Summary")
    print("=" * 60)
    
    # Show segmented results for best baseline
    best_baseline = best_rmse  # Use RMSE winner as default
    print(f"\n  Segmented results for: {best_baseline}")
    
    for segment_col, seg_df in segmented_results[best_baseline].items():
        print(f"\n  By {segment_col}:")
        print(seg_df[['segment', 'rmse', 'wmae', 'n_samples']].to_string(index=False))
    
    # Step 7: Save segmented summary
    summary_segments = []
    for baseline_name, segments in segmented_results.items():
        for segment_col, seg_df in segments.items():
            seg_df['baseline'] = baseline_name
            seg_df['segment_type'] = segment_col
            summary_segments.append(seg_df)
    
    if summary_segments:
        all_segments_df = pd.concat(summary_segments, ignore_index=True)
        all_segments_df.to_csv(data_dir / 'baseline_segments_all.csv', index=False)
        print(f"\n💾 All segmented results saved to: {data_dir / 'baseline_segments_all.csv'}")
    
    print("\n" + "=" * 60)
    print("✅ Baseline evaluation complete!")
    print("=" * 60)
    print("\n💡 Key Insights:")
    print("  - Review segmented results to understand failure modes")
    print("  - ML models must beat best baseline on BOTH RMSE and WMAE")
    print("  - Next step: Build ML model that beats these baselines!")


if __name__ == "__main__":
    main()
