"""
Inspect predictions to verify models are working correctly.

This script:
1. Loads actual validation data
2. Loads/generates predictions from each model
3. Shows side-by-side comparison
4. Identifies any issues
"""

import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.config import get_splits_dir
from src.model.baseline import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast
)


def inspect_predictions(store_id=1, dept_id=1, n_samples=10):
    """
    Inspect predictions for a specific store-dept combination.
    
    Args:
        store_id: Store to inspect
        dept_id: Department to inspect
        n_samples: Number of weeks to show
    """
    print("=" * 80)
    print(f"🔍 Inspecting Predictions: Store {store_id}, Dept {dept_id}")
    print("=" * 80)
    
    # Load data (paths overridable via SPLITS_DIR)
    splits_dir = get_splits_dir()
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    
    train_df['week_date'] = pd.to_datetime(train_df['week_date'])
    val_df['week_date'] = pd.to_datetime(val_df['week_date'])
    
    # Filter to specific store-dept
    train_filtered = train_df[
        (train_df['store_id'] == store_id) & 
        (train_df['dept_id'] == dept_id)
    ].sort_values('week_date')
    
    val_filtered = val_df[
        (val_df['store_id'] == store_id) & 
        (val_df['dept_id'] == dept_id)
    ].sort_values('week_date')
    
    print(f"\n📊 Training Data: {len(train_filtered)} weeks")
    print(f"   Date range: {train_filtered['week_date'].min()} to {train_filtered['week_date'].max()}")
    print(f"   Last 3 weeks of training:")
    print(train_filtered[['week_date', 'weekly_sales']].tail(3).to_string(index=False))
    
    print(f"\n📊 Validation Data: {len(val_filtered)} weeks")
    print(f"   Date range: {val_filtered['week_date'].min()} to {val_filtered['week_date'].max()}")
    print(f"   First {n_samples} weeks of validation:")
    print(val_filtered[['week_date', 'weekly_sales']].head(n_samples).to_string(index=False))
    
    # Generate predictions
    forecast_dates = sorted(val_filtered['week_date'].unique())
    
    print("\n" + "=" * 80)
    print("🔮 Predictions Comparison")
    print("=" * 80)
    
    # Naive
    naive_pred = naive_forecast(train_filtered, forecast_dates[:n_samples])
    naive_pred = naive_pred[naive_pred['store_id'] == store_id]
    naive_pred = naive_pred[naive_pred['dept_id'] == dept_id]
    
    # Seasonal Naive
    seasonal_pred = seasonal_naive_forecast(train_filtered, forecast_dates[:n_samples])
    seasonal_pred = seasonal_pred[seasonal_pred['store_id'] == store_id]
    seasonal_pred = seasonal_pred[seasonal_pred['dept_id'] == dept_id]
    
    # Moving Average
    ma_pred = moving_average_forecast(train_filtered, forecast_dates[:n_samples], window_size=4)
    ma_pred = ma_pred[ma_pred['store_id'] == store_id]
    ma_pred = ma_pred[ma_pred['dept_id'] == dept_id]
    
    # Combine for comparison
    comparison = val_filtered.head(n_samples)[['week_date', 'weekly_sales']].copy()
    comparison = comparison.rename(columns={'weekly_sales': 'actual'})
    
    comparison['naive'] = naive_pred['predicted_sales'].values
    comparison['seasonal_naive'] = seasonal_pred['predicted_sales'].values
    comparison['moving_avg'] = ma_pred['predicted_sales'].values
    
    # Calculate errors
    comparison['naive_error'] = comparison['actual'] - comparison['naive']
    comparison['seasonal_error'] = comparison['actual'] - comparison['seasonal_naive']
    comparison['ma_error'] = comparison['actual'] - comparison['moving_avg']
    
    print("\n" + comparison.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 Summary Statistics")
    print("=" * 80)
    print(f"\nNaive Forecast:")
    print(f"  Mean error: ${comparison['naive_error'].mean():,.2f}")
    print(f"  Std error: ${comparison['naive_error'].std():,.2f}")
    
    print(f"\nSeasonal Naive Forecast:")
    print(f"  Mean error: ${comparison['seasonal_error'].mean():,.2f}")
    print(f"  Std error: ${comparison['seasonal_error'].std():,.2f}")
    
    print(f"\nMoving Average Forecast:")
    print(f"  Mean error: ${comparison['ma_error'].mean():,.2f}")
    print(f"  Std error: ${comparison['ma_error'].std():,.2f}")


if __name__ == "__main__":
    # Inspect first store, first dept
    inspect_predictions(store_id=1, dept_id=1, n_samples=10)
    
    # You can also inspect other combinations:
    # inspect_predictions(store_id=1, dept_id=2, n_samples=10)