"""
Evaluation metrics for forecasting models.

Metrics:
- RMSE: Root Mean Squared Error (penalizes large errors)
- MAE: Mean Absolute Error (average error size)
- MAPE: Mean Absolute Percentage Error (error as %)
- R²: R-squared (coefficient of determination)
"""

import pandas as pd
import numpy as np


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Formula: sqrt(mean((actual - predicted)^2))
    
    Why RMSE?
    - Penalizes large errors more than small errors
    - Same units as target (dollars in our case)
    - Common in forecasting
    
    Example:
    - Actual: [100, 200, 300]
    - Predicted: [110, 190, 310]
    - Errors: [10, 10, 10]
    - Squared: [100, 100, 100]
    - Mean: 100
    - RMSE: sqrt(100) = 10
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: RMSE value
    """
    errors = actual - predicted
    squared_errors = errors ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error (MAE).
    
    Formula: mean(|actual - predicted|)
    
    Why MAE?
    - Easy to interpret: average error size
    - Not affected by outliers as much as RMSE
    - Same units as target
    
    Example:
    - Actual: [100, 200, 300]
    - Predicted: [110, 190, 310]
    - Errors: [10, 10, 10]
    - Absolute: [10, 10, 10]
    - MAE: 10
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: MAE value
    """
    errors = actual - predicted
    absolute_errors = np.abs(errors)
    mae = np.mean(absolute_errors)
    return mae


def calculate_mape(actual, predicted, threshold=1.0):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Formula: mean(|actual - predicted| / actual) * 100
    
    Why MAPE?
    - Percentage error (easy to understand: "5% error")
    - Scale-independent (works for any sales volume)
    - Business-friendly metric
    
    Why threshold?
    - MAPE explodes when actual values are very small (near zero)
    - Example: actual=$0.10, predicted=$1.00 → 900% error!
    - We exclude values below threshold to avoid this
    
    Example:
    - Actual: [100, 200, 300]
    - Predicted: [110, 190, 310]
    - Errors: [10, 10, 10]
    - Percentages: [10%, 5%, 3.33%]
    - MAPE: (10 + 5 + 3.33) / 3 = 6.11%
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        threshold: Minimum actual value to include (default: $1.00)
        
    Returns:
        float: MAPE value (as percentage), or NaN if no valid values
    """
    # Filter out very small actual values
    mask = actual >= threshold
    
    if mask.sum() == 0:
        # No valid values (all below threshold)
        return np.nan
    
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    # Calculate percentage errors
    percentage_errors = np.abs((actual_filtered - predicted_filtered) / actual_filtered) * 100
    mape = np.mean(percentage_errors)
    
    return mape


def calculate_r_squared(actual, predicted):
    """
    Calculate R-squared (coefficient of determination).
    
    Formula: 1 - (SS_res / SS_tot)
    - SS_res = sum((actual - predicted)^2)  # Sum of squared residuals
    - SS_tot = sum((actual - mean(actual))^2)  # Total sum of squares
    
    Interpretation:
    - R² = 1.0: Perfect predictions (all errors = 0)
    - R² = 0.0: Model performs as well as predicting the mean
    - R² < 0.0: Model is worse than predicting the mean
    
    Why R²?
    - Measures proportion of variance explained by the model
    - Useful for comparing models
    - Note: Can be misleading for time series (doesn't show error magnitude)
    
    Example:
    - Actual: [100, 200, 300]
    - Predicted: [110, 190, 310]
    - Mean actual: 200
    - SS_res: (100-110)² + (200-190)² + (300-310)² = 300
    - SS_tot: (100-200)² + (200-200)² + (300-200)² = 20000
    - R²: 1 - (300/20000) = 0.985
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: R-squared value, or NaN if SS_tot is zero
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    # Avoid division by zero
    if ss_tot == 0:
        return np.nan
    
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def evaluate_predictions(actual_df, predicted_df):
    """
    Evaluate predictions against actual values.
    
    This function:
    1. Merges actual and predicted DataFrames
    2. Calculates all metrics (RMSE, MAE, MAPE, R²)
    3. Returns summary dictionary
    
    Args:
        actual_df: DataFrame with [store_id, dept_id, week_date, weekly_sales]
        predicted_df: DataFrame with [store_id, dept_id, week_date, predicted_sales]
        
    Returns:
        dict: {
            'rmse': float,
            'mae': float,
            'mape': float,
            'r_squared': float,
            'n_samples': int
        }
    """
    # Merge actual and predicted on (store_id, dept_id, week_date)
    merged = pd.merge(
        actual_df,
        predicted_df,
        on=['store_id', 'dept_id', 'week_date'],
        how='inner'
    )
    
    # Extract arrays
    actual = merged['weekly_sales'].values
    predicted = merged['predicted_sales'].values
    
    # Calculate metrics
    rmse = calculate_rmse(actual, predicted)
    mae = calculate_mae(actual, predicted)
    mape = calculate_mape(actual, predicted)
    r_squared = calculate_r_squared(actual, predicted)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r_squared': r_squared,
        'n_samples': len(merged)
    }


def calculate_wmae(actual, predicted, weights):
    """
    Calculate Weighted Mean Absolute Error (WMAE).
    
    Formula: sum(|actual - predicted| * weights) / sum(weights)
    
    Why WMAE?
    - Allows weighting certain predictions more (e.g., holiday weeks)
    - Common in retail forecasting competitions (Kaggle Walmart competition)
    - Business-critical periods get higher weight
    
    Example:
    - Actual: [100, 200, 300]
    - Predicted: [110, 190, 310]
    - Weights: [1.0, 5.0, 1.0] (holiday week gets 5x weight)
    - Errors: [10, 10, 10]
    - Weighted errors: [10*1, 10*5, 10*1] = [10, 50, 10]
    - WMAE: (10 + 50 + 10) / (1 + 5 + 1) = 10
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        weights: Array of weights (same length as actual/predicted)
        
    Returns:
        float: WMAE value
    """
    errors = np.abs(actual - predicted)
    weighted_errors = errors * weights
    wmae = np.sum(weighted_errors) / np.sum(weights)
    return wmae


def evaluate_predictions_with_wmae(actual_df, predicted_df, holiday_weight=5.0):
    """
    Evaluate predictions with both standard metrics and WMAE.
    
    Args:
        actual_df: DataFrame with [store_id, dept_id, week_date, weekly_sales, isholiday]
        predicted_df: DataFrame with [store_id, dept_id, week_date, predicted_sales]
        holiday_weight: Weight for holiday weeks (default: 5.0, matching Kaggle competition)
        
    Returns:
        dict: Standard metrics plus 'wmae'
    """
    # Merge actual and predicted
    merged = pd.merge(
        actual_df,
        predicted_df,
        on=['store_id', 'dept_id', 'week_date'],
        how='inner'
    )
    
    # Extract arrays
    actual = merged['weekly_sales'].values
    predicted = merged['predicted_sales'].values
    
    # Create weights: holiday weeks get higher weight
    weights = np.where(merged['isholiday'].values, holiday_weight, 1.0)
    
    # Calculate all metrics
    rmse = calculate_rmse(actual, predicted)
    mae = calculate_mae(actual, predicted)
    mape = calculate_mape(actual, predicted)
    r_squared = calculate_r_squared(actual, predicted)
    wmae = calculate_wmae(actual, predicted, weights)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'wmae': wmae,
        'mape': mape,
        'r_squared': r_squared,
        'n_samples': len(merged)
    }


def evaluate_by_segments(actual_df, predicted_df, segment_col, holiday_weight=5.0):
    """
    Evaluate predictions segmented by a categorical column.
    
    This helps understand where models fail:
    - Which store types perform worse?
    - Are holiday predictions worse?
    - Which departments are hardest to predict?
    
    Args:
        actual_df: DataFrame with [store_id, dept_id, week_date, weekly_sales, isholiday, ...]
        predicted_df: DataFrame with [store_id, dept_id, week_date, predicted_sales]
        segment_col: Column name to segment by (e.g., 'store_type', 'isholiday')
        holiday_weight: Weight for holiday weeks in WMAE calculation
        
    Returns:
        DataFrame: Metrics for each segment
    """
    # Merge actual and predicted
    merged = pd.merge(
        actual_df,
        predicted_df,
        on=['store_id', 'dept_id', 'week_date'],
        how='inner'
    )
    
    if segment_col not in merged.columns:
        raise ValueError(f"Column '{segment_col}' not found in merged DataFrame")
    
    results = []
    
    for segment_value in merged[segment_col].unique():
        segment_data = merged[merged[segment_col] == segment_value]
        
        if len(segment_data) == 0:
            continue
        
        actual = segment_data['weekly_sales'].values
        predicted = segment_data['predicted_sales'].values
        
        # Create weights for WMAE
        if 'isholiday' in segment_data.columns:
            weights = np.where(segment_data['isholiday'].values, holiday_weight, 1.0)
        else:
            weights = np.ones(len(segment_data))
        
        metrics = {
            'segment': segment_value,
            'rmse': calculate_rmse(actual, predicted),
            'mae': calculate_mae(actual, predicted),
            'wmae': calculate_wmae(actual, predicted, weights),
            'mape': calculate_mape(actual, predicted),
            'r_squared': calculate_r_squared(actual, predicted),
            'n_samples': len(segment_data)
        }
        
        results.append(metrics)
    
    return pd.DataFrame(results)