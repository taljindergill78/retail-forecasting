"""
Feature engineering for retail sales forecasting.

This module creates features that help ML models beat baselines:
- Time features: week_of_year, month, year
- Lag features: sales from previous weeks
- Rolling stats: averages and standard deviations
- Store/dept features: already exist, we'll use them
"""

import sys
from pathlib import Path

# Ensure project root is on path when run as python src/features/feature_eng.py (e.g. by DVC)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
from typing import List

from src.config import load_params


def add_time_features(df):
    """
    Add time-based features from week_date.
    
    Features created:
    - week_of_year: Week number (1-52)
    - month: Month number (1-12)
    - year: Year (2010, 2011, etc.)
    - day_of_year: Day of year (1-365)
    
    Why these?
    - Capture seasonality (holidays, summer, winter)
    - No future data needed (just extracting from date)
    
    Args:
        df: DataFrame with 'week_date' column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Ensure week_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['week_date']):
        df['week_date'] = pd.to_datetime(df['week_date'])
    
    # Extract time features
    df['week_of_year'] = df['week_date'].dt.isocalendar().week
    df['month'] = df['week_date'].dt.month
    df['year'] = df['week_date'].dt.year
    df['day_of_year'] = df['week_date'].dt.dayofyear
    
    return df


def add_lag_features(df, group_cols=['store_id', 'dept_id'], target_col='weekly_sales', lags=[1, 2, 4, 52]):
    """
    Add lag features (sales from previous weeks).
    
    How it works:
    - Group by (store_id, dept_id) to keep series separate
    - Sort by date
    - Shift sales backward in time using .shift()
    - lag_1 = sales from 1 week ago
    - lag_52 = sales from 52 weeks ago (yearly pattern)
    
    Why lags?
    - lag_1: Recent trend (last week's sales)
    - lag_2, lag_4: Short-term patterns
    - lag_52: Yearly seasonality (same week last year)
    
    Example:
    Week 10: sales = $20,000
    Week 11: lag_1 = $20,000 (from week 10)
    Week 12: lag_1 = $22,000 (from week 11), lag_2 = $20,000 (from week 10)
    
    Args:
        df: DataFrame with sales data
        group_cols: Columns to group by (store, dept)
        target_col: Column to create lags from
        lags: List of lag periods (weeks)
        
    Returns:
        DataFrame with added lag features
    """
    df = df.copy()
    
    # Sort by date first (critical!)
    df = df.sort_values(group_cols + ['week_date'])
    
    # Group by store and dept, then shift
    for lag in lags:
        # Shift backward by 'lag' periods
        # This gives us sales from 'lag' weeks ago
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    
    return df


def add_rolling_features(df, group_cols=['store_id', 'dept_id'], target_col='weekly_sales', windows=[4, 8]):
    """
    Add rolling statistics (moving averages, standard deviations).
    
    How it works:
    - Group by (store_id, dept_id)
    - Shift by 1 to exclude current week (no data leakage)
    - Calculate rolling mean/std over last N weeks
    - Only uses past data (no future leakage)
    
    Why rolling stats?
    - Smooth out noise
    - Capture trends
    - rolling_std shows volatility
    
    Example:
    Weeks 10-13: [20k, 22k, 21k, 23k]
    At week 14 (predicting week 14):
    - rolling_mean_4 = mean([20k, 22k, 21k, 23k]) = $21,500
    - rolling_std_4 = std([20k, 22k, 21k, 23k]) = volatility
    
    Args:
        df: DataFrame with sales data
        group_cols: Columns to group by
        target_col: Column to calculate stats on
        windows: List of window sizes (weeks)
        
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    
    # Sort by date first (critical!)
    df = df.sort_values(group_cols + ['week_date'])
    
    # Group by store and dept, then calculate rolling stats
    for window in windows:
        # Shift by 1 first to exclude current week (prevents data leakage)
        # Then calculate rolling stats within each group
        # Using apply to chain shift + rolling correctly
        df[f'rolling_mean_{window}'] = (
            df.groupby(group_cols)[target_col]
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            .reset_index(level=group_cols, drop=True)
        )
        
        df[f'rolling_std_{window}'] = (
            df.groupby(group_cols)[target_col]
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
            .reset_index(level=group_cols, drop=True)
        )
    
    return df

def add_markdown_features(df):
    """
    Handle markdown columns (missing before Nov 2011).
    
    Strategy:
    1. Fill NaN with 0 (no markdown = 0)
    2. Create markdown_total = sum of all markdowns
    3. Create has_markdown flag (boolean)
    
    Why this works:
    - 0 is meaningful (no promotion happened)
    - markdown_total captures total promotional impact
    - has_markdown helps model identify promotional weeks
    
    Args:
        df: DataFrame with markdown1-5 columns
        
    Returns:
        DataFrame with added markdown features
    """
    df = df.copy()
    
    # Fill NaN with 0 (no markdown = 0)
    markdown_cols = ['markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5']
    for col in markdown_cols:
        df[col] = df[col].fillna(0)
    
    # Total markdown value
    df['markdown_total'] = df[markdown_cols].sum(axis=1)
    
    # Flag for any markdown
    df['has_markdown'] = (df['markdown_total'] > 0).astype(int)
    
    return df


def add_holiday_window_features(df):
    """
    Add holiday window features (previous/next holiday flags).
    
    Important: is_holiday_next1 is only valid if holidays are known from calendar
    (which they are for Walmart - Super Bowl, Labor Day, Thanksgiving, Christmas).
    This is NOT data leakage because holiday dates are known in advance.
    
    Features created:
    - is_holiday_prev1: Was last week a holiday?
    - is_holiday_next1: Is next week a holiday? (calendar-known)
    - days_since_holiday: Days since last holiday
    - days_until_holiday: Days until next holiday (calendar-known)
    
    Args:
        df: DataFrame with 'isholiday' and 'week_date' columns
        
    Returns:
        DataFrame with added holiday window features
    """
    df = df.copy()
    
    # Ensure sorted by date
    if 'store_id' in df.columns and 'dept_id' in df.columns:
        df = df.sort_values(['store_id', 'dept_id', 'week_date'])
    else:
        df = df.sort_values('week_date')
    
    # Previous holiday flag (shift backward)
    if 'store_id' in df.columns and 'dept_id' in df.columns:
        df['is_holiday_prev1'] = df.groupby(['store_id', 'dept_id'])['isholiday'].shift(1).fillna(0).astype(int)
    else:
        df['is_holiday_prev1'] = df['isholiday'].shift(1).fillna(0).astype(int)
    
    # Next holiday flag (shift forward - calendar-known, not leakage)
    if 'store_id' in df.columns and 'dept_id' in df.columns:
        df['is_holiday_next1'] = df.groupby(['store_id', 'dept_id'])['isholiday'].shift(-1).fillna(0).astype(int)
    else:
        df['is_holiday_next1'] = df['isholiday'].shift(-1).fillna(0).astype(int)
    
    # Days since/until holiday (simplified: just flag if within 7 days)
    # This is a simplified version - full implementation would track actual days
    df['within_7days_after_holiday'] = (df['is_holiday_prev1'] == 1).astype(int)
    df['within_7days_before_holiday'] = (df['is_holiday_next1'] == 1).astype(int)
    
    return df


def add_exogenous_lag_features(df, group_cols=['store_id', 'dept_id'], 
                               exogenous_cols=['markdown_total', 'fuel_price', 'temperature'],
                               lags=[1, 2, 4]):
    """
    Add lag features for exogenous variables (markdowns, fuel price, temperature).
    
    These are external factors that might influence sales.
    We only use past values (no future leakage).
    
    Args:
        df: DataFrame with exogenous columns
        group_cols: Columns to group by (store, dept)
        exogenous_cols: List of exogenous column names
        lags: List of lag periods (weeks)
        
    Returns:
        DataFrame with added exogenous lag features
    """
    df = df.copy()
    df = df.sort_values(group_cols + ['week_date'])
    
    for col in exogenous_cols:
        if col not in df.columns:
            continue
        
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(group_cols)[col].shift(lag)
    
    return df


def add_exogenous_rolling_features(df, group_cols=['store_id', 'dept_id'],
                                   exogenous_cols=['markdown_total', 'fuel_price', 'temperature'],
                                   windows=[4, 8]):
    """
    Add rolling statistics for exogenous variables.
    
    Args:
        df: DataFrame with exogenous columns
        group_cols: Columns to group by
        exogenous_cols: List of exogenous column names
        windows: List of window sizes (weeks)
        
    Returns:
        DataFrame with added exogenous rolling features
    """
    df = df.copy()
    df = df.sort_values(group_cols + ['week_date'])
    
    for col in exogenous_cols:
        if col not in df.columns:
            continue
        
        for window in windows:
            # Shift by 1 to exclude current week (no data leakage)
            df[f'{col}_rolling_mean_{window}'] = (
                df.groupby(group_cols)[col]
                .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
                .reset_index(level=group_cols, drop=True)
            )
    
    return df


def handle_negative_sales(df, strategy='clip'):
    """
    Handle negative weekly_sales values (returns/data issues).
    
    Strategies:
    - 'clip': Clip negative values to 0 (treat as returns/refunds)
    - 'keep': Keep negative values as-is (might be legitimate returns)
    - 'flag': Keep values but add a flag column
    
    Args:
        df: DataFrame with 'weekly_sales' column
        strategy: How to handle negatives ('clip', 'keep', 'flag')
        
    Returns:
        DataFrame with handled negative sales
    """
    df = df.copy()
    
    if strategy == 'clip':
        # Clip negatives to 0 (treat as returns)
        n_negative = (df['weekly_sales'] < 0).sum()
        if n_negative > 0:
            print(f"  ⚠️  Clipping {n_negative:,} negative sales values to 0")
            df['weekly_sales'] = df['weekly_sales'].clip(lower=0)
    elif strategy == 'flag':
        # Add flag but keep values
        df['is_return'] = (df['weekly_sales'] < 0).astype(int)
    # 'keep' strategy: do nothing
    
    return df


def build_features(train_df, val_df=None, test_df=None, negative_sales_strategy='clip', lags=None, rolling_windows=None):
    """
    Build features for train/val/test splits with explicit feature groups.

    Feature Groups:
    1. **Time features**: week_of_year, month, year, day_of_year
    2. **Target lags/rollings**: lag_1, lag_2, lag_4, lag_52, rolling_mean/std
    3. **Holiday window features**: is_holiday_prev1, is_holiday_next1, within_7days_*
    4. **Exogenous rollings/lags**: markdown_total, fuel_price, temperature lags/rollings
    5. **Static features**: store_type, store_size, dept_id (already in data)
    6. **Markdown features**: markdown_total, has_markdown

    Important: For validation/test, we need to use training data
    to calculate lags (can't use future data).

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        negative_sales_strategy: How to handle negative sales ('clip', 'keep', 'flag')
        lags: List of lag periods for target and exogenous (default [1, 2, 4, 52])
        rolling_windows: List of rolling window sizes (default [4, 8])

    Returns:
        Tuple of (train_features, val_features, test_features)
    """
    if lags is None:
        lags = [1, 2, 4, 52]
    if rolling_windows is None:
        rolling_windows = [4, 8]
    print("🔧 Building features...")
    print("  Feature groups: time, target_lags, target_rollings, holiday_windows, exogenous, static, markdown")
    
    # Step 0: Handle negative sales
    print("  ✅ Handling negative sales...")
    train_feat = handle_negative_sales(train_df, strategy=negative_sales_strategy)
    if val_df is not None:
        val_feat = handle_negative_sales(val_df, strategy=negative_sales_strategy)
    if test_df is not None:
        test_feat = handle_negative_sales(test_df, strategy=negative_sales_strategy)
    
    # Step 1: Time features (safe - no future data)
    print("  ✅ Adding time features...")
    train_feat = add_time_features(train_feat)
    if val_df is not None:
        val_feat = add_time_features(val_feat)
    if test_df is not None:
        test_feat = add_time_features(test_feat)
    
    # Step 1.5: Markdown features (handle missing values)
    print("  ✅ Adding markdown features...")
    train_feat = add_markdown_features(train_feat)
    if val_df is not None:
        val_feat = add_markdown_features(val_feat)
    if test_df is not None:
        test_feat = add_markdown_features(test_feat)
    
    # Step 2: Lag features
    print("  ✅ Adding lag features...")
    train_feat = add_lag_features(train_feat, lags=lags)

    # For validation: need to combine with train to get lags
    if val_df is not None:
        # Add marker to track train vs val before combining
        train_feat['_split_marker'] = 'train'
        val_feat['_split_marker'] = 'val'
        # Combine train + val, sort by date, calculate lags, then split
        combined = pd.concat([train_feat, val_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_lag_features(combined, lags=lags)
        # Split back using marker
        val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        # Remove marker columns
        val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])
    
    # For test: need to combine train + val + test
    if test_df is not None:
        # Add markers to track splits
        train_feat['_split_marker'] = 'train'
        if val_df is not None:
            val_feat['_split_marker'] = 'val'
        test_feat['_split_marker'] = 'test'
        # Combine, sort, calculate lags
        if val_df is not None:
            combined = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
        else:
            combined = pd.concat([train_feat, test_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_lag_features(combined, lags=lags)
        # Split back using markers
        test_feat = combined[combined['_split_marker'] == 'test'].copy()
        if val_df is not None:
            val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        # Remove marker columns
        test_feat = test_feat.drop(columns=['_split_marker'])
        if val_df is not None:
            val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])

    # Step 3: Rolling features
    print("  ✅ Adding rolling features...")
    train_feat = add_rolling_features(train_feat, windows=rolling_windows)

    if val_df is not None:
        # Add markers to track splits
        train_feat['_split_marker'] = 'train'
        val_feat['_split_marker'] = 'val'
        combined = pd.concat([train_feat, val_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_rolling_features(combined, windows=rolling_windows)
        # Split back using markers
        val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        # Remove marker columns
        val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])
    
    if test_df is not None:
        # Add markers to track splits
        train_feat['_split_marker'] = 'train'
        if val_df is not None:
            val_feat['_split_marker'] = 'val'
        test_feat['_split_marker'] = 'test'
        # Combine, sort, calculate rolling features
        if val_df is not None:
            combined = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
        else:
            combined = pd.concat([train_feat, test_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_rolling_features(combined, windows=rolling_windows)
        # Split back using markers
        test_feat = combined[combined['_split_marker'] == 'test'].copy()
        if val_df is not None:
            val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        # Remove marker columns
        test_feat = test_feat.drop(columns=['_split_marker'])
        if val_df is not None:
            val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])

    # Step 4: Holiday window features (calendar-known, not leakage)
    print("  ✅ Adding holiday window features...")
    train_feat = add_holiday_window_features(train_feat)
    if val_df is not None:
        val_feat = add_holiday_window_features(val_feat)
    if test_df is not None:
        test_feat = add_holiday_window_features(test_feat)
    
    # Step 5: Exogenous lag features (markdown_total, fuel_price, temperature)
    print("  ✅ Adding exogenous lag features...")
    train_feat = add_exogenous_lag_features(train_feat, lags=lags)

    if val_df is not None:
        train_feat['_split_marker'] = 'train'
        val_feat['_split_marker'] = 'val'
        combined = pd.concat([train_feat, val_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_exogenous_lag_features(combined, lags=lags)
        val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])
    
    if test_df is not None:
        train_feat['_split_marker'] = 'train'
        if val_df is not None:
            val_feat['_split_marker'] = 'val'
        test_feat['_split_marker'] = 'test'
        if val_df is not None:
            combined = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
        else:
            combined = pd.concat([train_feat, test_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_exogenous_lag_features(combined, lags=lags)
        test_feat = combined[combined['_split_marker'] == 'test'].copy()
        if val_df is not None:
            val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        test_feat = test_feat.drop(columns=['_split_marker'])
        if val_df is not None:
            val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])

    # Step 6: Exogenous rolling features
    print("  ✅ Adding exogenous rolling features...")
    train_feat = add_exogenous_rolling_features(train_feat, windows=rolling_windows)

    if val_df is not None:
        train_feat['_split_marker'] = 'train'
        val_feat['_split_marker'] = 'val'
        combined = pd.concat([train_feat, val_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_exogenous_rolling_features(combined, windows=rolling_windows)
        val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])

    if test_df is not None:
        train_feat['_split_marker'] = 'train'
        if val_df is not None:
            val_feat['_split_marker'] = 'val'
        test_feat['_split_marker'] = 'test'
        if val_df is not None:
            combined = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
        else:
            combined = pd.concat([train_feat, test_feat], ignore_index=True)
        combined = combined.sort_values(['store_id', 'dept_id', 'week_date'])
        combined = add_exogenous_rolling_features(combined, windows=rolling_windows)
        test_feat = combined[combined['_split_marker'] == 'test'].copy()
        if val_df is not None:
            val_feat = combined[combined['_split_marker'] == 'val'].copy()
        train_feat = combined[combined['_split_marker'] == 'train'].copy()
        test_feat = test_feat.drop(columns=['_split_marker'])
        if val_df is not None:
            val_feat = val_feat.drop(columns=['_split_marker'])
        train_feat = train_feat.drop(columns=['_split_marker'])
    
    print(f"  ✅ Training features: {train_feat.shape}")
    if val_df is not None:
        print(f"  ✅ Validation features: {val_feat.shape}")
    if test_df is not None:
        print(f"  ✅ Test features: {test_feat.shape}")
    
    return train_feat, val_feat if val_df is not None else None, test_feat if test_df is not None else None


def main():
    """
    Main function: load splits, build features, save results.
    """
    print("=" * 60)
    print("🔧 Feature Engineering")
    print("=" * 60)
    
    # Load splits
    print("\n📥 Loading data splits...")
    train_df = pd.read_csv('data/splits/train.csv')
    val_df = pd.read_csv('data/splits/val.csv')
    test_df = pd.read_csv('data/splits/test.csv')
    
    # Convert dates
    train_df['week_date'] = pd.to_datetime(train_df['week_date'])
    val_df['week_date'] = pd.to_datetime(val_df['week_date'])
    test_df['week_date'] = pd.to_datetime(test_df['week_date'])
    
    print(f"  ✅ Training: {len(train_df):,} rows")
    print(f"  ✅ Validation: {len(val_df):,} rows")
    print(f"  ✅ Test: {len(test_df):,} rows")

    # Build features (config from params.yaml)
    params = load_params()
    feat_params = params.get("features", {})
    train_feat, val_feat, test_feat = build_features(
        train_df, val_df, test_df,
        negative_sales_strategy=feat_params.get("negative_sales_strategy", "clip"),
        lags=feat_params.get("lags", [1, 2, 4, 52]),
        rolling_windows=feat_params.get("rolling_windows", [4, 8]),
    )
    
    # Save feature-engineered datasets
    print("\n💾 Saving feature-engineered datasets...")
    train_feat.to_csv('data/splits/train_features.csv', index=False)
    val_feat.to_csv('data/splits/val_features.csv', index=False)
    test_feat.to_csv('data/splits/test_features.csv', index=False)
    
    print("  ✅ Saved: data/splits/train_features.csv")
    print("  ✅ Saved: data/splits/val_features.csv")
    print("  ✅ Saved: data/splits/test_features.csv")
    
    # Show feature summary
    print("\n" + "=" * 60)
    print("📊 Feature Summary")
    print("=" * 60)
    print(f"\nTotal features: {len(train_feat.columns)}")
    print(f"\nFeature columns:")
    for col in train_feat.columns:
        print(f"  - {col}")
    
    # Show missing values
    print("\n" + "=" * 60)
    print("🔍 Missing Values Check")
    print("=" * 60)
    print("\nTraining set:")
    missing = train_feat.isnull().sum()
    print(missing[missing > 0])
    
    print("\n✅ Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
