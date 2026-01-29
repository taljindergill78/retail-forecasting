"""
Baseline forecasting models.

These are simple, rule-based models that don't use machine learning.
They serve as a minimum performance bar - any ML model should beat these.

Baselines:
1. Naive: next_week = this_week
2. Seasonal Naive: next_week = same_week_last_year
3. Moving Average: next_week = average(last_N_weeks)
"""

import pandas as pd
import numpy as np


def naive_forecast(train_df, forecast_dates):
    """
    Naive forecast: next week's sales = this week's sales.
    
    How it works:
    - For each (store, dept) combination
    - Find the last week's sales in training data
    - Use that value for all forecast dates
    
    Example:
    - Store 1, Dept 1, last week in train: $10,000
    - Forecast for next 4 weeks: all $10,000
    
    Args:
        train_df: DataFrame with columns [store_id, dept_id, week_date, weekly_sales]
        forecast_dates: List of dates to forecast (from validation set)
        
    Returns:
        DataFrame with columns [store_id, dept_id, week_date, predicted_sales]
    """
    # IMPORTANT: Sort by date first to ensure .last() gets the most recent week
    train_df = train_df.copy()
    train_df = train_df.sort_values(['store_id', 'dept_id', 'week_date'])
    
    # Step 1: Get last week's sales for each (store, dept)
    # Group by store and dept, get the last row (most recent week)
    last_sales = train_df.groupby(['store_id', 'dept_id']).last()
    
    # Step 2: Create predictions for each forecast date
    # For each (store, dept) and each forecast date, use last sales value
    predictions = []
    
    for date in forecast_dates:
        for (store_id, dept_id), row in last_sales.iterrows():
            # Safety check: if no data or NaN, skip (shouldn't happen with valid data)
            if pd.isna(row['weekly_sales']):
                continue
                
            predictions.append({
                'store_id': store_id,
                'dept_id': dept_id,
                'week_date': date,
                'predicted_sales': row['weekly_sales']
            })
    
    return pd.DataFrame(predictions)


def seasonal_naive_forecast(train_df, forecast_dates):
    """
    Seasonal naive forecast: next week's sales = same week last year.
    
    How it works:
    - For each forecast date, find the same week last year
    - Look up sales for that week in training data
    - Use that value as prediction
    
    Example:
    - Forecast date: 2012-01-06 (first Friday of 2012)
    - Same week last year: 2011-01-07 (first Friday of 2011)
    - If 2011-01-07 had $12,000 sales → predict $12,000
    
    Args:
        train_df: DataFrame with columns [store_id, dept_id, week_date, weekly_sales]
        forecast_dates: List of dates to forecast
        
    Returns:
        DataFrame with columns [store_id, dept_id, week_date, predicted_sales]
    """
    # IMPORTANT: Sort by date first
    train_df = train_df.copy()
    train_df = train_df.sort_values(['store_id', 'dept_id', 'week_date'])
    train_df['week_of_year'] = train_df['week_date'].dt.isocalendar().week
    train_df['year'] = train_df['week_date'].dt.year
    
    # Step 2: For each forecast date, find same week last year
    predictions = []
    
    for date in forecast_dates:
        # Get week of year for forecast date
        forecast_week = pd.to_datetime(date).isocalendar().week
        forecast_year = pd.to_datetime(date).year
        
        # Find same week in previous year
        lookup_year = forecast_year - 1
        
        # Look up sales for same week last year
        for (store_id, dept_id), group in train_df.groupby(['store_id', 'dept_id']):
            # Sort group by date (safety check)
            group = group.sort_values('week_date')
            
            # Find matching week in previous year
            match = group[
                (group['year'] == lookup_year) & 
                (group['week_of_year'] == forecast_week)
            ]
            
            if len(match) > 0:
                # Use the sales from that week
                predicted_sales = match.iloc[0]['weekly_sales']
            else:
                # Fallback: use last available week (naive forecast)
                if len(group) > 0:
                    predicted_sales = group.iloc[-1]['weekly_sales']
                else:
                    continue  # Skip if no data at all
            
            # Safety check: skip if predicted_sales is NaN
            if pd.isna(predicted_sales):
                continue
            
            predictions.append({
                'store_id': store_id,
                'dept_id': dept_id,
                'week_date': date,
                'predicted_sales': predicted_sales
            })
    
    return pd.DataFrame(predictions)


def moving_average_forecast(train_df, forecast_dates, window_size=4):
    """
    Moving average forecast: next week's sales = average of last N weeks.
    
    How it works:
    - For each (store, dept), calculate average of last N weeks
    - Use that average for all forecast dates
    
    Example (window_size=4):
    - Last 4 weeks: $10k, $12k, $11k, $13k
    - Average: ($10k + $12k + $11k + $13k) / 4 = $11,500
    - Forecast: $11,500 for all future weeks
    
    Args:
        train_df: DataFrame with columns [store_id, dept_id, week_date, weekly_sales]
        forecast_dates: List of dates to forecast
        window_size: How many weeks to average (default: 4)
        
    Returns:
        DataFrame with columns [store_id, dept_id, week_date, predicted_sales]
    """
    # IMPORTANT: Sort by date first
    train_df = train_df.copy()
    train_df = train_df.sort_values(['store_id', 'dept_id', 'week_date'])
    
    predictions = []
    
    for (store_id, dept_id), group in train_df.groupby(['store_id', 'dept_id']):
        # Group is already sorted from above
        
        # Get last N weeks (or all if fewer than N)
        last_n_weeks = group.tail(window_size)
        
        # Safety check: skip if no data
        if len(last_n_weeks) == 0:
            continue
        
        # Calculate average
        avg_sales = last_n_weeks['weekly_sales'].mean()
        
        # Safety check: skip if average is NaN
        if pd.isna(avg_sales):
            continue
        
        # Use this average for all forecast dates
        for date in forecast_dates:
            predictions.append({
                'store_id': store_id,
                'dept_id': dept_id,
                'week_date': date,
                'predicted_sales': avg_sales
            })
    
    return pd.DataFrame(predictions)


def seasonal_average_forecast(train_df, forecast_dates, k=3):
    """
    Per-series seasonal average: average of last K occurrences of that week-of-year.
    
    How it works:
    - For each forecast date, find the same week-of-year in previous years
    - Take the average of the last K occurrences (if available)
    - Falls back to fewer occurrences if K not available
    
    Example (k=3):
    - Forecast date: Week 5 of 2012
    - Look at Week 5 of 2011, 2010, 2009 (if available)
    - Average those 3 values → prediction
    
    Args:
        train_df: DataFrame with columns [store_id, dept_id, week_date, weekly_sales]
        forecast_dates: List of dates to forecast
        k: Number of previous occurrences to average (default: 3)
        
    Returns:
        DataFrame with columns [store_id, dept_id, week_date, predicted_sales]
    """
    train_df = train_df.copy()
    train_df = train_df.sort_values(['store_id', 'dept_id', 'week_date'])
    train_df['week_of_year'] = train_df['week_date'].dt.isocalendar().week
    train_df['year'] = train_df['week_date'].dt.year
    
    predictions = []
    
    for date in forecast_dates:
        forecast_week = pd.to_datetime(date).isocalendar().week
        forecast_year = pd.to_datetime(date).year
        
        for (store_id, dept_id), group in train_df.groupby(['store_id', 'dept_id']):
            group = group.sort_values('week_date')
            
            # Find all occurrences of this week-of-year in previous years
            matching_weeks = group[group['week_of_year'] == forecast_week]
            matching_weeks = matching_weeks[matching_weeks['year'] < forecast_year]
            
            if len(matching_weeks) > 0:
                # Take last K occurrences (most recent)
                last_k = matching_weeks.tail(k)
                predicted_sales = last_k['weekly_sales'].mean()
            else:
                # Fallback: use last available week (naive)
                if len(group) > 0:
                    predicted_sales = group.iloc[-1]['weekly_sales']
                else:
                    continue
            
            if pd.isna(predicted_sales):
                continue
            
            predictions.append({
                'store_id': store_id,
                'dept_id': dept_id,
                'week_date': date,
                'predicted_sales': predicted_sales
            })
    
    return pd.DataFrame(predictions)


def hybrid_fallback_forecast(train_df, forecast_dates):
    """
    Hybrid fallback strategy for cold-start groups.
    
    Strategy (in order of preference):
    1. Seasonal naive (if 52+ weeks of history)
    2. Seasonal average (if multiple years available)
    3. Moving average (if 4+ weeks available)
    4. Store-dept mean (if any data available)
    
    Args:
        train_df: DataFrame with columns [store_id, dept_id, week_date, weekly_sales]
        forecast_dates: List of dates to forecast
        
    Returns:
        DataFrame with columns [store_id, dept_id, week_date, predicted_sales]
    """
    train_df = train_df.copy()
    train_df = train_df.sort_values(['store_id', 'dept_id', 'week_date'])
    train_df['week_of_year'] = train_df['week_date'].dt.isocalendar().week
    train_df['year'] = train_df['week_date'].dt.year
    
    predictions = []
    
    for date in forecast_dates:
        forecast_week = pd.to_datetime(date).isocalendar().week
        forecast_year = pd.to_datetime(date).year
        lookup_year = forecast_year - 1
        
        for (store_id, dept_id), group in train_df.groupby(['store_id', 'dept_id']):
            group = group.sort_values('week_date')
            n_weeks = len(group)
            
            predicted_sales = None
            
            # Strategy 1: Seasonal naive (if 52+ weeks)
            if n_weeks >= 52:
                match = group[
                    (group['year'] == lookup_year) & 
                    (group['week_of_year'] == forecast_week)
                ]
                if len(match) > 0:
                    predicted_sales = match.iloc[0]['weekly_sales']
            
            # Strategy 2: Seasonal average (if multiple years)
            if predicted_sales is None or pd.isna(predicted_sales):
                matching_weeks = group[group['week_of_year'] == forecast_week]
                matching_weeks = matching_weeks[matching_weeks['year'] < forecast_year]
                if len(matching_weeks) >= 2:
                    predicted_sales = matching_weeks['weekly_sales'].mean()
            
            # Strategy 3: Moving average (if 4+ weeks)
            if predicted_sales is None or pd.isna(predicted_sales):
                if n_weeks >= 4:
                    predicted_sales = group.tail(4)['weekly_sales'].mean()
            
            # Strategy 4: Store-dept mean (fallback)
            if predicted_sales is None or pd.isna(predicted_sales):
                if n_weeks > 0:
                    predicted_sales = group['weekly_sales'].mean()
                else:
                    continue
            
            if pd.isna(predicted_sales):
                continue
            
            predictions.append({
                'store_id': store_id,
                'dept_id': dept_id,
                'week_date': date,
                'predicted_sales': predicted_sales
            })
    
    return pd.DataFrame(predictions)