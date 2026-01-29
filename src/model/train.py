"""
Train multiple candidate models and compare them.

This script:
1. Loads feature-engineered datasets
2. Trains multiple models (Linear, LightGBM, CatBoost)
3. Evaluates on validation set
4. Selects best model based on RMSE+WMAE rule
5. Retrains best model on train+val, evaluates on test
6. Saves models and results
"""

import sys
import os
import pickle
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from src.model.evaluate import evaluate_predictions_with_wmae, evaluate_by_segments


# Define feature groups for easier management
TIME_FEATURES = ['week_of_year', 'month', 'year', 'day_of_year']
STATIC_FEATURES = ['store_id', 'store_type', 'store_size', 'dept_id']
CATEGORICAL_FEATURES = ['store_type', 'dept_id']
TARGET_COL = 'weekly_sales'


def prepare_features(train_df, val_df, test_df=None):
    """
    Prepare features and targets for training.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (if test provided),
        feature_names, categorical_indices
    """
    # Identify feature columns (exclude target and date)
    exclude_cols = [TARGET_COL, 'week_date']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET_COL].values
    
    # Handle categoricals for tree models
    categorical_indices = []
    for i, col in enumerate(feature_cols):
        if col in CATEGORICAL_FEATURES:
            categorical_indices.append(i)
            # Convert to category type for LightGBM/CatBoost
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
    
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        y_test = test_df[TARGET_COL].values
        for col in CATEGORICAL_FEATURES:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('category')
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, categorical_indices
    
    return X_train, y_train, X_val, y_val, feature_cols, categorical_indices


def train_linear_model(X_train, y_train, X_val, y_val):
    """Train a regularized linear model with cross-validation for hyperparameter tuning."""
    print("  📊 Training Ridge Regression with CV...")
    
    # For linear models, we need to encode categoricals
    # Create a copy to avoid modifying original
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    
    # One-hot encode categorical features for linear model
    # Identify categorical columns (object type or explicitly categorical)
    cat_cols = X_train_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    ohe = None
    
    if len(cat_cols) > 0:
        # One-hot encode
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        X_train_cat = ohe.fit_transform(X_train_encoded[cat_cols])
        X_val_cat = ohe.transform(X_val_encoded[cat_cols])
        
        # Drop original categorical columns
        X_train_encoded = X_train_encoded.drop(columns=cat_cols)
        X_val_encoded = X_val_encoded.drop(columns=cat_cols)
        
        # Combine with encoded features
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        X_train_encoded = pd.concat([
            X_train_encoded.reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train_encoded.index)
        ], axis=1)
        X_val_encoded = pd.concat([
            X_val_encoded.reset_index(drop=True),
            pd.DataFrame(X_val_cat, columns=cat_feature_names, index=X_val_encoded.index)
        ], axis=1)
    
    # Handle NaN values (impute with median)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_encoded)
    X_val_imputed = imputer.transform(X_val_encoded)
    
    # Scale features for linear model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    # Use RidgeCV for automatic hyperparameter tuning
    # Alpha values to try: 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    model.fit(X_train_scaled, y_train)
    
    print(f"    Selected alpha: {model.alpha_:.2f}")
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'ohe': ohe,
        'cat_cols': cat_cols,
        'model_type': 'linear',
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val
        }
    }


def train_lightgbm(X_train, y_train, X_val, y_val, categorical_indices):
    """Train a LightGBM model with hyperparameter tuning."""
    print("  📊 Training LightGBM with hyperparameter tuning...")
    
    # Prepare categorical features (LightGBM uses column names)
    categorical_features = [X_train.columns[i] for i in categorical_indices]
    
    # Create datasets
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False
    )
    
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=categorical_features,
        free_raw_data=False,
        reference=train_data
    )
    
    # Hyperparameter search space
    # Try different combinations of key parameters
    param_grid = [
        {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
        },
        {
            'num_leaves': 50,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
        },
        {
            'num_leaves': 20,
            'learning_rate': 0.1,
            'feature_fraction': 0.95,
            'bagging_fraction': 0.85,
        }
    ]
    
    best_model = None
    best_score = float('inf')
    best_params = None
    
    # Try each parameter combination
    for params_base in param_grid:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            **params_base
        }
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        )
        
        # Evaluate on validation
        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = params
    
    print(f"    Best params: num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.3f}")
    print(f"    Best validation RMSE: ${best_score:,.2f}")
    
    # Predictions with best model
    y_pred_train = best_model.predict(X_train, num_iteration=best_model.best_iteration)
    y_pred_val = best_model.predict(X_val, num_iteration=best_model.best_iteration)
    
    return {
        'model': best_model,
        'model_type': 'lightgbm',
        'best_params': best_params,
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val
        },
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importance()))
    }


def train_catboost(X_train, y_train, X_val, y_val, categorical_indices):
    """Train a CatBoost model with hyperparameter tuning."""
    print("  📊 Training CatBoost with hyperparameter tuning...")
    
    # CatBoost uses indices for categorical features
    cat_features = categorical_indices
    
    # Hyperparameter search space
    param_grid = [
        {'learning_rate': 0.05, 'depth': 6},
        {'learning_rate': 0.03, 'depth': 8},
        {'learning_rate': 0.1, 'depth': 4},
    ]
    
    best_model = None
    best_score = float('inf')
    best_params = None
    
    # Try each parameter combination
    for params_base in param_grid:
        model = cb.CatBoostRegressor(
            iterations=1000,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=0,  # Suppress output during grid search
            early_stopping_rounds=50,
            cat_features=cat_features,
            allow_writing_files=False,  # Prevent catboost_info folder creation
            **params_base
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Evaluate on validation
        y_pred_val = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = params_base
    
    print(f"    Best params: lr={best_params['learning_rate']:.3f}, depth={best_params['depth']}")
    print(f"    Best validation RMSE: ${best_score:,.2f}")
    
    # Retrain with verbose output for final model
    final_model = cb.CatBoostRegressor(
        iterations=best_model.get_best_iteration(),
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
        cat_features=cat_features,
        allow_writing_files=False,  # Prevent catboost_info folder creation
        **best_params
    )
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    # Predictions
    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
    
    return {
        'model': final_model,
        'model_type': 'catboost',
        'best_params': best_params,
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val
        },
        'feature_importance': dict(zip(X_train.columns, final_model.get_feature_importance()))
    }


def train_xgboost(X_train, y_train, X_val, y_val, categorical_indices):
    """Train an XGBoost model with hyperparameter tuning."""
    print("  📊 Training XGBoost with hyperparameter tuning...")
    
    # XGBoost can handle categoricals natively (enable_categorical=True)
    # But we need to convert to category type
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    
    for idx in categorical_indices:
        col = X_train.columns[idx]
        X_train_xgb[col] = X_train_xgb[col].astype('category')
        X_val_xgb[col] = X_val_xgb[col].astype('category')
    
    # Hyperparameter search space
    param_grid = [
        {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8},
        {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.7},
        {'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9},
    ]
    
    best_model = None
    best_score = float('inf')
    best_params = None
    
    # Try each parameter combination
    for params_base in param_grid:
        model = xgb.XGBRegressor(
            n_estimators=1000,
            objective='reg:squarederror',
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=50,
            **params_base
        )
        
        model.fit(
            X_train_xgb, y_train,
            eval_set=[(X_val_xgb, y_val)],
            verbose=False
        )
        
        # Evaluate on validation
        y_pred_val = model.predict(X_val_xgb)
        val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = params_base
    
    print(f"    Best params: lr={best_params['learning_rate']:.3f}, depth={best_params['max_depth']}")
    print(f"    Best validation RMSE: ${best_score:,.2f}")
    
    # Retrain final model
    best_iteration = best_model.get_booster().num_boosted_rounds()
    final_model = xgb.XGBRegressor(
        n_estimators=best_iteration,
        objective='reg:squarederror',
        enable_categorical=True,
        random_state=42,
        **best_params
    )
    final_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
    
    # Predictions
    y_pred_train = final_model.predict(X_train_xgb)
    y_pred_val = final_model.predict(X_val_xgb)
    
    return {
        'model': final_model,
        'model_type': 'xgboost',
        'best_params': best_params,
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val
        },
        'feature_importance': dict(zip(X_train.columns, final_model.feature_importances_))
    }


def evaluate_model_predictions(actual_df, predictions, model_name):
    """Evaluate model predictions and return metrics."""
    # Create prediction DataFrame
    pred_df = actual_df[['store_id', 'dept_id', 'week_date']].copy()
    pred_df['predicted_sales'] = predictions
    
    # Evaluate
    metrics = evaluate_predictions_with_wmae(actual_df, pred_df, holiday_weight=5.0)
    metrics['model_name'] = model_name
    
    return metrics, pred_df


def create_ensemble(results, val_df, baseline_rmse, baseline_wmae):
    """
    Create ensemble of top-performing models.
    
    Strategy:
    - Only ensemble models that beat baseline
    - Try simple average and weighted average
    - Select best ensemble method
    """
    # Find models that beat baseline
    valid_models = []
    for model_name, result in results.items():
        metrics = result['metrics']
        if (metrics['rmse'] < baseline_rmse and metrics['wmae'] < baseline_wmae):
            valid_models.append((model_name, result))
    
    if len(valid_models) < 2:
        print("  ⚠️  Need at least 2 models beating baseline for ensemble")
        return None
    
    print(f"  📊 Ensembling {len(valid_models)} models: {[m[0] for m in valid_models]}")
    
    # Get predictions from valid models
    pred_dfs = []
    for model_name, result in valid_models:
        pred_dfs.append(result['predictions'].copy())
    
    # Strategy 1: Simple average
    ensemble_simple = pred_dfs[0].copy()
    for pred_df in pred_dfs[1:]:
        ensemble_simple['predicted_sales'] += pred_df['predicted_sales']
    ensemble_simple['predicted_sales'] /= len(pred_dfs)
    
    simple_metrics = evaluate_predictions_with_wmae(val_df, ensemble_simple, holiday_weight=5.0)
    
    # Strategy 2: Weighted average (weight by inverse RMSE - better models get more weight)
    weights = []
    for model_name, result in valid_models:
        rmse = result['metrics']['rmse']
        # Inverse RMSE as weight (lower RMSE = higher weight)
        weights.append(1.0 / rmse)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    ensemble_weighted = pred_dfs[0].copy()
    ensemble_weighted['predicted_sales'] = ensemble_weighted['predicted_sales'] * weights[0]
    for i, pred_df in enumerate(pred_dfs[1:], 1):
        ensemble_weighted['predicted_sales'] += pred_df['predicted_sales'] * weights[i]
    
    weighted_metrics = evaluate_predictions_with_wmae(val_df, ensemble_weighted, holiday_weight=5.0)
    
    # Select best ensemble method
    if weighted_metrics['rmse'] < simple_metrics['rmse']:
        best_ensemble = ensemble_weighted
        best_metrics = weighted_metrics
        ensemble_method = 'weighted'
        ensemble_weights = weights
    else:
        best_ensemble = ensemble_simple
        best_metrics = simple_metrics
        ensemble_method = 'simple'
        ensemble_weights = [1.0 / len(pred_dfs)] * len(pred_dfs)
    
    print(f"  ✅ Best ensemble method: {ensemble_method}")
    if ensemble_method == 'weighted':
        print(f"     Weights: {dict(zip([m[0] for m in valid_models], [f'{w:.3f}' for w in ensemble_weights]))}")
    print(f"     RMSE: ${best_metrics['rmse']:,.2f}, WMAE: ${best_metrics['wmae']:,.2f}")
    
    return {
        'model_result': {
            'model_type': 'ensemble',
            'ensemble_method': ensemble_method,
            'component_models': [m[0] for m in valid_models],
            'weights': ensemble_weights,
            'component_results': {m[0]: m[1] for m in valid_models}
        },
        'metrics': best_metrics,
        'predictions': best_ensemble
    }


def select_best_model(results, baseline_rmse, baseline_wmae):
    """
    Select best model based on custom rule:
    Must beat Seasonal Naive baseline on BOTH RMSE and WMAE.
    """
    print("\n" + "=" * 60)
    print("🏆 Model Selection")
    print("=" * 60)
    
    print(f"\nBaseline (Seasonal Naive) to beat:")
    print(f"  RMSE: ${baseline_rmse:,.2f}")
    print(f"  WMAE: ${baseline_wmae:,.2f}")
    
    # Filter models that beat baseline on both metrics
    valid_models = []
    for model_name, result in results.items():
        rmse = result['metrics']['rmse']
        wmae = result['metrics']['wmae']
        
        beats_rmse = rmse < baseline_rmse
        beats_wmae = wmae < baseline_wmae
        
        if beats_rmse and beats_wmae:
            valid_models.append((model_name, result))
            print(f"\n✅ {model_name}: Beats baseline on both metrics")
            print(f"   RMSE: ${rmse:,.2f} (baseline: ${baseline_rmse:,.2f})")
            print(f"   WMAE: ${wmae:,.2f} (baseline: ${baseline_wmae:,.2f})")
        else:
            print(f"\n❌ {model_name}: Does NOT beat baseline")
            print(f"   RMSE: ${rmse:,.2f} {'✅' if beats_rmse else '❌'}")
            print(f"   WMAE: ${wmae:,.2f} {'✅' if beats_wmae else '❌'}")
    
    if not valid_models:
        print("\n⚠️  WARNING: No models beat baseline on both metrics!")
        print("   Selecting best by RMSE anyway...")
        best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
    else:
        # Among valid models, pick best by RMSE
        best_model_name = min(valid_models, key=lambda x: x[1]['metrics']['rmse'])[0]
    
    print(f"\n🏆 Selected best model: {best_model_name}")
    print(f"   RMSE: ${results[best_model_name]['metrics']['rmse']:,.2f}")
    print(f"   WMAE: ${results[best_model_name]['metrics']['wmae']:,.2f}")
    
    return best_model_name


def main():
    """Main training function."""
    print("=" * 60)
    print("🤖 Model Training & Comparison")
    print("=" * 60)
    
    # Load feature-engineered data
    print("\n📥 Loading feature-engineered data...")
    train_df = pd.read_csv('data/splits/train_features.csv')
    val_df = pd.read_csv('data/splits/val_features.csv')
    test_df = pd.read_csv('data/splits/test_features.csv')
    
    # Convert dates
    for df in [train_df, val_df, test_df]:
        df['week_date'] = pd.to_datetime(df['week_date'])
    
    print(f"  ✅ Train: {len(train_df):,} rows")
    print(f"  ✅ Validation: {len(val_df):,} rows")
    print(f"  ✅ Test: {len(test_df):,} rows")
    
    # Load baseline results for comparison
    print("\n📊 Loading baseline results...")
    baseline_results = pd.read_csv('data/baseline_results.csv', index_col=0)
    seasonal_naive_rmse = baseline_results.loc['Seasonal Naive', 'rmse']
    seasonal_naive_wmae = baseline_results.loc['Seasonal Naive', 'wmae']
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, cat_indices = prepare_features(
        train_df, val_df, test_df
    )
    print(f"  ✅ Features: {len(feature_cols)}")
    print(f"  ✅ Categorical features: {len(cat_indices)}")
    
    # Train models
    print("\n" + "=" * 60)
    print("🚀 Training Models")
    print("=" * 60)
    
    results = {}
    
    # 1. Linear Model
    try:
        linear_result = train_linear_model(X_train, y_train, X_val, y_val)
        linear_metrics, linear_pred_df = evaluate_model_predictions(
            val_df, linear_result['predictions']['val'], 'Linear (Ridge)'
        )
        results['Linear (Ridge)'] = {
            'model_result': linear_result,
            'metrics': linear_metrics,
            'predictions': linear_pred_df
        }
        print(f"    RMSE: ${linear_metrics['rmse']:,.2f}, WMAE: ${linear_metrics['wmae']:,.2f}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    # 2. LightGBM
    try:
        lgb_result = train_lightgbm(X_train, y_train, X_val, y_val, cat_indices)
        lgb_metrics, lgb_pred_df = evaluate_model_predictions(
            val_df, lgb_result['predictions']['val'], 'LightGBM'
        )
        results['LightGBM'] = {
            'model_result': lgb_result,
            'metrics': lgb_metrics,
            'predictions': lgb_pred_df
        }
        print(f"    RMSE: ${lgb_metrics['rmse']:,.2f}, WMAE: ${lgb_metrics['wmae']:,.2f}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    # 3. CatBoost
    try:
        cb_result = train_catboost(X_train, y_train, X_val, y_val, cat_indices)
        cb_metrics, cb_pred_df = evaluate_model_predictions(
            val_df, cb_result['predictions']['val'], 'CatBoost'
        )
        results['CatBoost'] = {
            'model_result': cb_result,
            'metrics': cb_metrics,
            'predictions': cb_pred_df
        }
        print(f"    RMSE: ${cb_metrics['rmse']:,.2f}, WMAE: ${cb_metrics['wmae']:,.2f}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    # 4. XGBoost
    try:
        xgb_result = train_xgboost(X_train, y_train, X_val, y_val, cat_indices)
        xgb_metrics, xgb_pred_df = evaluate_model_predictions(
            val_df, xgb_result['predictions']['val'], 'XGBoost'
        )
        results['XGBoost'] = {
            'model_result': xgb_result,
            'metrics': xgb_metrics,
            'predictions': xgb_pred_df
        }
        print(f"    RMSE: ${xgb_metrics['rmse']:,.2f}, WMAE: ${xgb_metrics['wmae']:,.2f}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Try different ensemble combinations
    ensemble_results = {}
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("🎨 Creating Ensemble Models")
        print("=" * 60)
        
        # Ensemble 1: All tree models (LightGBM + CatBoost + XGBoost)
        all_tree_models = {k: v for k, v in results.items() 
                          if v['model_result']['model_type'] in ['lightgbm', 'catboost', 'xgboost']}
        if len(all_tree_models) >= 2:
            ensemble_all = create_ensemble(all_tree_models, val_df, seasonal_naive_rmse, seasonal_naive_wmae)
            if ensemble_all:
                comp_names = ensemble_all['model_result']['component_models']
                ensemble_name = f"Ensemble (All: {'+'.join(comp_names)})"
                ensemble_results[ensemble_name] = ensemble_all
                results[ensemble_name] = ensemble_all
        
        # Ensemble 2: Top 2 models only (XGBoost + LightGBM)
        top_2_models = {}
        sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['rmse'])
        for name, result in sorted_models[:2]:
            if result['model_result']['model_type'] in ['lightgbm', 'catboost', 'xgboost']:
                top_2_models[name] = result
        
        if len(top_2_models) >= 2:
            ensemble_top2 = create_ensemble(top_2_models, val_df, seasonal_naive_rmse, seasonal_naive_wmae)
            if ensemble_top2:
                comp_names = ensemble_top2['model_result']['component_models']
                ensemble_name = f"Ensemble (Top2: {'+'.join(comp_names)})"
                ensemble_results[ensemble_name] = ensemble_top2
                results[ensemble_name] = ensemble_top2
    
    # Select best model
    best_model_name = select_best_model(results, seasonal_naive_rmse, seasonal_naive_wmae)
    
    # Retrain all models on train+val, evaluate on test
    print("\n" + "=" * 60)
    print("🔄 Retraining All Models on Train+Val")
    print("=" * 60)
    
    # Combine train and val
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val, y_train_val, X_test_final, y_test_final, _, _ = prepare_features(
        train_val_df, test_df
    )
    
    # Store test predictions for all models
    all_test_predictions = {}
    
    # Retrain and get test predictions for all models that beat baseline
    valid_models = {k: v for k, v in results.items() 
                   if v['model_result']['model_type'] in ['lightgbm', 'catboost', 'xgboost']}
    
    for model_name, result in valid_models.items():
        model_result = result['model_result']
        model_type = model_result['model_type']
        
        print(f"\n📊 Retraining {model_name} ({model_type})...")
        
        try:
            if model_type == 'lightgbm':
                categorical_features = [X_train_val.columns[i] for i in cat_indices]
                train_data = lgb.Dataset(X_train_val, label=y_train_val, categorical_feature=categorical_features)
                
                best_params = model_result.get('best_params', {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                })
                
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42,
                    **best_params
                }
                
                final_model = lgb.train(params, train_data, num_boost_round=model_result['model'].best_iteration)
                y_test_pred = final_model.predict(X_test_final)
                
                # Only save if this is the best model
                if model_name == best_model_name:
                    final_model.save_model('models/best_model.txt')
                    with open('models/best_model_meta.pkl', 'wb') as f:
                        pickle.dump({
                            'model_type': 'lightgbm', 
                            'categorical_features': categorical_features,
                            'best_params': best_params
                        }, f)
            
            elif model_type == 'catboost':
                best_params = model_result.get('best_params', {'learning_rate': 0.05, 'depth': 6})
                
                final_model = cb.CatBoostRegressor(
                    iterations=model_result['model'].get_best_iteration(),
                    loss_function='RMSE',
                    random_seed=42,
                    cat_features=cat_indices,
                    verbose=False,
                    allow_writing_files=False,  # Prevent catboost_info folder creation
                    **best_params
                )
                final_model.fit(X_train_val, y_train_val, verbose=False)
                y_test_pred = final_model.predict(X_test_final)
                
                # Only save if this is the best model
                if model_name == best_model_name:
                    final_model.save_model('models/best_model.cbm')
                    with open('models/best_model_meta.pkl', 'wb') as f:
                        pickle.dump({
                            'model_type': 'catboost',
                            'best_params': best_params
                        }, f)
            
            elif model_type == 'xgboost':
                best_params = model_result.get('best_params', {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8})
                
                X_train_val_xgb = X_train_val.copy()
                X_test_xgb = X_test_final.copy()
                for idx in cat_indices:
                    col = X_train_val.columns[idx]
                    X_train_val_xgb[col] = X_train_val_xgb[col].astype('category')
                    X_test_xgb[col] = X_test_xgb[col].astype('category')
                
                best_iteration = model_result['model'].get_booster().num_boosted_rounds()
                final_model = xgb.XGBRegressor(
                    n_estimators=best_iteration,
                    objective='reg:squarederror',
                    enable_categorical=True,
                    random_state=42,
                    **best_params
                )
                final_model.fit(X_train_val_xgb, y_train_val, verbose=False)
                y_test_pred = final_model.predict(X_test_xgb)
                
                # Only save if this is the best model
                if model_name == best_model_name:
                    final_model.save_model('models/best_model.json')
                    with open('models/best_model_meta.pkl', 'wb') as f:
                        pickle.dump({
                            'model_type': 'xgboost',
                            'best_params': best_params
                        }, f)
            
            # Save test predictions for this model
            test_pred_df = test_df[['store_id', 'dept_id', 'week_date']].copy()
            test_pred_df['predicted_sales'] = y_test_pred
            all_test_predictions[model_name] = test_pred_df
            
        except Exception as e:
            print(f"  ❌ Error retraining {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Handle ensemble separately if it's the best model
    if best_model_name.startswith('Ensemble'):
        best_result = results[best_model_name]['model_result']
        print(f"\n📊 Retraining {best_model_name} (ensemble)...")
        print("  🎨 Retraining ensemble components...")
        
        component_models = best_result['component_models']
        component_results = best_result['component_results']
        ensemble_method = best_result['ensemble_method']
        weights = best_result['weights']
        
        # Retrain each component model
        component_predictions = []
        for comp_name in component_models:
            comp_result = component_results[comp_name]['model_result']
            comp_type = comp_result['model_type']
            
            print(f"    Retraining {comp_name} ({comp_type})...")
            
            if comp_type == 'lightgbm':
                categorical_features = [X_train_val.columns[i] for i in cat_indices]
                train_data = lgb.Dataset(X_train_val, label=y_train_val, categorical_feature=categorical_features)
                best_params = comp_result.get('best_params', {
                    'num_leaves': 31, 'learning_rate': 0.05,
                    'feature_fraction': 0.9, 'bagging_fraction': 0.8
                })
                params = {
                    'objective': 'regression', 'metric': 'rmse',
                    'boosting_type': 'gbdt', 'bagging_freq': 5,
                    'verbose': -1, 'random_state': 42, **best_params
                }
                comp_model = lgb.train(params, train_data, num_boost_round=comp_result['model'].best_iteration)
                comp_pred = comp_model.predict(X_test_final)
                
            elif comp_type == 'catboost':
                best_params = comp_result.get('best_params', {'learning_rate': 0.05, 'depth': 6})
                comp_model = cb.CatBoostRegressor(
                    iterations=comp_result['model'].get_best_iteration(),
                    loss_function='RMSE', random_seed=42,
                    cat_features=cat_indices, verbose=False,
                    allow_writing_files=False,  # Prevent catboost_info folder creation
                    **best_params
                )
                comp_model.fit(X_train_val, y_train_val, verbose=False)
                comp_pred = comp_model.predict(X_test_final)
            
            elif comp_type == 'xgboost':
                X_train_val_xgb = X_train_val.copy()
                X_test_xgb = X_test_final.copy()
                for idx in cat_indices:
                    col = X_train_val.columns[idx]
                    X_train_val_xgb[col] = X_train_val_xgb[col].astype('category')
                    X_test_xgb[col] = X_test_xgb[col].astype('category')
                
                best_params = comp_result.get('best_params', {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8})
                best_iteration = comp_result['model'].get_booster().num_boosted_rounds()
                comp_model = xgb.XGBRegressor(
                    n_estimators=best_iteration,
                    objective='reg:squarederror',
                    enable_categorical=True,
                    random_state=42,
                    **best_params
                )
                comp_model.fit(X_train_val_xgb, y_train_val, verbose=False)
                comp_pred = comp_model.predict(X_test_xgb)
            
            component_predictions.append(comp_pred)
        
        # Combine predictions
        if ensemble_method == 'weighted':
            y_test_pred = np.zeros(len(X_test_final))
            for pred, weight in zip(component_predictions, weights):
                y_test_pred += pred * weight
        else:  # simple average
            y_test_pred = np.mean(component_predictions, axis=0)
        
        # Save ensemble test predictions
        test_pred_df = test_df[['store_id', 'dept_id', 'week_date']].copy()
        test_pred_df['predicted_sales'] = y_test_pred
        all_test_predictions[best_model_name] = test_pred_df
        
        # Save ensemble metadata
        with open('models/best_model_meta.pkl', 'wb') as f:
            pickle.dump({
                'model_type': 'ensemble',
                'ensemble_method': ensemble_method,
                'component_models': component_models,
                'weights': weights
            }, f)
    
    # Save all test predictions to folder
    print("\n💾 Saving all test predictions...")
    os.makedirs('data/test_predictions_all', exist_ok=True)
    for model_name, pred_df in all_test_predictions.items():
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        filename = f"data/test_predictions_all/{safe_name}.csv"
        pred_df.to_csv(filename, index=False)
        print(f"  ✅ Saved: {filename}")
    
    # Save metadata
    import json
    metadata = {
        'models': list(all_test_predictions.keys()),
        'n_samples': len(test_df)
    }
    with open('data/test_predictions_all/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Saved: data/test_predictions_all/metadata.json")
    
    # Get best model's test predictions for final evaluation
    best_test_pred_df = all_test_predictions.get(best_model_name)
    if best_test_pred_df is None:
        # Fallback: use first available
        best_test_pred_df = list(all_test_predictions.values())[0]
    
    # Save best model's predictions to main location (for backward compatibility)
    best_test_pred_df.to_csv('data/test_predictions.csv', index=False)
    
    # Evaluate best model on test set
    test_metrics = evaluate_predictions_with_wmae(test_df, best_test_pred_df, holiday_weight=5.0)
    
    print("\n📊 Test Set Performance:")
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  MAE: ${test_metrics['mae']:,.2f}")
    print(f"  WMAE: ${test_metrics['wmae']:,.2f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  R²: {test_metrics['r_squared']:.4f}")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Model comparison
    comparison_data = []
    for model_name, result in results.items():
        metrics = result['metrics'].copy()
        metrics['model_name'] = model_name  # Ensure model name is included
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('data/model_comparison.csv', index=False)
    print(f"  ✅ Saved: data/model_comparison.csv")
    
    # Final test results
    test_results = {
        'model': best_model_name,
        'test_metrics': test_metrics
    }
    with open('data/final_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"  ✅ Saved: data/final_test_results.json")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
