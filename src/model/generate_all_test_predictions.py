"""
Generate test predictions for all models for comparison.
This script retrains all models on train+val and generates test predictions.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_splits_dir, get_data_dir
from src.model.train import prepare_features
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

def generate_all_test_predictions():
    """Generate test predictions for all models."""
    print("=" * 60)
    print("🔄 Generating Test Predictions for All Models")
    print("=" * 60)
    
    # Load data (paths overridable via SPLITS_DIR on SageMaker)
    splits_dir = get_splits_dir()
    data_dir = get_data_dir()
    print("\n📥 Loading data...")
    train_df = pd.read_csv(splits_dir / 'train_features.csv')
    val_df = pd.read_csv(splits_dir / 'val_features.csv')
    test_df = pd.read_csv(splits_dir / 'test_features.csv')
    
    train_df['week_date'] = pd.to_datetime(train_df['week_date'])
    val_df['week_date'] = pd.to_datetime(val_df['week_date'])
    test_df['week_date'] = pd.to_datetime(test_df['week_date'])
    
    # Load model comparison results to get best params
    print("📊 Loading model results...")
    comparison_df = pd.read_csv(data_dir / 'model_comparison.csv')
    
    # Combine train and val
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Prepare features
    print("🔧 Preparing features...")
    X_train_val, y_train_val, X_test, y_test, cat_indices, feature_cols = prepare_features(
        train_val_df, test_df
    )
    
    # Load saved model results to get best params
    # We'll need to reconstruct from model_comparison.csv or save params separately
    # For now, use reasonable defaults and retrain
    
    all_predictions = {}
    
    # 1. LightGBM
    if 'LightGBM' in comparison_df['model_name'].values:
        print("\n📊 Retraining LightGBM...")
        try:
            categorical_features = [X_train_val.columns[i] for i in cat_indices]
            train_data = lgb.Dataset(X_train_val, label=y_train_val, categorical_feature=categorical_features)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
            }
            
            model = lgb.train(params, train_data, num_boost_round=631)  # Use best iteration from training
            pred = model.predict(X_test)
            
            pred_df = test_df[['store_id', 'dept_id', 'week_date']].copy()
            pred_df['predicted_sales'] = pred
            all_predictions['LightGBM'] = pred_df
            print("  ✅ LightGBM predictions generated")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 2. CatBoost
    if 'CatBoost' in comparison_df['model_name'].values:
        print("\n📊 Retraining CatBoost...")
        try:
            model = cb.CatBoostRegressor(
                iterations=998,
                learning_rate=0.03,
                depth=8,
                loss_function='RMSE',
                random_seed=42,
                cat_features=cat_indices,
                verbose=False
            )
            model.fit(X_train_val, y_train_val, verbose=False)
            pred = model.predict(X_test)
            
            pred_df = test_df[['store_id', 'dept_id', 'week_date']].copy()
            pred_df['predicted_sales'] = pred
            all_predictions['CatBoost'] = pred_df
            print("  ✅ CatBoost predictions generated")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 3. XGBoost
    if 'XGBoost' in comparison_df['model_name'].values:
        print("\n📊 Retraining XGBoost...")
        try:
            X_train_val_xgb = X_train_val.copy()
            X_test_xgb = X_test.copy()
            for idx in cat_indices:
                col = X_train_val.columns[idx]
                X_train_val_xgb[col] = X_train_val_xgb[col].astype('category')
                X_test_xgb[col] = X_test_xgb[col].astype('category')
            
            model = xgb.XGBRegressor(
                n_estimators=998,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.7,
                objective='reg:squarederror',
                enable_categorical=True,
                random_state=42
            )
            model.fit(X_train_val_xgb, y_train_val, verbose=False)
            pred = model.predict(X_test_xgb)
            
            pred_df = test_df[['store_id', 'dept_id', 'week_date']].copy()
            pred_df['predicted_sales'] = pred
            all_predictions['XGBoost'] = pred_df
            print("  ✅ XGBoost predictions generated")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 4. Ensemble (if exists)
    ensemble_models = [name for name in comparison_df['model_name'].values if 'Ensemble' in str(name)]
    if ensemble_models:
        print("\n📊 Generating Ensemble predictions...")
        try:
            # Get component models
            component_models = ['LightGBM', 'CatBoost', 'XGBoost']
            component_preds = []
            weights = [0.338, 0.315, 0.347]  # From training output
            
            for comp_name in component_models:
                if comp_name in all_predictions:
                    component_preds.append(all_predictions[comp_name]['predicted_sales'].values)
            
            if len(component_preds) >= 2:
                # Weighted average
                ensemble_pred = np.zeros(len(component_preds[0]))
                total_weight = sum(weights[:len(component_preds)])
                for i, pred in enumerate(component_preds):
                    ensemble_pred += pred * (weights[i] / total_weight)
                
                pred_df = test_df[['store_id', 'dept_id', 'week_date']].copy()
                pred_df['predicted_sales'] = ensemble_pred
                all_predictions['Ensemble (All: LightGBM+CatBoost+XGBoost)'] = pred_df
                print("  ✅ Ensemble predictions generated")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Save all predictions (overridable via DATA_DIR on SageMaker)
    test_pred_dir = data_dir / 'test_predictions_all'
    print("\n💾 Saving predictions...")
    os.makedirs(test_pred_dir, exist_ok=True)
    
    for model_name, pred_df in all_predictions.items():
        filename = test_pred_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}.csv"
        pred_df.to_csv(filename, index=False)
        print(f"  ✅ Saved: {filename}")
    
    # Save metadata
    metadata = {
        'models': list(all_predictions.keys()),
        'n_samples': len(test_df)
    }
    with open(test_pred_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ All test predictions generated!")
    print("=" * 60)
    print(f"\n📊 Generated predictions for {len(all_predictions)} models")
    print(f"📁 Saved to: {test_pred_dir}")


if __name__ == "__main__":
    generate_all_test_predictions()
