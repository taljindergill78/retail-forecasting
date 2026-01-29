"""
Comprehensive model evaluation with visualizations.

This script:
1. Loads model predictions and actuals
2. Computes metrics overall and by segments
3. Produces diagnostic plots
4. Saves evaluation report
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.evaluate import evaluate_predictions_with_wmae, evaluate_by_segments

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

REPORTS_DIR = project_root / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_actual_vs_predicted(actual_df, pred_df, model_name, n_samples=5):
    """
    Plot actual vs predicted for sample series.
    
    Args:
        actual_df: DataFrame with actual values
        pred_df: DataFrame with predictions
        model_name: Name of model (for title)
        n_samples: Number of store-dept combinations to plot
    """
    # Merge
    merged = pd.merge(
        actual_df,
        pred_df,
        on=['store_id', 'dept_id', 'week_date'],
        how='inner'
    )
    
    # Get top N series by total sales
    series_totals = merged.groupby(['store_id', 'dept_id'])['weekly_sales'].sum().sort_values(ascending=False)
    top_series = series_totals.head(n_samples).index
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for idx, (store_id, dept_id) in enumerate(top_series):
        series_data = merged[
            (merged['store_id'] == store_id) & 
            (merged['dept_id'] == dept_id)
        ].sort_values('week_date')
        
        ax = axes[idx]
        ax.plot(series_data['week_date'], series_data['weekly_sales'], 
               label='Actual', linewidth=2, marker='o', markersize=4)
        ax.plot(series_data['week_date'], series_data['predicted_sales'], 
               label='Predicted', linewidth=2, marker='s', markersize=4, linestyle='--')
        ax.set_title(f'Store {store_id}, Dept {dept_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Weekly Sales ($)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name}: Actual vs Predicted (Sample Series)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name.replace(' ', '_')}_actual_vs_predicted.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_over_time(actual_df, pred_df, model_name):
    """Plot residuals over time."""
    merged = pd.merge(
        actual_df,
        pred_df,
        on=['store_id', 'dept_id', 'week_date'],
        how='inner'
    )
    
    merged['residual'] = merged['weekly_sales'] - merged['predicted_sales']
    merged = merged.sort_values('week_date')
    
    plt.figure(figsize=(14, 6))
    plt.scatter(merged['week_date'], merged['residual'], alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title(f'{model_name}: Residuals Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name.replace(' ', '_')}_residuals.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_by_holiday(actual_df, pred_df, model_name):
    """Plot error distribution by holiday vs non-holiday."""
    merged = pd.merge(
        actual_df,
        pred_df,
        on=['store_id', 'dept_id', 'week_date'],
        how='inner'
    )
    
    merged['error'] = np.abs(merged['weekly_sales'] - merged['predicted_sales'])
    merged['holiday_label'] = merged['isholiday'].map({True: 'Holiday', False: 'Non-Holiday'})
    
    plt.figure(figsize=(10, 6))
    merged.boxplot(column='error', by='holiday_label', ax=plt.gca())
    plt.title(f'{model_name}: Error Distribution by Holiday Status', fontsize=16, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Holiday Status', fontsize=12)
    plt.ylabel('Absolute Error ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name.replace(' ', '_')}_error_by_holiday.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model_result, model_name, top_n=20):
    """Plot feature importance for tree models."""
    if 'feature_importance' not in model_result:
        return
    
    importance = model_result['feature_importance']
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'{model_name}: Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{model_name.replace(' ', '_')}_feature_importance.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(actual_df, all_predictions, n_samples=3):
    """
    Plot comparison of all models vs actual for sample series.
    
    Args:
        actual_df: DataFrame with actual values
        all_predictions: Dict of {model_name: pred_df}
        n_samples: Number of store-dept combinations to plot
    """
    # Get top N series by total sales
    series_totals = actual_df.groupby(['store_id', 'dept_id'])['weekly_sales'].sum().sort_values(ascending=False)
    top_series = series_totals.head(n_samples).index
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 5 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_predictions)))
    model_colors = dict(zip(all_predictions.keys(), colors))
    
    for idx, (store_id, dept_id) in enumerate(top_series):
        ax = axes[idx]
        
        # Plot actual
        actual_series = actual_df[
            (actual_df['store_id'] == store_id) & 
            (actual_df['dept_id'] == dept_id)
        ].sort_values('week_date')
        
        ax.plot(actual_series['week_date'], actual_series['weekly_sales'], 
               label='Actual', linewidth=3, marker='o', markersize=6, color='black', zorder=10)
        
        # Plot each model
        for model_name, pred_df in all_predictions.items():
            pred_series = pred_df[
                (pred_df['store_id'] == store_id) & 
                (pred_df['dept_id'] == dept_id)
            ].sort_values('week_date')
            
            if len(pred_series) > 0:
                ax.plot(pred_series['week_date'], pred_series['predicted_sales'], 
                       label=model_name, linewidth=2, linestyle='--', alpha=0.8,
                       color=model_colors[model_name], zorder=5)
        
        ax.set_title(f'Store {store_id}, Dept {dept_id} - All Models Comparison', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Weekly Sales ($)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Comparison: All Models vs Actual (Sample Series)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_models_comparison.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(all_metrics):
    """
    Plot comparison of metrics across all models.
    
    Args:
        all_metrics: Dict of {model_name: metrics_dict}
    """
    models = list(all_metrics.keys())
    metrics_to_plot = ['rmse', 'wmae', 'mae', 'r_squared']
    metric_labels = ['RMSE ($)', 'WMAE ($)', 'MAE ($)', 'R²']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = [all_metrics[model][metric] for model in models]
        
        bars = ax.bar(models, values, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color bars - best model in green, others in blue
        best_idx = np.argmin(values) if metric != 'r_squared' else np.argmax(values)
        for i, bar in enumerate(bars):
            if i == best_idx:
                bar.set_color('green')
                bar.set_alpha(0.8)
            else:
                bar.set_color('steelblue')
        
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if metric == 'r_squared':
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Metrics Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_comparison.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_comparison(actual_df, all_predictions):
    """
    Plot residuals comparison across all models.
    
    Args:
        actual_df: DataFrame with actual values
        all_predictions: Dict of {model_name: pred_df}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (model_name, pred_df) in enumerate(all_predictions.items()):
        if idx >= 4:  # Only plot first 4 models
            break
        
        ax = axes[idx]
        
        merged = pd.merge(
            actual_df,
            pred_df,
            on=['store_id', 'dept_id', 'week_date'],
            how='inner'
        )
        
        merged['residual'] = merged['weekly_sales'] - merged['predicted_sales']
        merged = merged.sort_values('week_date')
        
        ax.scatter(merged['week_date'], merged['residual'], alpha=0.3, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_title(f'{model_name}: Residuals Over Time', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Residuals Comparison Across Models', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_comparison.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def create_evaluation_report(results, output_file='reports/model_evaluation_report.csv'):
    """Create comprehensive evaluation report."""
    report_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        report_data.append({
            'model': model_name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'wmae': metrics['wmae'],
            'mape': metrics['mape'],
            'r_squared': metrics['r_squared'],
            'n_samples': metrics['n_samples']
        })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_file, index=False)
    print(f"  ✅ Saved: {output_file}")
    
    return report_df


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("📊 Model Evaluation & Visualization (All Models)")
    print("=" * 60)
    
    # Load data
    print("\n📥 Loading data...")
    try:
        test_df = pd.read_csv('data/splits/test_features.csv')
        test_df['week_date'] = pd.to_datetime(test_df['week_date'])
        print(f"  ✅ Test data: {len(test_df):,} rows")
    except FileNotFoundError:
        print("  ❌ Error: test_features.csv not found. Run feature engineering first.")
        return
    
    # Load all model predictions
    print("\n📊 Loading all model predictions...")
    all_predictions = {}
    
    # Try to load from test_predictions_all folder first
    predictions_dir = Path('data/test_predictions_all')
    if predictions_dir.exists():
        # Try to load from metadata.json first (if exists)
        metadata_file = predictions_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for model_name in metadata['models']:
                    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
                    pred_file = predictions_dir / f"{safe_name}.csv"
                    if pred_file.exists():
                        pred_df = pd.read_csv(pred_file)
                        pred_df['week_date'] = pd.to_datetime(pred_df['week_date'])
                        all_predictions[model_name] = pred_df
                        print(f"  ✅ Loaded: {model_name}")
            except Exception as e:
                print(f"  ⚠️  Could not load from metadata.json: {e}")
        
        # Fallback: Load all CSV files directly from directory
        if not all_predictions:
            csv_files = list(predictions_dir.glob('*.csv'))
            if csv_files:
                print("  📂 Loading CSV files directly from directory...")
                for csv_file in csv_files:
                    # Extract model name from filename
                    model_name = csv_file.stem.replace('_', ' ')
                    # Try to reconstruct original name
                    if 'LightGBM' in csv_file.stem:
                        model_name = 'LightGBM'
                    elif 'CatBoost' in csv_file.stem:
                        model_name = 'CatBoost'
                    elif 'XGBoost' in csv_file.stem:
                        model_name = 'XGBoost'
                    elif 'Ensemble' in csv_file.stem:
                        # Try to get full ensemble name from model_comparison.csv
                        try:
                            comp_df = pd.read_csv('data/model_comparison.csv')
                            ensemble_names = comp_df[comp_df['model_name'].str.contains('Ensemble', na=False)]['model_name'].values
                            if len(ensemble_names) > 0:
                                model_name = ensemble_names[0]
                            else:
                                model_name = 'Ensemble'
                        except:
                            model_name = 'Ensemble'
                    
                    pred_df = pd.read_csv(csv_file)
                    pred_df['week_date'] = pd.to_datetime(pred_df['week_date'])
                    all_predictions[model_name] = pred_df
                    print(f"  ✅ Loaded: {model_name}")
    
    # Fallback: Load best model predictions
    if not all_predictions:
        try:
            test_pred_df = pd.read_csv('data/test_predictions.csv')
            test_pred_df['week_date'] = pd.to_datetime(test_pred_df['week_date'])
            
            # Load final test results to get model name
            import json
            with open('data/final_test_results.json', 'r') as f:
                final_results = json.load(f)
            model_name = final_results['model']
            all_predictions[model_name] = test_pred_df
            print(f"  ✅ Loaded: {model_name} (best model only)")
            print("  💡 Tip: Run src/model/generate_all_test_predictions.py to generate predictions for all models")
        except FileNotFoundError:
            print("  ❌ Error: No predictions found. Run train.py first.")
            return
    
    if not all_predictions:
        print("  ❌ Error: No model predictions found.")
        return
    
    # Load final test results for best model name
    try:
        import json
        with open('data/final_test_results.json', 'r') as f:
            final_results = json.load(f)
        best_model_name = final_results['model']
        print(f"\n  🏆 Best model: {best_model_name}")
    except FileNotFoundError:
        best_model_name = list(all_predictions.keys())[0]
        print(f"\n  ⚠️  Using first model as best: {best_model_name}")
    
    # Evaluate all models
    print("\n📊 Evaluating all models...")
    from src.model.evaluate import evaluate_predictions_with_wmae, evaluate_by_segments
    
    all_metrics = {}
    results = {}
    
    for model_name, pred_df in all_predictions.items():
        print(f"  📈 Evaluating {model_name}...")
        metrics = evaluate_predictions_with_wmae(
            test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales', 'isholiday']],
            pred_df,
            holiday_weight=5.0
        )
        all_metrics[model_name] = metrics
        results[model_name] = {'metrics': metrics, 'predictions': pred_df}
        print(f"    RMSE: ${metrics['rmse']:,.2f}, WMAE: ${metrics['wmae']:,.2f}, R²: {metrics['r_squared']:.4f}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    # 1. Model comparison - all models vs actual
    print("  📊 Plotting all models comparison...")
    plot_model_comparison(
        test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales']],
        all_predictions,
        n_samples=3
    )
    
    # 2. Metrics comparison
    print("  📊 Plotting metrics comparison...")
    plot_metrics_comparison(all_metrics)
    
    # 3. Residuals comparison
    print("  📊 Plotting residuals comparison...")
    plot_residuals_comparison(
        test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales']],
        all_predictions
    )
    
    # 4. Individual model visualizations (for best model)
    print(f"  📊 Plotting individual visualizations for {best_model_name}...")
    best_pred_df = all_predictions[best_model_name]
    
    plot_actual_vs_predicted(
        test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales']],
        best_pred_df,
        best_model_name,
        n_samples=5
    )
    
    plot_residuals_over_time(
        test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales']],
        best_pred_df,
        best_model_name
    )
    
    plot_error_by_holiday(
        test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales', 'isholiday']],
        best_pred_df,
        best_model_name
    )
    
    # 5. Feature importance (for best model if XGBoost)
    print("  📊 Plotting feature importance...")
    try:
        import pickle
        with open('models/best_model_meta.pkl', 'rb') as f:
            model_meta = pickle.load(f)
        
        if model_meta['model_type'] == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model('models/best_model.json')
            
            feature_cols = [col for col in test_df.columns 
                          if col not in ['store_id', 'dept_id', 'week_date', 'weekly_sales']]
            
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            plot_feature_importance(
                {'feature_importance': feature_importance},
                best_model_name,
                top_n=20
            )
        else:
            print("    ⚠️  Feature importance plotting currently supports XGBoost only")
    except Exception as e:
        print(f"    ⚠️  Could not plot feature importance: {e}")
    
    # 6. Create comprehensive evaluation report
    print("\n📝 Creating evaluation report...")
    report_df = create_evaluation_report(results, 'reports/model_evaluation_report.csv')
    
    # Segmented evaluation for best model
    print("\n📊 Creating segmented evaluation...")
    segmented_results = evaluate_by_segments(
        test_df[['store_id', 'dept_id', 'week_date', 'weekly_sales', 'isholiday']],
        best_pred_df,
        segment_col='isholiday',
        holiday_weight=5.0
    )
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
    print(f"\n📁 Visualizations saved to: {FIGURES_DIR}")
    print(f"📁 Report saved to: reports/model_evaluation_report.csv")
    print(f"\n📊 Model Performance Summary:")
    for model_name, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['rmse']):
        marker = "🏆" if model_name == best_model_name else "  "
        print(f"{marker} {model_name}:")
        print(f"     RMSE: ${metrics['rmse']:,.2f}, WMAE: ${metrics['wmae']:,.2f}, R²: {metrics['r_squared']:.4f}")


if __name__ == "__main__":
    main()
