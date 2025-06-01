#!/usr/bin/env python3
"""
AutoGluon Time Series Forecasting Script
Forecasts at SKU-day level using deep learning models
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import numpy as np
import holidays
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def extract_static_features(
    df: pd.DataFrame,
    id_col: str,
    static_feature_cols: list
) -> pd.DataFrame:
    """Extract static features into a separate DataFrame"""
    if not static_feature_cols:
        return None
    
    # Get unique values per item_id for static features
    static_df = df.groupby(id_col)[static_feature_cols].first().reset_index()
    
    # AutoGluon expects the id column to be present, not just as index
    # Keep id_col as a regular column
    
    return static_df


def create_timeseries_dataframe(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    id_col: str,
    static_features_df: pd.DataFrame = None
) -> TimeSeriesDataFrame:
    """Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame"""
    # Remove static features from main dataframe if they exist
    # But keep id_column and timestamp_column
    if static_features_df is not None:
        static_cols = static_features_df.columns.tolist()
        # Remove id_col from static_cols to keep it in the main dataframe
        static_cols = [col for col in static_cols if col != id_col]
        cols_to_keep = [col for col in df.columns if col not in static_cols]
        df = df[cols_to_keep]
    
    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column=id_col,
        timestamp_column=timestamp_col,
        static_features_df=static_features_df
    )


def train_forecast_model(
    train_data: TimeSeriesDataFrame,
    config: dict,
    logger: logging.Logger
) -> TimeSeriesPredictor:
    """Train AutoGluon TimeSeriesPredictor with DL models only"""
    
    model_config = config['model']
    
    # Deep learning models in AutoGluon
    dl_models = [
        'DeepAR',
        'SimpleFeedForward', 
        'TemporalFusionTransformer',
        'PatchTST',
        'DLinear',
        'TiDE',
        'TimesNet',
        'Chronos'
    ]
    
    # Filter to only requested DL models
    models_to_use = [m for m in dl_models if m in model_config.get('models', dl_models)]
    
    logger.info(f"Training with models: {models_to_use}")
    
    # Get covariate names from config
    known_covariates = config['features'].get('known_covariates', [])
    static_features = config['features'].get('static_features', [])
    
    # Filter to only include features that exist in the data
    available_columns = set(train_data.columns)
    known_covariates = [col for col in known_covariates if col in available_columns]
    
    # Get quantile levels from config
    quantile_levels = model_config.get('quantile_levels', None)
    
    # Log available features for debugging
    logger.info(f"Available columns in training data: {list(available_columns)}")
    logger.info(f"Known covariates to use: {known_covariates}")
    
    predictor = TimeSeriesPredictor(
        target=config['data']['target_column'],
        known_covariates_names=known_covariates,
        prediction_length=model_config['prediction_length'],
        eval_metric=model_config.get('eval_metric', 'MAE'),
        path=model_config['save_path'],
        quantile_levels=quantile_levels,
        verbosity=2
    )
    
    # Get model-specific hyperparameters
    hyperparameters = {}
    use_custom_hyperparams = model_config.get('use_custom_hyperparameters', True)
    
    if use_custom_hyperparams:
        for model in models_to_use:
            if model in model_config.get('hyperparameters', {}):
                hyperparameters[model] = model_config['hyperparameters'][model].copy()
                
                # Apply asymmetric loss configuration if enabled
                asymmetric_config = model_config.get('asymmetric_loss', {})
                if asymmetric_config.get('enabled', False):
                    # For models that support custom loss functions
                    if model in ['DeepAR', 'TemporalFusionTransformer']:
                        # Adjust quantile levels to emphasize under-prediction penalty
                        if 'quantile_levels' not in hyperparameters[model]:
                            hyperparameters[model]['quantile_levels'] = [0.05, 0.2, 0.5, 0.8, 0.95]
                        
                        # Add custom loss configuration
                        hyperparameters[model]['loss_function'] = 'QuantileLoss'
                        
                    # For all models, we can adjust learning rate and regularization
                    # to make models more conservative
                    over_penalty = asymmetric_config.get('over_penalty_factor', 1.5)
                    if over_penalty > 1.0:
                        # Reduce learning rate slightly to make training more conservative
                        if 'lr' in hyperparameters[model]:
                            hyperparameters[model]['lr'] *= 0.8
                        
                        # Increase regularization to prevent overfitting to optimistic patterns
                        if 'dropout_rate' in hyperparameters[model]:
                            hyperparameters[model]['dropout_rate'] = min(
                                hyperparameters[model]['dropout_rate'] * 1.2, 
                                0.5
                            )
                
                logger.info(f"Using custom hyperparameters for {model}: {hyperparameters[model]}")
            else:
                hyperparameters[model] = {}
                logger.info(f"No custom hyperparameters found for {model}, using defaults")
    else:
        logger.info("Using default hyperparameters for all models")
        hyperparameters = {model: {} for model in models_to_use}
    
    predictor.fit(
        train_data,
        hyperparameters=hyperparameters,
        #time_limit=model_config.get('time_limit', 3600),
        num_val_windows=model_config.get('num_val_windows', 1),
        val_step_size=model_config.get('val_step_size', 1),
        enable_ensemble=model_config.get('enable_ensemble', True)
    )
    
    return predictor


def create_future_known_covariates(
    test_data: TimeSeriesDataFrame,
    prediction_length: int,
    known_covariate_names: list,
    config: dict
) -> TimeSeriesDataFrame:
    """Create future known covariates for prediction"""
    
    # Get the last timestamp for each item
    # In TimeSeriesDataFrame, timestamp is part of the multi-index
    last_timestamps = test_data.reset_index(level='timestamp').groupby(level='item_id')['timestamp'].last()
    
    # Load future promo data if available
    future_promo_df = None
    future_promo_path = config['data'].get('future_promo_path')
    if future_promo_path and os.path.exists(future_promo_path):
        future_promo_df = pd.read_csv(future_promo_path)
        future_promo_df[config['data']['timestamp_column']] = pd.to_datetime(
            future_promo_df[config['data']['timestamp_column']]
        )
    
    # Create future dates for each item
    future_dfs = []
    
    for item_id, last_timestamp in last_timestamps.items():
        # Generate future dates
        future_dates = pd.date_range(
            start=last_timestamp + pd.Timedelta(days=1),
            periods=prediction_length,
            freq='D'
        )
        
        # Create dataframe for this item
        item_df = pd.DataFrame({
            config['data']['id_column']: item_id,
            config['data']['timestamp_column']: future_dates
        })
        
        # Add calendar features
        item_df['day_of_week'] = future_dates.dayofweek
        item_df['day_of_month'] = future_dates.day
        item_df['month'] = future_dates.month
        item_df['quarter'] = future_dates.quarter
        item_df['week_of_year'] = future_dates.isocalendar().week
        item_df['year'] = future_dates.year
        item_df['day_of_year'] = future_dates.dayofyear
        
        # Weekend flag
        item_df['is_weekend'] = (item_df['day_of_week'] >= 5).astype(int)
        
        # Month start/end flags
        item_df['is_month_start'] = (future_dates.day <= 5).astype(int)
        item_df['is_month_end'] = (future_dates.day >= 25).astype(int)
        
        # Cyclical encoding
        item_df['day_of_week_sin'] = np.sin(2 * np.pi * item_df['day_of_week'] / 7)
        item_df['day_of_week_cos'] = np.cos(2 * np.pi * item_df['day_of_week'] / 7)
        item_df['month_sin'] = np.sin(2 * np.pi * item_df['month'] / 12)
        item_df['month_cos'] = np.cos(2 * np.pi * item_df['month'] / 12)
        
        # Advanced Fourier features for multiple seasonality periods
        item_df['day_of_year_sin_weekly'] = np.sin(2 * np.pi * item_df['day_of_year'] / 7)
        item_df['day_of_year_cos_weekly'] = np.cos(2 * np.pi * item_df['day_of_year'] / 7)
        item_df['day_of_year_sin_monthly'] = np.sin(2 * np.pi * item_df['day_of_year'] / 30.44)
        item_df['day_of_year_cos_monthly'] = np.cos(2 * np.pi * item_df['day_of_year'] / 30.44)
        item_df['day_of_year_sin_quarterly'] = np.sin(2 * np.pi * item_df['day_of_year'] / 91.31)
        item_df['day_of_year_cos_quarterly'] = np.cos(2 * np.pi * item_df['day_of_year'] / 91.31)
        
        # Holiday features
        us_holidays = holidays.US()
        item_df['is_holiday'] = future_dates.to_series().apply(lambda x: x.date() in us_holidays).astype(int)
        
        # Calculate days to nearest holiday
        holiday_dates = [d for d in pd.date_range(future_dates[0] - pd.Timedelta(days=30), 
                                                  future_dates[-1] + pd.Timedelta(days=30), 
                                                  freq='D').date if d in us_holidays]
        
        if holiday_dates:
            item_df['days_to_holiday'] = future_dates.to_series().apply(
                lambda x: min([abs((h - x.date()).days) for h in holiday_dates])
            )
        else:
            item_df['days_to_holiday'] = 999
            
        item_df['is_pre_holiday'] = (item_df['days_to_holiday'] == 1).astype(int)
        item_df['is_post_holiday'] = item_df['is_pre_holiday'].shift(-1, fill_value=0)
        
        # Interaction features
        item_df['weekend_month_interaction'] = item_df['is_weekend'] * item_df['month']
        item_df['holiday_weekend_interaction'] = item_df['is_holiday'] * item_df['is_weekend']
        
        # Get promo features from file or use defaults
        if future_promo_df is not None:
            # Merge with future promo data
            promo_cols = ['promo_flag', 'promo_discount', 'special_event']
            item_promo_df = future_promo_df[
                future_promo_df[config['data']['id_column']] == item_id
            ]
            
            if not item_promo_df.empty:
                # Merge promo data for this item
                item_df = item_df.merge(
                    item_promo_df[[config['data']['timestamp_column']] + promo_cols],
                    on=config['data']['timestamp_column'],
                    how='left'
                )
                # Fill any missing values with defaults
                item_df['promo_flag'] = item_df['promo_flag'].fillna(0).astype(int)
                item_df['promo_discount'] = item_df['promo_discount'].fillna(0.0)
                item_df['special_event'] = item_df['special_event'].fillna(0).astype(int)
            else:
                # No promo data for this item, use defaults
                item_df['promo_flag'] = 0
                item_df['promo_discount'] = 0.0
                item_df['special_event'] = 0
        else:
            # No promo file provided, use defaults
            item_df['promo_flag'] = 0
            item_df['promo_discount'] = 0.0
            item_df['special_event'] = 0
        
        # Generate interaction features
        item_df['promo_dow_interaction'] = item_df['promo_flag'] * item_df['day_of_week']
        item_df['promo_weekend_interaction'] = item_df['promo_flag'] * item_df['is_weekend']
        
        future_dfs.append(item_df)
    
    # Combine all items
    future_df = pd.concat(future_dfs, ignore_index=True)
    
    # Filter to only include requested known covariates
    columns_to_keep = [config['data']['id_column'], config['data']['timestamp_column']] + \
                      [col for col in known_covariate_names if col in future_df.columns]
    
    future_df = future_df[columns_to_keep]
    
    # Convert to TimeSeriesDataFrame
    future_known = TimeSeriesDataFrame.from_data_frame(
        future_df,
        id_column=config['data']['id_column'],
        timestamp_column=config['data']['timestamp_column']
    )
    
    return future_known


def generate_forecasts(
    predictor: TimeSeriesPredictor,
    test_data: TimeSeriesDataFrame,
    config: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """Generate forecasts using trained model"""
    
    logger.info("Generating forecasts...")
    
    # Get prediction length and known covariate names
    prediction_length = config['model']['prediction_length']
    known_covariate_names = predictor.known_covariates_names
    
    if known_covariate_names:
        logger.info(f"Creating future known covariates for: {known_covariate_names}")
        
        # Check if future promo file exists
        future_promo_path = config['data'].get('future_promo_path')
        if future_promo_path and os.path.exists(future_promo_path):
            logger.info(f"Loading future promo data from: {future_promo_path}")
        else:
            logger.info("No future promo file found, using default values")
        
        # Create future known covariates
        future_known_covariates = create_future_known_covariates(
            test_data,
            prediction_length,
            known_covariate_names,
            config
        )
        
        # Generate predictions with known covariates
        predictions = predictor.predict(
            test_data,
            known_covariates=future_known_covariates
        )
    else:
        # Generate predictions without known covariates
        predictions = predictor.predict(test_data)
    
    return predictions


def save_results(
    predictions: pd.DataFrame,
    output_path: str,
    logger: logging.Logger
):
    """Save forecast results"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions.to_csv(output_path, index=True)
    logger.info(f"Predictions saved to {output_path}")


def calculate_evaluation_metrics(
    test_data: TimeSeriesDataFrame,
    predictions: pd.DataFrame,
    predictor: TimeSeriesPredictor,
    config: dict,
    logger: logging.Logger
) -> dict:
    """
    Calculate comprehensive evaluation metrics excluding OOS and promo periods
    
    Metrics:
    - Weighted MAPE
    - Bias % = sum(pred-actual)/sum(actual) * 100
    - Under Bias % = sum(pred-actual)[where pred<actual]/sum(actual) * 100
    - Over Bias % = sum(pred-actual)[where pred>actual]/sum(actual) * 100
    """
    
    logger.info("Calculating comprehensive evaluation metrics...")
    
    # Get prediction column
    if 'mean' in predictions.columns:
        pred_col = 'mean'
    elif '0.5' in predictions.columns:
        pred_col = '0.5'
    else:
        pred_col = predictions.columns[0]
    
    # Initialize lists to collect all SKU-day combinations
    all_actuals = []
    all_predictions = []
    all_weights = []
    
    # Get all unique items
    item_ids = test_data.item_ids
    
    # Get test period length
    test_period_length = config['model']['prediction_length']
    
    for item_id in item_ids:
        # Get data for this item
        item_test = test_data.loc[test_data.index.get_level_values('item_id') == item_id]
        item_pred = predictions.loc[predictions.index.get_level_values('item_id') == item_id]
        
        # Get the test period values (last prediction_length values)
        test_values = item_test[predictor.target].values[-test_period_length:]
        test_timestamps = item_test.index.get_level_values('timestamp')[-test_period_length:]
        
        # Get prediction values
        pred_values = item_pred[pred_col].values
        
        if len(test_values) != len(pred_values):
            logger.warning(f"Skipping {item_id}: length mismatch ({len(test_values)} vs {len(pred_values)})")
            continue
        
        # Filter out OOS (zero sales) and promo periods
        for i in range(len(test_values)):
            actual = test_values[i]
            pred = pred_values[i]
            
            # Skip if OOS (actual sales = 0)
            if actual == 0:
                continue
            
            # Check if it's a promo period
            # Get promo flag from the test data if available
            is_promo = False
            if 'promo_flag' in item_test.columns:
                # Find the promo flag for this timestamp
                timestamp_mask = item_test.index.get_level_values('timestamp') == test_timestamps[i]
                if timestamp_mask.any():
                    promo_flag_value = item_test.loc[timestamp_mask, 'promo_flag'].values
                    if len(promo_flag_value) > 0:
                        is_promo = promo_flag_value[0] == 1
            
            # Skip if promo period
            if is_promo:
                continue
            
            # Add to our lists for metric calculation
            all_actuals.append(actual)
            all_predictions.append(pred)
            all_weights.append(actual)  # Weight by actual sales volume
    
    # Convert to arrays
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)
    all_weights = np.array(all_weights)
    
    # Calculate metrics
    metrics = {}
    
    if len(all_actuals) > 0:
        # Weighted MAPE
        ape = np.abs((all_predictions - all_actuals) / all_actuals)
        weighted_mape = np.average(ape, weights=all_weights) * 100
        metrics['weighted_mape'] = weighted_mape
        
        # Overall Bias %
        bias_pct = (np.sum(all_predictions - all_actuals) / np.sum(all_actuals)) * 100
        metrics['bias_pct'] = bias_pct
        
        # Under Bias % (where predictions < actuals)
        under_mask = all_predictions < all_actuals
        if under_mask.any():
            under_bias_pct = (np.sum(all_predictions[under_mask] - all_actuals[under_mask]) / np.sum(all_actuals)) * 100
        else:
            under_bias_pct = 0.0
        metrics['under_bias_pct'] = under_bias_pct
        
        # Over Bias % (where predictions > actuals)
        over_mask = all_predictions > all_actuals
        if over_mask.any():
            over_bias_pct = (np.sum(all_predictions[over_mask] - all_actuals[over_mask]) / np.sum(all_actuals)) * 100
        else:
            over_bias_pct = 0.0
        metrics['over_bias_pct'] = over_bias_pct
        
        # Additional useful metrics
        metrics['rmse'] = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
        metrics['mae'] = np.mean(np.abs(all_predictions - all_actuals))
        metrics['n_samples'] = len(all_actuals)
        metrics['n_skus'] = len(item_ids)
        
        # Log the metrics
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS (excluding OOS and promo periods)")
        logger.info("=" * 60)
        logger.info(f"Number of SKUs evaluated: {metrics['n_skus']}")
        logger.info(f"Number of valid samples: {metrics['n_samples']}")
        logger.info(f"Weighted MAPE: {metrics['weighted_mape']:.2f}%")
        logger.info(f"Overall Bias %: {metrics['bias_pct']:.2f}%")
        logger.info(f"Under Bias % (pred < actual): {metrics['under_bias_pct']:.2f}%")
        logger.info(f"Over Bias % (pred > actual): {metrics['over_bias_pct']:.2f}%")
        logger.info(f"RMSE: {metrics['rmse']:.2f}")
        logger.info(f"MAE: {metrics['mae']:.2f}")
        logger.info("=" * 60)
        
    else:
        logger.warning("No valid samples found after filtering OOS and promo periods!")
        metrics = {
            'weighted_mape': np.nan,
            'bias_pct': np.nan,
            'under_bias_pct': np.nan,
            'over_bias_pct': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'n_samples': 0,
            'n_skus': 0
        }
    
    return metrics


def generate_evaluation_plots(
    predictor: TimeSeriesPredictor,
    test_data: TimeSeriesDataFrame,
    predictions: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
    config: dict,
    num_items_to_plot: int = 5
):
    """Generate and save evaluation plots using AutoGluon's built-in plotting functions"""
    
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Generating evaluation plots in {plots_dir}")
    
    try:
        # 1. Plot predictions for sample items
        logger.info(f"Plotting predictions for {num_items_to_plot} sample items...")
        
        # Get unique item IDs
        item_ids = test_data.item_ids[:num_items_to_plot]
        
        for i, item_id in enumerate(item_ids):
            plt.figure(figsize=(12, 6))
            
            # Filter data for this item
            item_test_data = test_data.loc[test_data.index.get_level_values('item_id') == item_id]
            item_predictions = predictions.loc[predictions.index.get_level_values('item_id') == item_id]
            
            # Use AutoGluon's plot method
            predictor.plot(
                item_test_data,
                predictions=item_predictions,
                quantile_levels=[0.1, 0.9]
            )
            
            plt.title(f'Forecast for {item_id}')
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f'forecast_{item_id}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved forecast plot for {item_id}")
        
        # 2. Plot model performance comparison (if ensemble)
        if hasattr(predictor, 'leaderboard'):
            logger.info("Plotting model performance comparison...")
            
            leaderboard = predictor.leaderboard()
            
            # Create bar plot of model scores
            plt.figure(figsize=(10, 6))
            models = leaderboard['model'].tolist()
            scores = leaderboard['score_val'].tolist()
            
            plt.barh(models, scores)
            plt.xlabel('Validation Score (negative MAE)')
            plt.ylabel('Model')
            plt.title('Model Performance Comparison')
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Saved model comparison plot")
        
        # 3. Plot residuals analysis
        logger.info("Generating residuals analysis...")
        
        # Get predictions at median quantile
        if 'mean' in predictions.columns:
            pred_col = 'mean'
        elif '0.5' in predictions.columns:
            pred_col = '0.5'
        else:
            pred_col = predictions.columns[0]
        
        # Calculate residuals for the first few items
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, item_id in enumerate(item_ids[:4]):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            try:
                # Get data for this item
                item_test = test_data.loc[test_data.index.get_level_values('item_id') == item_id]
                item_pred = predictions.loc[predictions.index.get_level_values('item_id') == item_id]
                
                # Use the same approach as in actual vs forecast comparison
                test_timestamps = item_test.index.get_level_values('timestamp')
                test_values = item_test[predictor.target].values
                
                # Get the test period (last prediction_length values)
                test_period_length = config['model']['prediction_length']
                actual_test_values = test_values[-test_period_length:]
                
                # Get prediction values
                pred_values = item_pred[pred_col].values
                
                # Calculate residuals if lengths match
                if len(actual_test_values) == len(pred_values):
                    residuals = actual_test_values - pred_values
                    
                    # Plot residuals over time
                    ax.plot(range(len(residuals)), residuals, 'o-', alpha=0.7, markersize=4)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                    ax.set_title(f'Residuals: {item_id}')
                    ax.set_xlabel('Time Index (Test Period)')
                    ax.set_ylabel('Residual (Actual - Predicted)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add some statistics
                    rmse = np.sqrt(np.mean(residuals**2))
                    ax.text(0.02, 0.98, f'RMSE: {rmse:.2f}', transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0.5, 0.5, f'Data mismatch\n{len(actual_test_values)} vs {len(pred_values)}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Residuals: {item_id} (No Data)')
                    
            except Exception as e:
                logger.warning(f"Could not generate residuals for {item_id}: {str(e)}")
                ax.text(0.5, 0.5, f'Error generating\nresiduals for {item_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Residuals: {item_id} (Error)')
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'residuals_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved residuals analysis plot")
        
        # 4. Feature importance (if available)
        if hasattr(predictor, 'feature_importance'):
            logger.info("Plotting feature importance...")
            
            importance = predictor.feature_importance()
            
            if importance is not None and len(importance) > 0:
                plt.figure(figsize=(10, 8))
                
                # Sort by importance
                importance_sorted = importance.sort_values('importance', ascending=True).tail(20)
                
                plt.barh(importance_sorted.index, importance_sorted['importance'])
                plt.xlabel('Importance Score')
                plt.ylabel('Feature')
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                
                plot_path = os.path.join(plots_dir, 'feature_importance.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info("Saved feature importance plot")
        
        # 5. Actual vs Forecast Comparison Plot
        logger.info("Generating actual vs forecast comparison plots...")
        
        # When evaluating on test data, we want to compare predictions with the actual test values
        # The predictions should be for the last `prediction_length` timestamps in test_data
        
        # Create subplot grid for multiple items
        n_items = min(4, len(item_ids))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, item_id in enumerate(item_ids[:n_items]):
            ax = axes[idx]
            
            # Get data for this item
            item_test = test_data.loc[test_data.index.get_level_values('item_id') == item_id]
            item_pred = predictions.loc[predictions.index.get_level_values('item_id') == item_id]
            
            # Get the test data timestamps and values
            test_timestamps = item_test.index.get_level_values('timestamp')
            test_values = item_test[predictor.target].values
            
            # Get prediction timestamps
            pred_timestamps = item_pred.index.get_level_values('timestamp')
            
            # The test period is the last `prediction_length` values
            # This is what we held out for evaluation
            test_period_length = len(pred_timestamps)
            
            # Get the cutoff point (where training ended and test began)
            cutoff_timestamp = test_timestamps[-test_period_length - 1] if len(test_timestamps) > test_period_length else test_timestamps[0]
            
            # Plot historical data (before cutoff)
            hist_mask = test_timestamps <= cutoff_timestamp
            if hist_mask.any():
                # Show last 60 days of training data
                hist_timestamps = test_timestamps[hist_mask][-60:]
                hist_values = test_values[hist_mask][-60:]
                ax.plot(hist_timestamps, hist_values, 'b-', label='Training Data', alpha=0.7)
            
            # For proper comparison, we need to align timestamps
            # The predictions are made using test_data, so they should correspond to the test period
            # But AutoGluon might predict future periods from the end of test_data
            
            # Find the actual test period by looking at the last `prediction_length` timestamps
            test_period_length = config['model']['prediction_length']
            
            # Get the actual test timestamps and values (the holdout period)
            actual_test_timestamps = test_timestamps[-test_period_length:]
            actual_test_values = test_values[-test_period_length:]
            
            # Plot actual test values
            ax.plot(actual_test_timestamps, actual_test_values, 'g-', label='Actual (Test)', linewidth=2)
            
            # For predictions, we need to map them to the same time period as the actual test data
            # Since AutoGluon predicts from the end of the input data, we need to align them properly
            if 'mean' in item_pred.columns:
                pred_values = item_pred['mean'].values
            else:
                pred_values = item_pred['0.5'].values
            
            # Use the actual test timestamps for plotting predictions to ensure alignment
            if len(pred_values) == len(actual_test_timestamps):
                ax.plot(actual_test_timestamps, pred_values, 'r--', label='Forecast', linewidth=2)
                
                # Add prediction intervals using the same timestamps
                if '0.1' in item_pred.columns and '0.9' in item_pred.columns:
                    ax.fill_between(actual_test_timestamps,
                                   item_pred['0.1'].values,
                                   item_pred['0.9'].values,
                                   alpha=0.3, color='red', label='80% PI')
            else:
                # If lengths don't match, truncate or pad as needed
                min_length = min(len(pred_values), len(actual_test_timestamps))
                ax.plot(actual_test_timestamps[:min_length], pred_values[:min_length], 'r--', label='Forecast', linewidth=2)
                logger.warning(f"Length mismatch for {item_id}: {len(pred_values)} predictions vs {len(actual_test_timestamps)} test timestamps")
            
            # Add vertical line at cutoff
            ax.axvline(x=cutoff_timestamp, color='black', linestyle=':', alpha=0.5, label='Train/Test Split')
            
            # Calculate evaluation metrics for this item
            if len(actual_test_values) == len(pred_values):
                mae = np.mean(np.abs(actual_test_values - pred_values))
                # Avoid division by zero in MAPE calculation
                non_zero_mask = actual_test_values != 0
                if non_zero_mask.any():
                    mape = np.mean(np.abs((actual_test_values[non_zero_mask] - pred_values[non_zero_mask]) / actual_test_values[non_zero_mask])) * 100
                    ax.set_title(f'{item_id} - MAE: {mae:.2f}, MAPE: {mape:.1f}%')
                else:
                    ax.set_title(f'{item_id} - MAE: {mae:.2f}')
            else:
                ax.set_title(f'{item_id} - Actual vs Forecast')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(n_items, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'actual_vs_forecast_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved actual vs forecast comparison plot")
        
        # 6. Future Predictions Plot (beyond available data)
        logger.info("Generating future predictions plot...")
        
        # Generate future predictions
        try:
            # Get the last available data point for prediction
            last_train_data = test_data  # This contains full history
            
            # Create future known covariates if needed
            if predictor.known_covariates_names:
                logger.info("Creating future known covariates for future predictions...")
                future_known_covariates = create_future_known_covariates(
                    last_train_data,
                    predictor.prediction_length,
                    predictor.known_covariates_names,
                    config
                )
                # Make future predictions with known covariates
                future_predictions = predictor.predict(last_train_data, known_covariates=future_known_covariates)
            else:
                # Make future predictions without known covariates
                future_predictions = predictor.predict(last_train_data)
            
            # Plot future predictions for sample items
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for idx, item_id in enumerate(item_ids[:4]):
                ax = axes[idx]
                
                # Get historical data for this item
                item_hist = test_data.loc[test_data.index.get_level_values('item_id') == item_id]
                item_future = future_predictions.loc[future_predictions.index.get_level_values('item_id') == item_id]
                
                # Plot last 90 days of historical data
                hist_timestamps = item_hist.index.get_level_values('timestamp')[-90:]
                hist_values = item_hist[predictor.target].values[-90:]
                
                ax.plot(hist_timestamps, hist_values, 'b-', label='Historical', linewidth=2)
                
                # Plot future predictions
                future_timestamps = item_future.index.get_level_values('timestamp')
                
                if 'mean' in item_future.columns:
                    future_mean = item_future['mean'].values
                else:
                    future_mean = item_future['0.5'].values
                
                ax.plot(future_timestamps, future_mean, 'g--', label='Future Forecast', linewidth=2)
                
                # Add prediction intervals
                if '0.1' in item_future.columns and '0.9' in item_future.columns:
                    ax.fill_between(future_timestamps,
                                   item_future['0.1'].values,
                                   item_future['0.9'].values,
                                   alpha=0.3, color='green', label='80% PI')
                
                # Add vertical line at prediction start
                ax.axvline(x=future_timestamps[0], color='red', linestyle=':', alpha=0.7, label='Forecast Start')
                
                ax.set_title(f'Future Predictions: {item_id}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for idx in range(min(4, len(item_ids)), 4):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, 'future_predictions.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Saved future predictions plot")
            
        except Exception as e:
            logger.warning(f"Could not generate future predictions plot: {str(e)}")
        
        logger.info("All evaluation plots generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        logger.error("Continuing without plots...")


def main():
    parser = argparse.ArgumentParser(description='AutoGluon Time Series Forecasting')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'train_predict'], 
                       default='train_predict', help='Execution mode')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'], config['logging']['level'])
    logger.info(f"Starting forecasting pipeline with config: {args.config}")
    
    # Initialize preprocessor and feature engineer
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    
    # Load and preprocess data
    logger.info("Loading data...")
    raw_data = pd.read_csv(config['data']['train_path'])
    
    logger.info("Preprocessing data...")
    processed_data = preprocessor.preprocess(raw_data)
    
    logger.info("Engineering features...")
    featured_data = feature_engineer.create_features(processed_data)
    
    # Extract static features
    static_feature_cols = config['features'].get('static_features', [])
    static_features_df = extract_static_features(
        featured_data,
        config['data']['id_column'],
        static_feature_cols
    )
    
    if static_features_df is not None:
        logger.info(f"Extracted static features: {static_feature_cols}")
        logger.info(f"Static features shape: {static_features_df.shape}")
    
    # Convert to TimeSeriesDataFrame
    ts_data = create_timeseries_dataframe(
        featured_data,
        target_col=config['data']['target_column'],
        timestamp_col=config['data']['timestamp_column'],
        id_col=config['data']['id_column'],
        static_features_df=static_features_df
    )
    
    # Split data into train and test sets
    prediction_length = config['model']['prediction_length']
    test_size = config['data'].get('test_size', prediction_length)
    
    if args.mode in ['train', 'train_predict']:
        # Use AutoGluon's train_test_split
        logger.info(f"Splitting data with prediction_length={test_size}")
        train_data, test_data = ts_data.train_test_split(prediction_length=test_size)
        
        logger.info(f"Train data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")
        
        # Train model
        logger.info("Training model...")
        predictor = train_forecast_model(train_data, config, logger)
        
        # Log model performance
        logger.info("Model leaderboard:")
        logger.info(predictor.leaderboard())
        
        if args.mode == 'train':
            logger.info("Training completed. Model saved.")
            return
    
    if args.mode in ['predict', 'train_predict']:
        # Load predictor if only predicting
        if args.mode == 'predict':
            predictor = TimeSeriesPredictor.load(config['model']['save_path'])
            
            # For predict-only mode, use the last test_size periods as test data
            logger.info(f"Using last {test_size} periods as test data for prediction")
            train_data, test_data = ts_data.train_test_split(prediction_length=test_size)
            
            # Ensure we have static features for prediction if they were used in training
            if static_features_df is None and hasattr(predictor, '_learner') and \
               hasattr(predictor._learner, 'static_features') and \
               predictor._learner.static_features is not None:
                logger.warning("Model was trained with static features but none provided for prediction")
        
        # Generate forecasts using test_data
        predictions = generate_forecasts(predictor, test_data, config, logger)
        
        # Save results
        save_results(
            predictions,
            config['output']['predictions_path'],
            logger
        )
        
        # Evaluate predictions if we have ground truth
        if args.mode == 'train_predict':
            logger.info("Evaluating predictions...")
            # AutoGluon can evaluate predictions against the test data
            scores = predictor.evaluate(test_data)
            logger.info(f"Evaluation scores: {scores}")
            
            # Calculate comprehensive evaluation metrics
            evaluation_metrics = calculate_evaluation_metrics(
                test_data,
                predictions,
                predictor,
                config,
                logger
            )
            
            # Save metrics to file
            metrics_path = config['output'].get('metrics_path', 'output/metrics.json')
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(evaluation_metrics, f, indent=2, default=str)
            logger.info(f"Evaluation metrics saved to {metrics_path}")
            
            # Also save a readable report
            report_path = metrics_path.replace('.json', '_report.txt')
            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("FORECAST EVALUATION REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {config['model'].get('models', 'Unknown')}\n")
                f.write(f"Prediction Length: {config['model']['prediction_length']} days\n")
                f.write("\n")
                f.write("EVALUATION METRICS (excluding OOS and promo periods)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Number of SKUs evaluated: {evaluation_metrics.get('n_skus', 'N/A')}\n")
                f.write(f"Number of valid samples: {evaluation_metrics.get('n_samples', 'N/A')}\n")
                f.write(f"Weighted MAPE: {evaluation_metrics.get('weighted_mape', np.nan):.2f}%\n")
                f.write(f"Overall Bias %: {evaluation_metrics.get('bias_pct', np.nan):.2f}%\n")
                f.write(f"Under Bias % (pred < actual): {evaluation_metrics.get('under_bias_pct', np.nan):.2f}%\n")
                f.write(f"Over Bias % (pred > actual): {evaluation_metrics.get('over_bias_pct', np.nan):.2f}%\n")
                f.write(f"RMSE: {evaluation_metrics.get('rmse', np.nan):.2f}\n")
                f.write(f"MAE: {evaluation_metrics.get('mae', np.nan):.2f}\n")
                f.write("=" * 60 + "\n")
            logger.info(f"Evaluation report saved to {report_path}")
            
            # Generate evaluation plots
            plots_dir = config['output']['plots_dir']
            generate_evaluation_plots(
                predictor,
                test_data,
                predictions,
                plots_dir,
                logger,
                config,
                num_items_to_plot=min(5, len(test_data.item_ids))
            )
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()