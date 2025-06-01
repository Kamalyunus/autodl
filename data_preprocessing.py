"""
Data preprocessing utilities for time series forecasting
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df = df.copy()
        
        # Convert timestamp column to datetime
        df = self._convert_timestamp(df)
        
        # Sort by ID and timestamp
        df = self._sort_data(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Filter SKUs with sufficient history
        df = self._filter_by_history_length(df)
        
        # Remove zero sales SKUs if configured
        if self.config['preprocessing'].get('remove_zero_sales_skus', False):
            df = self._remove_zero_sales_skus(df)
        
        # Ensure complete time series
        df = self._ensure_complete_timeseries(df)
        
        return df
    
    def _convert_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to datetime"""
        timestamp_col = self.config['data']['timestamp_column']
        datetime_format = self.config['data'].get('datetime_format', None)
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=datetime_format)
        self.logger.info(f"Converted {timestamp_col} to datetime")
        
        return df
    
    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data by ID and timestamp"""
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        
        df = df.sort_values([id_col, timestamp_col])
        self.logger.info("Sorted data by ID and timestamp")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with advanced methods"""
        missing_config = self.config['preprocessing']['handle_missing']
        method = missing_config['method']
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        
        # Track missing values before processing
        initial_missing = df[target_col].isna().sum()
        
        if method == 'interpolate':
            interpolation_method = missing_config.get('interpolation_method', 'linear')
            limit = missing_config.get('limit', 7)
            
            def advanced_interpolation(group):
                # Sort by timestamp to ensure proper interpolation
                group = group.sort_values(timestamp_col)
                
                # Calculate gap sizes
                missing_mask = group[target_col].isna()
                gap_groups = (missing_mask != missing_mask.shift()).cumsum()
                gap_sizes = missing_mask.groupby(gap_groups).transform('sum')
                
                # For small gaps (<=limit), use interpolation
                small_gaps = missing_mask & (gap_sizes <= limit)
                if small_gaps.any():
                    # Try seasonal interpolation for better results
                    if len(group) > 14:  # Need sufficient history
                        # Create a simple seasonal pattern (weekly)
                        group['day_of_week'] = pd.to_datetime(group[timestamp_col]).dt.dayofweek
                        seasonal_means = group.groupby('day_of_week')[target_col].transform('mean')
                        
                        # Interpolate and blend with seasonal pattern
                        interpolated = group[target_col].interpolate(method=interpolation_method, limit=limit)
                        
                        # Blend interpolation with seasonal pattern for missing values
                        blend_factor = 0.7  # 70% interpolation, 30% seasonal
                        group.loc[small_gaps, target_col] = (
                            blend_factor * interpolated[small_gaps] + 
                            (1 - blend_factor) * seasonal_means[small_gaps]
                        )
                    else:
                        # Not enough history, use standard interpolation
                        group[target_col] = group[target_col].interpolate(
                            method=interpolation_method, limit=limit
                        )
                
                # For large gaps, use forward/backward fill with decay
                large_gaps = missing_mask & (gap_sizes > limit)
                if large_gaps.any():
                    # Forward fill with exponential decay
                    filled_forward = group[target_col].fillna(method='ffill')
                    days_since_last = group.groupby((~missing_mask).cumsum())[target_col].cumcount()
                    decay_factor = 0.95 ** days_since_last  # 5% daily decay
                    
                    group.loc[large_gaps, target_col] = filled_forward[large_gaps] * decay_factor[large_gaps]
                    
                    # If still missing (at the beginning), use backward fill
                    still_missing = group[target_col].isna()
                    if still_missing.any():
                        group.loc[still_missing, target_col] = group[target_col].fillna(method='bfill')
                
                # Clean up temporary column
                if 'day_of_week' in group.columns:
                    group = group.drop('day_of_week', axis=1)
                
                return group
            
            df = df.groupby(id_col).apply(advanced_interpolation).reset_index(drop=True)
            
        elif method == 'forward_fill':
            limit = missing_config.get('limit', None)
            df[target_col] = df.groupby(id_col)[target_col].transform(
                lambda x: x.fillna(method='ffill', limit=limit).fillna(method='bfill')
            )
            
        elif method == 'zero_fill':
            # For intermittent demand, zero might be appropriate
            df[target_col] = df[target_col].fillna(0)
        
        # Handle any remaining missing values
        remaining_missing = df[target_col].isna().sum()
        if remaining_missing > 0:
            # Use median of the SKU as last resort
            df[target_col] = df.groupby(id_col)[target_col].transform(
                lambda x: x.fillna(x.median()).fillna(0)
            )
        
        # Log missing value stats
        final_missing = df[target_col].isna().sum()
        self.logger.info(f"Handled missing values using {method}. "
                        f"Initial: {initial_missing}, Final: {final_missing}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers with SKU-specific and seasonal awareness"""
        outlier_config = self.config['preprocessing']['handle_outliers']
        method = outlier_config['method']
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        
        if method == 'clip':
            # Track outliers for logging
            total_outliers = 0
            
            def clip_outliers_advanced(group):
                nonlocal total_outliers
                
                # Calculate statistics for this SKU
                sku_data = group[target_col]
                sku_mean = sku_data.mean()
                sku_std = sku_data.std()
                
                # Use IQR method for robust outlier detection
                q1 = sku_data.quantile(0.25)
                q3 = sku_data.quantile(0.75)
                iqr = q3 - q1
                
                # Adaptive bounds based on SKU characteristics
                cv = sku_std / (sku_mean + 1e-6)  # Coefficient of variation
                
                if cv > 1.5:  # High variance SKU (e.g., seasonal, intermittent)
                    # Use wider bounds
                    iqr_multiplier = 2.5
                elif cv > 0.8:  # Medium variance
                    iqr_multiplier = 2.0
                else:  # Low variance (stable SKU)
                    iqr_multiplier = 1.5
                
                lower_bound = max(0, q1 - iqr_multiplier * iqr)
                upper_bound = q3 + iqr_multiplier * iqr
                
                # Account for promotional periods
                if 'promo_flag' in group.columns:
                    # Calculate separate bounds for promo periods
                    promo_mask = group['promo_flag'] == 1
                    if promo_mask.sum() > 5:  # Need sufficient promo data
                        promo_data = sku_data[promo_mask]
                        promo_q3 = promo_data.quantile(0.75)
                        promo_iqr = promo_data.quantile(0.75) - promo_data.quantile(0.25)
                        promo_upper = promo_q3 + 1.5 * promo_iqr
                        
                        # Apply different bounds for promo vs non-promo
                        group.loc[~promo_mask, target_col] = group.loc[~promo_mask, target_col].clip(
                            lower=lower_bound, upper=upper_bound
                        )
                        group.loc[promo_mask, target_col] = group.loc[promo_mask, target_col].clip(
                            lower=lower_bound, upper=promo_upper
                        )
                    else:
                        # Not enough promo data, use regular bounds
                        group[target_col] = group[target_col].clip(lower=lower_bound, upper=upper_bound)
                else:
                    # No promo flag, use regular bounds
                    group[target_col] = group[target_col].clip(lower=lower_bound, upper=upper_bound)
                
                # Count outliers
                outliers_clipped = ((group[target_col] == lower_bound) | 
                                  (group[target_col] == upper_bound)).sum()
                total_outliers += outliers_clipped
                
                return group
            
            df = df.groupby(id_col).apply(clip_outliers_advanced).reset_index(drop=True)
            self.logger.info(f"Handled {total_outliers} outliers using advanced SKU-specific {method}")
            
        elif method == 'remove':
            # Similar logic but remove instead of clip
            lower_q = outlier_config.get('lower_quantile', 0.01)
            upper_q = outlier_config.get('upper_quantile', 0.99)
            
            def remove_outliers(group):
                lower = group[target_col].quantile(lower_q)
                upper = group[target_col].quantile(upper_q)
                mask = (group[target_col] >= lower) & (group[target_col] <= upper)
                return group[mask]
            
            df = df.groupby(id_col).apply(remove_outliers).reset_index(drop=True)
        
        return df
    
    def _filter_by_history_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter SKUs with insufficient history"""
        min_history = self.config['preprocessing']['min_history_length']
        id_col = self.config['data']['id_column']
        
        # Count observations per SKU
        sku_counts = df.groupby(id_col).size()
        valid_skus = sku_counts[sku_counts >= min_history].index
        
        # Filter dataframe
        df_filtered = df[df[id_col].isin(valid_skus)]
        
        removed_skus = len(sku_counts) - len(valid_skus)
        self.logger.info(f"Removed {removed_skus} SKUs with less than {min_history} observations")
        
        return df_filtered
    
    def _remove_zero_sales_skus(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove SKUs with all zero sales"""
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        
        # Find SKUs with non-zero sales
        non_zero_skus = df[df[target_col] > 0][id_col].unique()
        
        # Filter dataframe
        df_filtered = df[df[id_col].isin(non_zero_skus)]
        
        removed_skus = df[id_col].nunique() - len(non_zero_skus)
        self.logger.info(f"Removed {removed_skus} SKUs with all zero sales")
        
        return df_filtered
    
    def _ensure_complete_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure complete time series for each SKU"""
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        
        # Create complete date range
        date_range = pd.date_range(
            start=df[timestamp_col].min(),
            end=df[timestamp_col].max(),
            freq='D'
        )
        
        # Create complete index for each SKU
        all_skus = df[id_col].unique()
        complete_index = pd.MultiIndex.from_product(
            [all_skus, date_range],
            names=[id_col, timestamp_col]
        )
        
        # Reindex to ensure completeness
        df_complete = df.set_index([id_col, timestamp_col]).reindex(complete_index).reset_index()
        
        # Forward fill static features
        static_features = self.config['features']['static_features']
        for feature in static_features:
            if feature in df_complete.columns:
                df_complete[feature] = df_complete.groupby(id_col)[feature].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        self.logger.info("Ensured complete time series for all SKUs")
        return df_complete