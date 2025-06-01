"""
Feature engineering utilities for time series forecasting
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import holidays


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        df = df.copy()
        
        # Create calendar features
        df = self._create_calendar_features(df)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Create rolling window features
        df = self._create_rolling_features(df)
        
        # Create exponentially weighted features
        df = self._create_ewm_features(df)
        
        # Create holiday features
        df = self._create_holiday_features(df)
        
        # Create cross-SKU features
        df = self._create_cross_sku_features(df)
        
        # Create advanced temporal features
        df = self._create_advanced_temporal_features(df)
        
        # Create enhanced promotional features
        df = self._create_enhanced_promo_features(df)
        
        # Create conservative counter-features to balance over-optimistic bias
        df = self._create_conservative_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Sort again to ensure proper order
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        df = df.sort_values([id_col, timestamp_col])
        
        # Final data validation and cleanup
        df = self._validate_and_clean_features(df)
        
        return df
    
    def _validate_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean features to prevent infinite/NaN values"""
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with NaN first
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Handle remaining NaN values
        for col in numeric_cols:
            if df[col].isna().any():
                if col.endswith('_flag') or col.startswith('is_'):
                    # Binary features should be 0 or 1
                    df[col] = df[col].fillna(0).astype(int)
                elif 'rank' in col or 'share' in col or 'index' in col:
                    # Proportion-like features
                    df[col] = df[col].fillna(0).clip(0, 1)
                elif 'growth' in col or 'effectiveness' in col:
                    # Growth/change features
                    df[col] = df[col].fillna(0).clip(-10, 10)
                else:
                    # General numeric features - use median or 0
                    median_val = df[col].median()
                    fill_val = median_val if not np.isnan(median_val) else 0
                    df[col] = df[col].fillna(fill_val)
        
        # Final check: clip extremely large values
        for col in numeric_cols:
            if col not in [self.config['data']['target_column'], 
                          self.config['data']['id_column'], 
                          self.config['data']['timestamp_column']]:
                # Clip to reasonable float32 range
                df[col] = np.clip(df[col], -1e6, 1e6)
        
        # Log any remaining issues
        infinite_count = np.isinf(df[numeric_cols]).sum().sum()
        nan_count = df[numeric_cols].isna().sum().sum()
        
        if infinite_count > 0 or nan_count > 0:
            self.logger.warning(f"After cleanup: {infinite_count} infinite values, {nan_count} NaN values remain")
        else:
            self.logger.info("Feature validation passed - no infinite or NaN values")
        
        return df
    
    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features"""
        timestamp_col = self.config['data']['timestamp_column']
        
        # Basic calendar features
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
        df['year'] = df[timestamp_col].dt.year
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month start/end flags
        df['is_month_start'] = (df[timestamp_col].dt.day <= 5).astype(int)
        df['is_month_end'] = (df[timestamp_col].dt.day >= 25).astype(int)
        
        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        self.logger.info("Created calendar features")
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features based on configuration"""
        lag_config = self.config['features']['lag_features']
        id_col = self.config['data']['id_column']
        
        for column, lags in lag_config.items():
            if column in df.columns:
                for lag in lags:
                    feature_name = f"{column}_lag_{lag}"
                    df[feature_name] = df.groupby(id_col)[column].shift(lag)
                    self.logger.info(f"Created lag feature: {feature_name}")
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        rolling_config = self.config['features']['rolling_features']
        id_col = self.config['data']['id_column']
        
        for column, config in rolling_config.items():
            if column in df.columns:
                windows = config['windows']
                functions = config['functions']
                
                for window in windows:
                    for func in functions:
                        feature_name = f"{column}_rolling_{func}_{window}"
                        
                        if func == 'mean':
                            df[feature_name] = df.groupby(id_col)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).mean()
                            )
                        elif func == 'std':
                            df[feature_name] = df.groupby(id_col)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=2).std()
                            )
                        elif func == 'min':
                            df[feature_name] = df.groupby(id_col)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).min()
                            )
                        elif func == 'max':
                            df[feature_name] = df.groupby(id_col)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).max()
                            )
                        
                        self.logger.info(f"Created rolling feature: {feature_name}")
        
        return df
    
    def _create_ewm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create exponentially weighted moving average features"""
        ewm_config = self.config['features']['ewm_features']
        id_col = self.config['data']['id_column']
        
        for column, config in ewm_config.items():
            if column in df.columns:
                spans = config['spans']
                
                for span in spans:
                    feature_name = f"{column}_ewm_{span}"
                    df[feature_name] = df.groupby(id_col)[column].transform(
                        lambda x: x.ewm(span=span, adjust=False).mean()
                    )
                    self.logger.info(f"Created EWM feature: {feature_name}")
        
        return df
    
    def _create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday features"""
        timestamp_col = self.config['data']['timestamp_column']
        
        # Create US holidays (can be configured for other countries)
        us_holidays = holidays.US()
        
        # Basic holiday flag
        df['is_holiday'] = df[timestamp_col].apply(lambda x: x in us_holidays).astype(int)
        
        # Days to/from nearest holiday
        dates = df[timestamp_col].dt.date.unique()
        holiday_dates = [d for d in dates if d in us_holidays]
        
        if holiday_dates:
            df['days_to_holiday'] = df[timestamp_col].apply(
                lambda x: min([abs((h - x.date()).days) for h in holiday_dates])
            )
            
            # Holiday proximity features
            df['is_pre_holiday'] = (df['days_to_holiday'] == 1).astype(int)
            df['is_post_holiday'] = df['is_pre_holiday'].shift(-1).fillna(0)
        
        self.logger.info("Created holiday features")
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between calendar and other features"""
        
        # Weekend × Month interaction
        df['weekend_month_interaction'] = df['is_weekend'] * df['month']
        
        # Promo × Day of week interaction (if promo exists)
        if 'promo_flag' in df.columns:
            df['promo_dow_interaction'] = df['promo_flag'] * df['day_of_week']
            df['promo_weekend_interaction'] = df['promo_flag'] * df['is_weekend']
        
        # Holiday × Weekend interaction
        if 'is_holiday' in df.columns:
            df['holiday_weekend_interaction'] = df['is_holiday'] * df['is_weekend']
        
        self.logger.info("Created interaction features")
        return df
    
    def _create_cross_sku_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-SKU features for category trends and brand indices"""
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        
        # Category-level aggregations
        if 'category' in df.columns:
            # Category mean sales (excluding current SKU)
            category_totals = df.groupby(['category', timestamp_col])[target_col].transform('sum')
            sku_category_sales = df.groupby([id_col, timestamp_col])[target_col].transform('first')
            df['category_total_sales'] = np.maximum(0, category_totals - sku_category_sales)
            
            # Category growth rate (with bounds checking)
            category_growth = df.groupby('category')['category_total_sales'].pct_change(7)
            df['category_growth_rate'] = np.clip(category_growth.fillna(0), -10, 10)  # Clip extreme growth rates
            
            # Category volatility (7-day rolling std)
            category_vol = df.groupby('category')['category_total_sales'].transform(
                lambda x: x.rolling(7, min_periods=1).std().fillna(0)
            )
            df['category_volatility'] = np.clip(category_vol, 0, 1e6)  # Prevent extreme volatility values
        
        # Brand-level features
        if 'brand' in df.columns:
            # Brand market share within category
            if 'category' in df.columns:
                brand_category_sales = df.groupby(['brand', 'category', timestamp_col])[target_col].transform('sum')
                category_total_sales = df.groupby(['category', timestamp_col])[target_col].transform('sum')
                market_share = brand_category_sales / np.maximum(category_total_sales, 1e-6)
                df['brand_market_share'] = np.clip(market_share.fillna(0), 0, 1)  # Market share should be 0-1
            
            # Brand popularity index (how many SKUs are selling)
            brand_active_skus = df[df[target_col] > 0].groupby(['brand', timestamp_col])[id_col].transform('nunique')
            brand_total_skus = df.groupby(['brand', timestamp_col])[id_col].transform('nunique')
            popularity = brand_active_skus / np.maximum(brand_total_skus, 1)
            df['brand_popularity_index'] = np.clip(popularity.fillna(0), 0, 1)  # Should be 0-1
        
        # Cannibalization effects (similar products in same category)
        if 'category' in df.columns and 'subcategory' in df.columns:
            # Subcategory competition intensity
            subcategory_skus = df.groupby(['subcategory', timestamp_col])[id_col].transform('nunique')
            df['subcategory_competition'] = np.clip(subcategory_skus, 1, 1000)  # Reasonable bounds
            
            # Similar product performance (subcategory average excluding self)
            subcategory_totals = df.groupby(['subcategory', timestamp_col])[target_col].transform('sum')
            similar_avg = (subcategory_totals - sku_category_sales) / np.maximum(subcategory_skus - 1, 1)
            df['similar_products_avg'] = np.clip(similar_avg.fillna(0), 0, 1e6)  # Prevent extreme values
        
        # Price positioning features
        if 'price' in df.columns and 'category' in df.columns:
            # Price rank within category
            df['price_rank_in_category'] = df.groupby(['category', timestamp_col])['price'].rank(pct=True)
            
            # Price relative to category median
            category_median_price = df.groupby(['category', timestamp_col])['price'].transform('median')
            price_ratio = df['price'] / np.maximum(category_median_price, 1e-6)
            df['price_vs_category_median'] = np.clip(price_ratio.fillna(1), 0.01, 100)  # Reasonable price ratio bounds
        
        self.logger.info("Created cross-SKU features")
        return df
    
    def _create_advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced temporal features including Fourier and trend analysis"""
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        
        # Fourier features for multiple seasonality periods
        df['day_of_year_sin_weekly'] = np.sin(2 * np.pi * df['day_of_year'] / 7)
        df['day_of_year_cos_weekly'] = np.cos(2 * np.pi * df['day_of_year'] / 7)
        df['day_of_year_sin_monthly'] = np.sin(2 * np.pi * df['day_of_year'] / 30.44)
        df['day_of_year_cos_monthly'] = np.cos(2 * np.pi * df['day_of_year'] / 30.44)
        df['day_of_year_sin_quarterly'] = np.sin(2 * np.pi * df['day_of_year'] / 91.31)
        df['day_of_year_cos_quarterly'] = np.cos(2 * np.pi * df['day_of_year'] / 91.31)
        
        # Trend strength and velocity features with multiple time windows
        def calculate_trend_features(group):
            sales = group[target_col].values
            min_length = max(28, len(sales) // 4)  # Need sufficient history
            
            if len(sales) < min_length:
                # Initialize all features to 0 for insufficient data
                group['trend_strength'] = 0
                for window in [7, 14, 21, 28]:
                    group[f'sales_velocity_{window}d'] = 0
                    group[f'sales_acceleration_{window}d'] = 0
                return group
            
            # Linear trend strength (R² of linear regression)
            x = np.arange(len(sales))
            if np.std(sales) > 1e-6:  # Check for sufficient variance
                try:
                    correlation = np.corrcoef(x, sales)[0, 1]
                    if np.isfinite(correlation):
                        trend_strength = np.clip(correlation ** 2, 0, 1)
                    else:
                        trend_strength = 0
                except:
                    trend_strength = 0
            else:
                trend_strength = 0
            
            # Calculate velocity and acceleration for multiple windows
            velocity_windows = [7, 14, 21, 28]
            
            # Base velocity (day-over-day changes)
            base_velocity = np.gradient(sales)
            base_velocity = np.where(np.isfinite(base_velocity), base_velocity, 0)
            
            for window in velocity_windows:
                if len(sales) >= window:
                    # Smoothed velocity using moving average
                    if window <= len(base_velocity):
                        smoothed_velocity = np.convolve(
                            base_velocity, 
                            np.ones(window)/window, 
                            mode='same'
                        )
                    else:
                        smoothed_velocity = np.full_like(base_velocity, base_velocity.mean())
                    
                    smoothed_velocity = np.clip(smoothed_velocity, -1e6, 1e6)
                    
                    # Acceleration (change in velocity)
                    acceleration = np.gradient(smoothed_velocity)
                    acceleration = np.where(np.isfinite(acceleration), acceleration, 0)
                    acceleration = np.clip(acceleration, -1e6, 1e6)
                    
                    # Store features
                    group[f'sales_velocity_{window}d'] = smoothed_velocity
                    group[f'sales_acceleration_{window}d'] = acceleration
                else:
                    # Not enough data for this window
                    group[f'sales_velocity_{window}d'] = 0
                    group[f'sales_acceleration_{window}d'] = 0
            
            group['trend_strength'] = trend_strength
            return group
        
        df = df.groupby(id_col).apply(calculate_trend_features).reset_index(drop=True)
        
        # Change point indicators
        def detect_change_points(group):
            sales = group[target_col].values
            if len(sales) < 21:  # Need sufficient history
                group['change_point_score'] = 0
                return group
            
            # Simple change point detection using rolling variance
            window = 7
            rolling_var = pd.Series(sales).rolling(window).var()
            var_change = rolling_var.pct_change(window).fillna(0)
            
            group['change_point_score'] = var_change.values
            return group
        
        df = df.groupby(id_col).apply(detect_change_points).reset_index(drop=True)
        
        # Seasonality strength by period
        def calculate_seasonality_strength(group):
            sales = group[target_col].values
            if len(sales) < 28:  # Need at least 4 weeks
                group['weekly_seasonality'] = 0
                group['monthly_seasonality'] = 0
                return group
            
            # Weekly seasonality (day of week effect)
            dow_means = group.groupby('day_of_week')[target_col].mean()
            overall_mean = group[target_col].mean()
            weekly_var = np.var(dow_means) if overall_mean > 0 else 0
            weekly_seasonality = weekly_var / (overall_mean + 1e-6)
            
            # Monthly seasonality (day of month effect)
            dom_means = group.groupby('day_of_month')[target_col].mean()
            monthly_var = np.var(dom_means) if overall_mean > 0 else 0
            monthly_seasonality = monthly_var / (overall_mean + 1e-6)
            
            group['weekly_seasonality'] = weekly_seasonality
            group['monthly_seasonality'] = monthly_seasonality
            
            return group
        
        df = df.groupby(id_col).apply(calculate_seasonality_strength).reset_index(drop=True)
        
        self.logger.info("Created advanced temporal features")
        return df
    
    def _create_enhanced_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced promotional features with effectiveness metrics"""
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        
        if 'promo_flag' not in df.columns:
            return df
        
        # Promotional effectiveness features
        def calculate_promo_effectiveness(group):
            # Calculate baseline sales (non-promo periods)
            non_promo_sales = group[group['promo_flag'] == 0][target_col]
            promo_sales = group[group['promo_flag'] == 1][target_col]
            
            if len(non_promo_sales) > 5 and len(promo_sales) > 5:  # Need sufficient data
                baseline = non_promo_sales.mean()
                promo_mean = promo_sales.mean()
                if baseline > 1e-6:  # Avoid division by zero
                    promo_lift = (promo_mean - baseline) / baseline
                    promo_effectiveness = np.clip(promo_lift, -10, 10)  # Clip extreme values
                else:
                    promo_effectiveness = 0
            else:
                promo_effectiveness = 0
            
            group['historical_promo_effectiveness'] = promo_effectiveness
            return group
        
        df = df.groupby(id_col).apply(calculate_promo_effectiveness).reset_index(drop=True)
        
        # Promotional frequency and fatigue
        df['promo_frequency_30d'] = df.groupby(id_col)['promo_flag'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        
        # Days since last promotion
        def days_since_promo(group):
            promo_dates = group[group['promo_flag'] == 1].index
            if len(promo_dates) == 0:
                group['days_since_last_promo'] = 999
                return group
            
            days_since = []
            last_promo_idx = -1
            
            for i, row in group.iterrows():
                if row['promo_flag'] == 1:
                    last_promo_idx = i
                    days_since.append(0)
                else:
                    if last_promo_idx >= 0:
                        days_since.append(i - last_promo_idx)
                    else:
                        days_since.append(999)
            
            group['days_since_last_promo'] = days_since
            return group
        
        df = df.groupby(id_col).apply(days_since_promo).reset_index(drop=True)
        
        # Promotional depth and discount features
        if 'promo_discount' in df.columns:
            # Discount relative to historical average
            discount_ratio = df.groupby(id_col)['promo_discount'].transform(
                lambda x: x / np.maximum(x.rolling(90, min_periods=1).mean(), 1e-6)
            )
            df['discount_vs_historical_avg'] = np.clip(discount_ratio.fillna(1), 0.1, 10)  # Reasonable bounds
            
            # Deep discount flag (top 25% of discounts for this SKU)
            df['is_deep_discount'] = (df.groupby(id_col)['promo_discount'].transform(
                lambda x: x >= x.quantile(0.75)
            ) & (df['promo_flag'] == 1)).astype(int)
        
        self.logger.info("Created enhanced promotional features")
        return df
    
    def _create_conservative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create conservative features to counterbalance over-optimistic bias"""
        target_col = self.config['data']['target_column']
        id_col = self.config['data']['id_column']
        timestamp_col = self.config['data']['timestamp_column']
        
        # Sales decline indicators
        def calculate_decline_features(group):
            sales = group[target_col].values
            if len(sales) < 14:
                group['sales_decline_indicator'] = 0
                group['consecutive_decline_days'] = 0
                group['decline_magnitude'] = 0
                return group
            
            # Recent decline indicator (last 7 days trending down)
            recent_sales = sales[-7:]
            if len(recent_sales) >= 7:
                # Check if recent trend is declining
                x = np.arange(len(recent_sales))
                if np.std(recent_sales) > 1e-6:
                    try:
                        slope = np.polyfit(x, recent_sales, 1)[0]
                        decline_indicator = 1 if slope < -0.1 else 0  # Significant decline
                    except:
                        decline_indicator = 0
                else:
                    decline_indicator = 0
            else:
                decline_indicator = 0
            
            # Consecutive declining days
            daily_changes = np.diff(sales[-14:])  # Last 2 weeks
            consecutive_decline = 0
            current_streak = 0
            for change in reversed(daily_changes):
                if change < 0:
                    current_streak += 1
                else:
                    break
            consecutive_decline = min(current_streak, 14)  # Cap at 14 days
            
            # Decline magnitude (how much below recent peak)
            recent_peak = np.max(sales[-30:]) if len(sales) >= 30 else np.max(sales)
            current_level = sales[-1]
            decline_magnitude = max(0, (recent_peak - current_level) / (recent_peak + 1e-6))
            decline_magnitude = np.clip(decline_magnitude, 0, 1)
            
            group['sales_decline_indicator'] = decline_indicator
            group['consecutive_decline_days'] = consecutive_decline
            group['decline_magnitude'] = decline_magnitude
            
            return group
        
        df = df.groupby(id_col).apply(calculate_decline_features).reset_index(drop=True)
        
        # Volatility risk score
        def calculate_volatility_risk(group):
            sales = group[target_col].values
            if len(sales) < 21:
                group['volatility_risk_score'] = 0
                return group
            
            # Calculate coefficient of variation over different periods
            cv_7d = np.std(sales[-7:]) / (np.mean(sales[-7:]) + 1e-6) if len(sales) >= 7 else 0
            cv_14d = np.std(sales[-14:]) / (np.mean(sales[-14:]) + 1e-6) if len(sales) >= 14 else 0
            cv_21d = np.std(sales[-21:]) / (np.mean(sales[-21:]) + 1e-6) if len(sales) >= 21 else 0
            
            # Average CV as volatility risk (higher = more risky = more conservative)
            avg_cv = np.mean([cv_7d, cv_14d, cv_21d])
            volatility_risk = np.clip(avg_cv, 0, 5)  # Cap extreme volatility
            
            group['volatility_risk_score'] = volatility_risk
            return group
        
        df = df.groupby(id_col).apply(calculate_volatility_risk).reset_index(drop=True)
        
        # Category growth stability index (conservative growth assessment)
        if 'category' in df.columns:
            # Calculate category growth stability rather than assuming maturity
            category_stats = df.groupby(['category', timestamp_col]).agg({
                target_col: 'sum'
            }).reset_index()
            
            # Calculate growth rates over multiple periods
            category_stats['growth_7d'] = category_stats.groupby('category')[target_col].pct_change(7).fillna(0)
            category_stats['growth_30d'] = category_stats.groupby('category')[target_col].pct_change(30).fillna(0)
            category_stats['growth_90d'] = category_stats.groupby('category')[target_col].pct_change(90).fillna(0)
            
            # Growth instability = high variance in growth rates
            growth_stability = category_stats.groupby('category').agg({
                'growth_7d': 'std',
                'growth_30d': 'std', 
                'growth_90d': 'mean'
            }).reset_index()
            
            # Instability score: high short-term volatility + negative long-term trend
            growth_stability['instability_score'] = np.clip(
                growth_stability['growth_7d'] * 0.4 +  # Short-term volatility
                growth_stability['growth_30d'] * 0.3 +  # Medium-term volatility
                np.maximum(0, -growth_stability['growth_90d']) * 0.3,  # Long-term decline
                0, 2
            )
            
            # Merge back to main dataframe
            df = df.merge(
                growth_stability[['category', 'instability_score']], 
                on='category', 
                how='left'
            )
            df['category_instability_index'] = df['instability_score'].fillna(0.5)
            df = df.drop('instability_score', axis=1)
        else:
            df['category_instability_index'] = 0.5
        
        # Market fragmentation score (internal competition proxy)
        if 'category' in df.columns and 'subcategory' in df.columns:
            # Calculate how fragmented the subcategory sales are
            fragmentation_stats = df.groupby(['subcategory', timestamp_col]).agg({
                target_col: ['sum', 'std', 'count']
            }).reset_index()
            
            fragmentation_stats.columns = ['subcategory', timestamp_col, 'total_sales', 'sales_std', 'sku_count']
            
            # Fragmentation = high std relative to mean (sales spread across many SKUs)
            fragmentation_stats['sales_concentration'] = (
                fragmentation_stats['sales_std'] / (fragmentation_stats['total_sales'] / fragmentation_stats['sku_count'] + 1e-6)
            ).fillna(0)
            
            # High fragmentation = more internal competition for market share
            fragmentation_stats['fragmentation_score'] = np.clip(
                fragmentation_stats['sales_concentration'], 0, 3
            )
            
            # Get average fragmentation by subcategory
            avg_fragmentation = fragmentation_stats.groupby('subcategory')['fragmentation_score'].mean().reset_index()
            
            # Merge back to main dataframe
            df = df.merge(avg_fragmentation, on='subcategory', how='left')
            df['market_fragmentation_score'] = df['fragmentation_score'].fillna(0.5)
            df = df.drop('fragmentation_score', axis=1)
        else:
            df['market_fragmentation_score'] = 0.5
        
        # Portfolio concentration risk (how dependent we are on few SKUs)
        if 'category' in df.columns:
            # Calculate how concentrated sales are within each category
            concentration_stats = df.groupby(['category', timestamp_col]).apply(
                lambda x: self._calculate_concentration_ratio(x[target_col], x[id_col])
            ).reset_index()
            concentration_stats.columns = ['category', timestamp_col, 'concentration_ratio']
            
            # Get average concentration by category
            avg_concentration = concentration_stats.groupby('category')['concentration_ratio'].mean().reset_index()
            
            # High concentration = risk (few SKUs drive most sales)
            # This makes forecasts riskier as they depend on fewer products
            df = df.merge(avg_concentration, on='category', how='left')
            df['portfolio_concentration_risk'] = df['concentration_ratio'].fillna(0.5)
            df = df.drop('concentration_ratio', axis=1)
        else:
            df['portfolio_concentration_risk'] = 0.5
        
        # Economic headwind indicator (based on overall market trends)
        # Create a temporary dataframe to avoid index/column conflicts
        market_data = df.groupby(timestamp_col)[target_col].sum().reset_index()
        market_data['market_trend'] = market_data[target_col].pct_change(7).fillna(0)
        market_data['market_headwind'] = (market_data['market_trend'] < -0.02).astype(int)
        
        # Keep only necessary columns for merge
        market_trend_df = market_data[[timestamp_col, 'market_headwind']].copy()
        
        # Merge back to main dataframe
        df = df.merge(market_trend_df, on=timestamp_col, how='left')
        df['market_headwind'] = df['market_headwind'].fillna(0)
        
        self.logger.info("Created conservative counter-features")
        return df
    
    def _calculate_concentration_ratio(self, sales_values, sku_ids):
        """Calculate Herfindahl concentration ratio for portfolio risk assessment"""
        if len(sales_values) == 0:
            return 0.5
        
        # Calculate market share of each SKU within the group
        total_sales = sales_values.sum()
        if total_sales <= 0:
            return 0.5
        
        # Get sales by SKU
        sku_sales = pd.Series(sales_values.values, index=sku_ids).groupby(level=0).sum()
        market_shares = sku_sales / total_sales
        
        # Herfindahl index: sum of squared market shares
        # Higher values indicate more concentration (riskier)
        herfindahl = (market_shares ** 2).sum()
        
        return np.clip(herfindahl, 0, 1)