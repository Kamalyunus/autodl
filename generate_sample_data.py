#!/usr/bin/env python3
"""
Generate realistic sample dataset for SKU-day level forecasting
Creates various patterns: new SKUs, discontinued, intermittent, seasonal, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)
NUM_SKUS = 500

# SKU categories and their characteristics
CATEGORIES = {
    'Electronics': {'base_demand': 50, 'seasonality': 'low', 'price_range': (100, 1000)},
    'Clothing': {'base_demand': 100, 'seasonality': 'high', 'price_range': (20, 200)},
    'Food': {'base_demand': 200, 'seasonality': 'medium', 'price_range': (5, 50)},
    'Home': {'base_demand': 30, 'seasonality': 'medium', 'price_range': (30, 300)},
    'Sports': {'base_demand': 40, 'seasonality': 'high', 'price_range': (25, 250)}
}

SUBCATEGORIES = {
    'Electronics': ['Phones', 'Laptops', 'Tablets', 'Accessories'],
    'Clothing': ['Shirts', 'Pants', 'Shoes', 'Accessories'],
    'Food': ['Snacks', 'Beverages', 'Dairy', 'Frozen'],
    'Home': ['Furniture', 'Kitchen', 'Decor', 'Bedding'],
    'Sports': ['Equipment', 'Apparel', 'Footwear', 'Accessories']
}

BRANDS = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Private Label']

def generate_sku_metadata():
    """Generate SKU metadata with different characteristics"""
    skus = []
    
    for i in range(NUM_SKUS):
        category = np.random.choice(list(CATEGORIES.keys()))
        subcategory = np.random.choice(SUBCATEGORIES[category])
        brand = np.random.choice(BRANDS)
        
        # Determine SKU pattern
        pattern_prob = np.random.random()
        if pattern_prob < 0.15:
            pattern = 'new'  # 15% new SKUs (launch during the period)
        elif pattern_prob < 0.25:
            pattern = 'discontinued'  # 10% discontinued
        elif pattern_prob < 0.40:
            pattern = 'intermittent'  # 15% intermittent/low volume
        elif pattern_prob < 0.60:
            pattern = 'seasonal'  # 20% seasonal
        else:
            pattern = 'regular'  # 40% regular
        
        # Price based on category
        price_range = CATEGORIES[category]['price_range']
        price = np.random.uniform(price_range[0], price_range[1])
        
        sku = {
            'sku': f'SKU_{i:04d}',
            'category': category,
            'subcategory': subcategory,
            'brand': brand,
            'product_group': f'{category}_{subcategory}',
            'pattern': pattern,
            'price': round(price, 2),
            'base_demand': CATEGORIES[category]['base_demand'] * np.random.uniform(0.5, 1.5)
        }
        skus.append(sku)
    
    return pd.DataFrame(skus)

def generate_sales_pattern(sku_info, dates):
    """Generate sales pattern based on SKU characteristics"""
    pattern = sku_info['pattern']
    base_demand = sku_info['base_demand']
    category = sku_info['category']
    
    # Initialize sales array
    sales = np.zeros(len(dates))
    
    # Date indices
    date_indices = pd.DatetimeIndex(dates)
    days_of_year = date_indices.dayofyear
    days_of_week = date_indices.dayofweek
    months = date_indices.month
    
    if pattern == 'new':
        # New SKU launches randomly in the period
        launch_day = np.random.randint(90, len(dates) - 90)
        # Ramp up period
        for i in range(launch_day, len(dates)):
            ramp_factor = min(1.0, (i - launch_day) / 60)  # 60-day ramp up
            sales[i] = base_demand * ramp_factor
    
    elif pattern == 'discontinued':
        # Discontinued SKU stops randomly
        discontinue_day = np.random.randint(len(dates) // 2, len(dates) - 60)
        # Decline period
        for i in range(discontinue_day):
            sales[i] = base_demand
        for i in range(discontinue_day, min(discontinue_day + 60, len(dates))):
            decline_factor = max(0, 1 - (i - discontinue_day) / 60)
            sales[i] = base_demand * decline_factor
    
    elif pattern == 'intermittent':
        # Low volume with many zeros
        for i in range(len(dates)):
            if np.random.random() < 0.3:  # 30% chance of sales
                sales[i] = base_demand * np.random.uniform(0.1, 0.5)
    
    elif pattern == 'seasonal':
        # Strong seasonality based on category
        for i in range(len(dates)):
            if category == 'Clothing':
                # Peak in spring/summer and winter
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * days_of_year[i] / 365 - np.pi/2)
                if months[i] in [11, 12]:  # Holiday boost
                    seasonal_factor *= 1.5
            elif category == 'Sports':
                # Peak in summer
                seasonal_factor = 1 + 0.7 * np.sin(2 * np.pi * days_of_year[i] / 365 - np.pi/4)
            else:
                # Mild seasonality
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * days_of_year[i] / 365)
            
            sales[i] = base_demand * seasonal_factor
    
    else:  # regular
        # Stable demand with mild variations
        for i in range(len(dates)):
            sales[i] = base_demand
    
    # Add common patterns to all non-zero sales
    for i in range(len(sales)):
        if sales[i] > 0:
            # Day of week effect
            if days_of_week[i] in [5, 6]:  # Weekend
                sales[i] *= np.random.uniform(1.1, 1.3)
            elif days_of_week[i] == 0:  # Monday
                sales[i] *= np.random.uniform(0.8, 0.9)
            
            # Random noise
            sales[i] *= np.random.uniform(0.7, 1.3)
            
            # Trend (slight growth over time)
            trend_factor = 1 + (i / len(dates)) * 0.1
            sales[i] *= trend_factor
    
    # Round to integers
    sales = np.round(sales).astype(int)
    
    # Add occasional stockouts (zeros)
    stockout_prob = 0.02 if pattern != 'intermittent' else 0
    for i in range(len(sales)):
        if np.random.random() < stockout_prob:
            sales[i] = 0
    
    return sales

def generate_promo_data(dates, sales):
    """Generate promotional flags and discounts"""
    promo_flags = np.zeros(len(dates), dtype=int)
    promo_discounts = np.zeros(len(dates))
    
    # Regular promotions (weekly/biweekly)
    for i in range(0, len(dates), np.random.randint(7, 21)):
        promo_duration = np.random.randint(1, 4)
        for j in range(i, min(i + promo_duration, len(dates))):
            promo_flags[j] = 1
            promo_discounts[j] = np.random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
            # Boost sales during promo
            if sales[j] > 0:
                sales[j] = int(sales[j] * (1 + promo_discounts[j] * np.random.uniform(1.5, 3)))
    
    # Special events (Black Friday, holidays, etc.)
    special_event_dates = [
        datetime(2022, 11, 25),  # Black Friday 2022
        datetime(2023, 11, 24),  # Black Friday 2023
        datetime(2022, 12, 26),  # After Christmas 2022
        datetime(2023, 12, 26),  # After Christmas 2023
        datetime(2022, 7, 4),    # July 4th 2022
        datetime(2023, 7, 4),    # July 4th 2023
    ]
    
    special_events = np.zeros(len(dates), dtype=int)
    date_list = [START_DATE + timedelta(days=i) for i in range(len(dates))]
    
    for event_date in special_event_dates:
        if START_DATE <= event_date <= END_DATE:
            idx = (event_date - START_DATE).days
            if 0 <= idx < len(dates):
                special_events[idx] = 1
                # Bigger sales boost for special events
                if sales[idx] > 0:
                    sales[idx] = int(sales[idx] * np.random.uniform(2, 4))
    
    return promo_flags, promo_discounts, special_events, sales

def generate_full_dataset():
    """Generate the complete dataset"""
    # Generate date range
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    
    # Generate SKU metadata
    sku_metadata = generate_sku_metadata()
    
    # Generate sales data for each SKU
    all_data = []
    
    for _, sku_info in sku_metadata.iterrows():
        # Generate sales pattern
        sales = generate_sales_pattern(sku_info, date_range)
        
        # Generate promo data
        promo_flags, promo_discounts, special_events, sales = generate_promo_data(date_range, sales)
        
        # Create dataframe for this SKU
        sku_data = pd.DataFrame({
            'sku': sku_info['sku'],
            'date': date_range,
            'sales': sales,
            'category': sku_info['category'],
            'subcategory': sku_info['subcategory'],
            'brand': sku_info['brand'],
            'product_group': sku_info['product_group'],
            'promo_flag': promo_flags,
            'promo_discount': promo_discounts,
            'special_event': special_events,
            'price': sku_info['price']
        })
        
        all_data.append(sku_data)
    
    # Combine all SKU data
    full_dataset = pd.concat(all_data, ignore_index=True)
    
    # Add is_holiday flag (simplified)
    holidays = [
        '2022-01-01', '2022-07-04', '2022-11-24', '2022-12-25',
        '2023-01-01', '2023-07-04', '2023-11-23', '2023-12-25'
    ]
    full_dataset['is_holiday'] = full_dataset['date'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)
    
    return full_dataset, sku_metadata

def generate_future_promo_data(sku_metadata, start_date, num_days=60):
    """Generate future promotional data for forecasting"""
    future_promo_data = []
    
    # Generate dates for future period
    future_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    for _, sku_info in sku_metadata.iterrows():
        sku_id = sku_info['sku']
        category = sku_info['category']
        
        # Initialize arrays
        promo_flags = np.zeros(len(future_dates), dtype=int)
        promo_discounts = np.zeros(len(future_dates))
        special_events = np.zeros(len(future_dates), dtype=int)
        
        # Regular promotions - vary frequency by category
        if category in ['Clothing', 'Sports']:
            promo_freq = np.random.randint(7, 14)  # More frequent promos
        else:
            promo_freq = np.random.randint(14, 28)  # Less frequent
        
        # Add regular promotions
        for i in range(0, len(future_dates), promo_freq):
            if np.random.random() < 0.7:  # 70% chance of having a promo
                promo_duration = np.random.randint(1, 4)
                for j in range(i, min(i + promo_duration, len(future_dates))):
                    promo_flags[j] = 1
                    promo_discounts[j] = np.random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
        
        # Add special events
        # Example: New Year sale, Valentine's Day, etc.
        special_event_indices = []
        for i, date in enumerate(future_dates):
            # New Year period (first week of January)
            if date.month == 1 and date.day <= 7:
                special_event_indices.append(i)
            # Valentine's Day
            elif date.month == 2 and 12 <= date.day <= 14:
                special_event_indices.append(i)
            # Spring sale (mid-March)
            elif date.month == 3 and 15 <= date.day <= 17:
                special_event_indices.append(i)
            # Summer sale (early June)
            elif date.month == 6 and 1 <= date.day <= 3:
                special_event_indices.append(i)
            # Back to school (late August)
            elif date.month == 8 and 20 <= date.day <= 25:
                special_event_indices.append(i)
            # Black Friday period
            elif date.month == 11 and 24 <= date.day <= 27:
                special_event_indices.append(i)
            # Holiday season
            elif date.month == 12 and date.day in [24, 25, 26]:
                special_event_indices.append(i)
        
        for idx in special_event_indices:
            special_events[idx] = 1
            # Special events often have bigger discounts
            if promo_flags[idx] == 0:  # If no regular promo, add one
                promo_flags[idx] = 1
                promo_discounts[idx] = np.random.choice([0.25, 0.3, 0.4, 0.5])
            else:
                # Increase existing discount
                promo_discounts[idx] = min(0.5, promo_discounts[idx] * 1.5)
        
        # Create dataframe for this SKU
        sku_future_promo = pd.DataFrame({
            'sku': sku_id,
            'date': future_dates,
            'promo_flag': promo_flags,
            'promo_discount': promo_discounts,
            'special_event': special_events
        })
        
        future_promo_data.append(sku_future_promo)
    
    # Combine all SKU data
    return pd.concat(future_promo_data, ignore_index=True)

def main():
    """Generate and save sample dataset"""
    print("Generating sample dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    dataset, sku_metadata = generate_full_dataset()
    
    # Generate future promotional data (60 days from end of dataset)
    future_promo_start = END_DATE + timedelta(days=1)
    future_promo_data = generate_future_promo_data(sku_metadata, future_promo_start, num_days=60)
    
    # Save datasets
    dataset.to_csv('data/train.csv', index=False)  # Save full dataset as train.csv
    sku_metadata.to_csv('data/sku_metadata.csv', index=False)
    future_promo_data.to_csv('data/future_promo.csv', index=False)
    
    # Print summary statistics
    print(f"\nDataset generated successfully!")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Number of SKUs: {NUM_SKUS}")
    print(f"Total records: {len(dataset):,}")
    
    print(f"\nSKU patterns:")
    pattern_counts = sku_metadata['pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} SKUs ({count/NUM_SKUS*100:.1f}%)")
    
    print(f"\nSales statistics:")
    print(f"  Mean daily sales: {dataset['sales'].mean():.1f}")
    print(f"  Max daily sales: {dataset['sales'].max()}")
    print(f"  Zero sales days: {(dataset['sales'] == 0).sum():,} ({(dataset['sales'] == 0).sum()/len(dataset)*100:.1f}%)")
    
    print(f"\nFuture promo statistics:")
    print(f"  Future promo period: {future_promo_start.date()} to {(future_promo_start + timedelta(days=59)).date()}")
    print(f"  Total promo days: {future_promo_data['promo_flag'].sum():,}")
    print(f"  Average discount: {future_promo_data[future_promo_data['promo_discount'] > 0]['promo_discount'].mean()*100:.1f}%")
    print(f"  Special events: {future_promo_data['special_event'].sum():,}")
    
    print(f"\nFiles saved:")
    print(f"  - data/train.csv (full dataset - AutoGluon will handle train/test split)")
    print(f"  - data/sku_metadata.csv")
    print(f"  - data/future_promo.csv")

if __name__ == "__main__":
    main()