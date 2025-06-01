# AutoGluon Time Series Forecasting System

A production-ready SKU-level sales forecasting system using AutoGluon TimeSeries with deep learning models. The system processes daily sales data and generates 30-day ahead forecasts with uncertainty quantification.

## Features

- **Deep Learning Models**: Leverages AutoGluon's implementation of state-of-the-art time series models:
  - DeepAR
  - Temporal Fusion Transformer
  - PatchTST
  - DLinear
  - TiDE

- **Advanced Feature Engineering**:
  - Automatic lag features (1, 7, 14, 30 days)
  - Rolling statistics (mean, std, min, max)
  - Exponential weighted moving averages
  - Calendar features with cyclical encoding
  - Holiday detection and proximity features
  - Promotional features with interaction terms
  - Static categorical features (category, brand, etc.)

- **Robust Data Processing**:
  - Automatic missing value interpolation
  - Outlier detection and clipping
  - Time series completion for gaps
  - SKU filtering by minimum history length

- **Comprehensive Evaluation**:
  - Business-focused metrics excluding out-of-stock and promotional periods
  - Weighted MAPE using actual sales as weights
  - Bias analysis (overall, under-forecast, over-forecast)
  - Visual evaluation with multiple plot types
  - Automated evaluation reports

## Installation

1. Create and activate a Python 3.12 virtual environment:
```bash
python3.12 -m venv autodl
source autodl/bin/activate  # On Windows: autodl\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Generate sample data** (optional):
```bash
python generate_sample_data.py
```

2. **Run the forecasting pipeline**:
```bash
python forecast_main.py --config config.json --mode train_predict
```

## Usage

### Command Line Options

```bash
python forecast_main.py --config CONFIG_PATH --mode {train,predict,train_predict}
```

- `--config`: Path to configuration file (required)
- `--mode`: Execution mode
  - `train`: Train models only
  - `predict`: Generate predictions using saved models
  - `train_predict`: Train and evaluate (default)

### Data Format

Input CSV must contain:
- `sku_id`: Product identifier
- `date`: Daily timestamp (YYYY-MM-DD)
- `sales`: Target variable (units sold)

Optional columns:
- `price`: Product price
- `promo_flag`: Promotional indicator (0/1)
- `promo_discount`: Discount percentage
- `category`, `subcategory`, `brand`, `product_group`: Static features

### Future Promotional Data

To incorporate known future promotions, create a CSV file with:
- `sku_id`: Product identifier
- `date`: Future dates
- `promo_flag`: Promotional indicator (0/1)
- `promo_discount`: Discount percentage
- `special_event`: Special event indicator (0/1)

Update `config.json` to point to this file:
```json
"data": {
    "future_promo_path": "data/future_promo.csv"
}
```

## Configuration

The `config.json` file controls all aspects of the pipeline:

### Key Configuration Sections

- **data**: Input/output paths, column mappings
- **features**: Feature categorization (static, known, past covariates)
- **model**: Model selection, hyperparameters, training settings
- **preprocessing**: Missing value handling, outlier treatment
- **output**: Results paths for predictions, metrics, and plots

### Feature Types

1. **Static Features**: Time-invariant SKU attributes
   - category, subcategory, brand, product_group

2. **Known Covariates**: Features known in advance
   - Calendar features (day of week, month, holidays)
   - Promotional features (if known in advance)

3. **Past Covariates**: Historical features
   - Price, lagged sales, rolling statistics

## Output

The pipeline generates:

1. **Predictions** (`output/predictions.csv`):
   - Point forecasts and prediction intervals
   - 30-day ahead forecasts for each SKU

2. **Evaluation Metrics** (`output/metrics.json`):
   - Weighted MAPE (excluding OOS and promo periods)
   - Bias metrics (overall, under, over)
   - RMSE and MAE

3. **Evaluation Report** (`output/metrics_report.txt`):
   - Human-readable summary of performance
   - Detailed breakdown by metric

4. **Visualizations** (`output/plots/`):
   - Individual SKU forecast plots
   - Model performance comparison
   - Residual analysis
   - Feature importance (if available)
   - Actual vs forecast comparisons

## Model Details

### Hyperparameter Tuning

Each model can be configured with specific hyperparameters in `config.json`:

```json
"hyperparameters": {
    "DeepAR": {
        "max_epochs": 50,
        "batch_size": 64,
        "num_layers": 2,
        "hidden_size": 40,
        "dropout_rate": 0.1,
        "lr": 0.001
    }
}
```

### Ensemble Learning

AutoGluon automatically creates weighted ensembles of the trained models for improved accuracy. Control this with:
```json
"enable_ensemble": true
```

## Performance Considerations

- **Training Time**: Deep learning models can take significant time. Adjust `time_limit` in config.
- **Memory Usage**: Large datasets may require batch processing or reduced model complexity.
- **GPU Acceleration**: Models will automatically use GPU if available (requires CUDA-enabled PyTorch).

## Troubleshooting

1. **Memory Issues**: Reduce batch size or number of models
2. **Training Timeout**: Increase `time_limit` in configuration
3. **Missing Data**: Ensure interpolation settings handle gaps appropriately
4. **Poor Performance**: Check data quality and consider feature engineering

## License

This project is licensed under the MIT License.