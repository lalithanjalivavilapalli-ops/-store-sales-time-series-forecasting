# Store Sales Forecasting Using Historical Retail Data

Foundations of Data Science project using the Kaggle dataset **Store Sales - Time Series Forecasting** (Corporación Favorita, Ecuador).

## Objective
Predict daily sales for individual retail stores from historical sales data, using store information, promotions, holidays, transactions and oil prices.

## Dataset
Upload `store-sales-time-series-forecasting.zip` when the notebook asks for it. Files used:

| File | Used for |
|---|---|
| `train.csv` | Sales and promotions per date, store and product family |
| `stores.csv` | City, state, store type, cluster |
| `transactions.csv` | Daily transactions per store |
| `oil.csv` | Daily oil price |
| `holidays_events.csv` | National, regional and local holidays/events |

`test.csv` and `sample_submission.csv` are not used because they contain no actual sales values.

Data covers 2013-01-01 to 2017-08-15, with 54 stores and 33 product families (3,000,888 training rows).

## How to Run (Google Colab)
1. Open `store_sales_forecasting.ipynb` in Google Colab (File > Upload notebook), or paste `store_sales_forecasting.py` into a cell.
2. Run all cells and upload the dataset ZIP when prompted.
3. Plots are shown in the notebook and saved to the `outputs/` folder. At the end, a ZIP of all outputs downloads automatically.

## Method
1. **Load**: ZIP is extracted and the CSV files are detected automatically.
2. **Clean**: missing values checked, duplicates removed (none found), 43 missing oil prices filled by interpolation, dates parsed, transferred holidays and "Work Day" entries excluded.
3. **Aggregate**: sales and promotion counts summed over all product families to get one row per date and store (90,936 rows).
4. **Merge**: transactions, store info, oil price and holiday flags (national, regional, local).
5. **Features**: year, month, day, day of week, day of year, week of year, weekend, payday, store type, cluster, city/state codes, promotions, transactions, oil price, holiday flags.
6. **Split**: chronological, first 80% of dates for training and last 20% for testing (no shuffling).
7. **Model**: Random Forest Regressor (150 trees, max depth 18, min leaf 3).
8. **Evaluate**: MAE, RMSE and R².

## Results
Training: 2013-01-01 to 2016-09-11 (72,738 rows). Testing: 2016-09-12 to 2017-08-15 (18,198 rows).

| Dataset | MAE | RMSE | R² |
|---|---|---|---|
| Training (store-day level) | 521.97 | 1,137.10 | 0.9850 |
| Testing (store-day level) | 1,987.07 | 3,317.56 | 0.9149 |
| Testing (total daily sales, all stores) | 79,901.17 | 107,759.42 | 0.6631 |

Top features by importance: `transactions` (0.7689), `onpromotion` (0.0888), `store_nbr` (0.0259), `year` (0.0182), `day_of_year` (0.0166).

Observations from the data:
- Average store-day sales: 13,893.61 on holidays vs 11,620.31 on non-holidays.
- Average store-day sales: 14,789.94 on weekends vs 10,613.76 on weekdays.
- Best-selling product family: GROCERY I.

## Output Graphs (`outputs/`)
| File | Shows |
|---|---|
| `01_total_daily_sales_trend.png` | Daily sales with 30-day moving average |
| `02_monthly_total_sales.png` | Monthly total sales |
| `03_seasonality_month_dayofweek.png` | Average sales by month and day of week |
| `04_sales_by_product_family.png` | Total sales per product family |
| `05_store_type_and_top_stores.png` | Sales by store type and top 10 stores |
| `06_holiday_and_promotion_effect.png` | Holiday vs non-holiday, promotions vs sales |
| `07_oil_price_and_transactions.png` | Oil price over time, transactions vs sales |
| `08_correlation_heatmap.png` | Correlation between key variables |
| `09_actual_vs_predicted_total_daily.png` | Actual vs predicted total daily sales |
| `10_actual_vs_predicted_scatter.png` | Actual vs predicted per store-day |
| `11_actual_vs_predicted_example_store.png` | Actual vs predicted for the highest-selling store |
| `12_prediction_error_distribution.png` | Distribution of prediction errors |
| `13_feature_importance.png` | Top 15 feature importances |

Other files: `model_metrics.csv`, `feature_importance.csv`, `final_summary.txt`.

## Limitations
- The model uses same-day transactions, which would not be known in advance in a real forecast. This is the most important feature and likely raises the store-day score.
- The total daily R² (0.6631) is lower than the store-day R² (0.9149). In the plot, the model underpredicts many days in 2017 and does not capture the New Year's Day closure, when total sales drop to almost zero.
- No lag or rolling sales features are used.
- Product families are aggregated, so per-family forecasts are not produced.

## Future Improvements (Version 2)
- Add lag and rolling-average sales features.
- Forecast per product family and evaluate with RMSLE.
- Remove or forecast transactions to make predictions usable in advance.
- Tune hyperparameters and compare with gradient boosting.
- Add store closure and special-event handling.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
