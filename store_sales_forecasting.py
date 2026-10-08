# %% [markdown]
# # Store Sales Forecasting Using Historical Retail Data
# Foundations of Data Science project - Kaggle "Store Sales - Time Series Forecasting"

# %% 1. Import libraries
import os
import glob
import shutil
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    matplotlib.use("Agg")

try:
    display
except NameError:
    def display(x):
        print(x)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
pd.set_option("display.max_columns", 50)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(name):
    """Save the current figure as a PNG in the outputs folder and show it."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# %% 2. Upload and extract the ZIP file
if IN_COLAB:
    print("Please upload the dataset ZIP file (store-sales-time-series-forecasting.zip)...")
    uploaded = files.upload()
    zip_name = [f for f in uploaded.keys() if f.lower().endswith(".zip")][0]
else:
    zip_name = os.environ["DATASET_ZIP"]  # used only when running outside Colab

extract_dir = "store_sales_data"
with zipfile.ZipFile(zip_name, "r") as z:
    z.extractall(extract_dir)
print(f"Extracted '{os.path.basename(zip_name)}' into folder '{extract_dir}'")

# %% 3. Identify and load the CSV files
csv_paths = [p for p in glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
             if "__MACOSX" not in p]
csv_map = {os.path.basename(p).lower().replace(".csv", ""): p for p in csv_paths}
print("CSV files found:", list(csv_map.keys()))

required = ["train", "stores", "oil", "holidays_events", "transactions"]
missing_files = [r for r in required if r not in csv_map]
if missing_files:
    raise FileNotFoundError(f"Required CSV files missing from ZIP: {missing_files}")

train = pd.read_csv(csv_map["train"], parse_dates=["date"])
stores = pd.read_csv(csv_map["stores"])
oil = pd.read_csv(csv_map["oil"], parse_dates=["date"])
holidays = pd.read_csv(csv_map["holidays_events"], parse_dates=["date"])
transactions = pd.read_csv(csv_map["transactions"], parse_dates=["date"])

for name, d in [("train", train), ("stores", stores), ("oil", oil),
                ("holidays_events", holidays), ("transactions", transactions)]:
    print(f"\n{name}: shape = {d.shape}")
    display(d.head(3))

# %% 4. Data cleaning
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

datasets = {"train": train, "stores": stores, "oil": oil,
            "holidays": holidays, "transactions": transactions}

# Missing values report
for name, d in datasets.items():
    nulls = d.isnull().sum()
    print(f"\nMissing values in {name}:")
    print(nulls[nulls > 0] if nulls.sum() > 0 else "None")

# Remove duplicates
for name in datasets:
    before = len(datasets[name])
    datasets[name] = datasets[name].drop_duplicates().reset_index(drop=True)
    print(f"{name}: removed {before - len(datasets[name])} duplicate rows")
train, stores, oil, holidays, transactions = (datasets["train"], datasets["stores"],
                                              datasets["oil"], datasets["holidays"],
                                              datasets["transactions"])

# Handle missing / invalid values
train["onpromotion"] = train["onpromotion"].fillna(0)
train["sales"] = train["sales"].fillna(0).clip(lower=0)

# Oil prices: fill missing dates and values using interpolation
oil = oil.rename(columns={"dcoilwtico": "oil_price"})
all_dates = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
oil = oil.set_index("date").reindex(all_dates).rename_axis("date").reset_index()
oil["oil_price"] = oil["oil_price"].interpolate(method="linear").bfill().ffill()

# Holidays: keep real holidays/events (remove transferred days and work days)
holidays = holidays[(holidays["transferred"] == False) & (holidays["type"] != "Work Day")].copy()
print(f"\nHoliday/event records used: {len(holidays)}")

# %% 5. Basic EDA on raw data
print("\n" + "=" * 60)
print("BASIC EDA")
print("=" * 60)
print(f"Date range: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Number of stores: {train['store_nbr'].nunique()}")
print(f"Number of product families: {train['family'].nunique()}")
print("\nSales summary statistics:")
print(train["sales"].describe())

# Total sales by product family (computed before aggregation)
family_sales = train.groupby("family")["sales"].sum().sort_values()

# %% 6. Aggregate sales (date x store level)
daily_store = (train.groupby(["date", "store_nbr"])
               .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
               .reset_index())
del train  # free memory
print(f"\nAggregated data shape (date x store): {daily_store.shape}")

# %% 7. Merge datasets
df = daily_store.merge(transactions, on=["date", "store_nbr"], how="left")
df["transactions"] = df["transactions"].fillna(0)

df = df.merge(stores, on="store_nbr", how="left")
df = df.merge(oil, on="date", how="left")

# Holiday features (national, regional, local)
national_dates = set(holidays.loc[holidays["locale"] == "National", "date"])
df["is_national_holiday"] = df["date"].isin(national_dates).astype(int)

regional = (holidays[holidays["locale"] == "Regional"][["date", "locale_name"]]
            .drop_duplicates().rename(columns={"locale_name": "state"}))
regional["is_regional_holiday"] = 1
df = df.merge(regional, on=["date", "state"], how="left")

local = (holidays[holidays["locale"] == "Local"][["date", "locale_name"]]
         .drop_duplicates().rename(columns={"locale_name": "city"}))
local["is_local_holiday"] = 1
df = df.merge(local, on=["date", "city"], how="left")

df[["is_regional_holiday", "is_local_holiday"]] = df[["is_regional_holiday", "is_local_holiday"]].fillna(0)
df["is_holiday"] = ((df["is_national_holiday"] + df["is_regional_holiday"] + df["is_local_holiday"]) > 0).astype(int)

print("\nMerged dataset shape:", df.shape)
print("Remaining missing values:", int(df.isnull().sum().sum()))

# %% 8. Feature engineering (time-based features)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_payday"] = ((df["day"] == 15) | (df["date"].dt.is_month_end)).astype(int)

# Encode store information
df["city_code"] = df["city"].astype("category").cat.codes
df["state_code"] = df["state"].astype("category").cat.codes
df = pd.concat([df, pd.get_dummies(df["type"], prefix="store_type", dtype=int)], axis=1)

df = df.sort_values(["date", "store_nbr"]).reset_index(drop=True)

# %% 9. EDA visualizations
total_daily = df.groupby("date")["sales"].sum()

# 9.1 Total daily sales trend
plt.figure()
plt.plot(total_daily.index, total_daily.values, alpha=0.4, label="Daily Sales")
plt.plot(total_daily.index, total_daily.rolling(30).mean(), color="red", linewidth=2, label="30-Day Moving Average")
plt.title("Total Daily Sales Trend (All Stores)")
plt.xlabel("Date"); plt.ylabel("Total Sales"); plt.legend()
save_fig("01_total_daily_sales_trend.png")

# 9.2 Monthly total sales
monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
monthly["period"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
plt.figure()
plt.plot(monthly["period"], monthly["sales"], marker="o")
plt.title("Monthly Total Sales")
plt.xlabel("Month"); plt.ylabel("Total Sales")
save_fig("02_monthly_total_sales.png")

# 9.3 Seasonality: month and day of week
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x="month", y="sales", data=df, ax=axes[0], color="steelblue", errorbar=None)
axes[0].set_title("Average Store Sales by Month")
sns.barplot(x="day_of_week", y="sales", data=df, ax=axes[1], color="darkorange", errorbar=None)
axes[1].set_title("Average Store Sales by Day of Week (0=Mon, 6=Sun)")
save_fig("03_seasonality_month_dayofweek.png")

# 9.4 Sales by product family
plt.figure(figsize=(10, 9))
family_sales.plot(kind="barh", color="teal")
plt.title("Total Sales by Product Family")
plt.xlabel("Total Sales")
save_fig("04_sales_by_product_family.png")

# 9.5 Sales by store type and top stores
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x="type", y="sales", data=df, ax=axes[0], errorbar=None, order=sorted(df["type"].unique()))
axes[0].set_title("Average Store Sales by Store Type")
top_stores = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_stores.index.astype(str), y=top_stores.values, ax=axes[1], color="purple")
axes[1].set_title("Top 10 Stores by Total Sales")
axes[1].set_xlabel("Store Number"); axes[1].set_ylabel("Total Sales")
save_fig("05_store_type_and_top_stores.png")

# 9.6 Holiday and promotion effect
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x="is_holiday", y="sales", data=df, ax=axes[0], errorbar=None)
axes[0].set_title("Average Store Sales: Holiday (1) vs Non-Holiday (0)")
promo_rows = df[df["onpromotion"] > 0]
promo_sample = promo_rows.sample(min(5000, len(promo_rows)), random_state=42)
sns.scatterplot(x="onpromotion", y="sales", data=promo_sample, alpha=0.4, ax=axes[1])
axes[1].set_title("Promotions vs Sales (sample of store-days)")
save_fig("06_holiday_and_promotion_effect.png")

# 9.7 Oil price and transactions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
oil_sales = df.groupby("date").agg(sales=("sales", "sum"), oil_price=("oil_price", "first")).reset_index()
axes[0].plot(oil_sales["date"], oil_sales["oil_price"], color="black")
axes[0].set_title("Oil Price Over Time"); axes[0].set_xlabel("Date"); axes[0].set_ylabel("Oil Price (USD)")
trans_rows = df[df["transactions"] > 0]
trans_sample = trans_rows.sample(min(5000, len(trans_rows)), random_state=42)
sns.scatterplot(x="transactions", y="sales", data=trans_sample, alpha=0.4, ax=axes[1], color="green")
axes[1].set_title("Transactions vs Sales (sample of store-days)")
save_fig("07_oil_price_and_transactions.png")

# 9.8 Correlation heatmap
corr_cols = ["sales", "onpromotion", "transactions", "oil_price", "is_holiday",
             "is_weekend", "month", "day_of_week", "cluster"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
save_fig("08_correlation_heatmap.png")

# %% 10. Chronological train/test split
feature_cols = ["store_nbr", "onpromotion", "transactions", "oil_price", "cluster",
                "city_code", "state_code", "is_national_holiday", "is_regional_holiday",
                "is_local_holiday", "is_holiday", "year", "month", "day", "day_of_week",
                "day_of_year", "week_of_year", "is_weekend", "is_payday"] + \
               [c for c in df.columns if c.startswith("store_type_")]

unique_dates = np.sort(df["date"].unique())
split_date = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.8)])

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

X_train, y_train = train_df[feature_cols], train_df["sales"]
X_test, y_test = test_df[feature_cols], test_df["sales"]

print("\n" + "=" * 60)
print("CHRONOLOGICAL TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training period: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)")
print(f"Testing period : {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")

# %% 11. Train the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=3,
                              random_state=42, n_jobs=-1)
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)
print("Training complete.")

# %% 12. Predictions and evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


tr_mae, tr_rmse, tr_r2 = evaluate(y_train, train_pred)
te_mae, te_rmse, te_r2 = evaluate(y_test, test_pred)

test_results = test_df[["date", "store_nbr", "sales"]].copy()
test_results["predicted"] = test_pred
daily_results = test_results.groupby("date")[["sales", "predicted"]].sum()
d_mae, d_rmse, d_r2 = evaluate(daily_results["sales"], daily_results["predicted"])

metrics_table = pd.DataFrame({
    "MAE": [tr_mae, te_mae, d_mae],
    "RMSE": [tr_rmse, te_rmse, d_rmse],
    "R2": [tr_r2, te_r2, d_r2]
}, index=["Training (store-day level)", "Testing (store-day level)", "Testing (total daily sales)"])

print("\n" + "=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
display(metrics_table.round(4))
metrics_table.round(4).to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"))

# %% 13. Actual vs predicted graphs
plt.figure()
plt.plot(daily_results.index, daily_results["sales"], label="Actual", linewidth=2)
plt.plot(daily_results.index, daily_results["predicted"], label="Predicted", linewidth=2, linestyle="--")
plt.title("Actual vs Predicted Total Daily Sales (Test Period)")
plt.xlabel("Date"); plt.ylabel("Total Sales"); plt.legend()
save_fig("09_actual_vs_predicted_total_daily.png")

plt.figure(figsize=(7, 7))
sample = test_results.sample(min(5000, len(test_results)), random_state=42)
plt.scatter(sample["sales"], sample["predicted"], alpha=0.4)
lims = [0, max(sample["sales"].max(), sample["predicted"].max())]
plt.plot(lims, lims, color="red", linestyle="--", label="Perfect Prediction")
plt.title("Actual vs Predicted Sales (Store-Day Level)")
plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales"); plt.legend()
save_fig("10_actual_vs_predicted_scatter.png")

example_store = test_results.groupby("store_nbr")["sales"].sum().idxmax()
store_data = test_results[test_results["store_nbr"] == example_store].set_index("date")
plt.figure()
plt.plot(store_data.index, store_data["sales"], label="Actual", linewidth=2)
plt.plot(store_data.index, store_data["predicted"], label="Predicted", linewidth=2, linestyle="--")
plt.title(f"Actual vs Predicted Sales - Store {example_store} (Highest-Selling Store in Test Period)")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend()
save_fig("11_actual_vs_predicted_example_store.png")

residuals = test_results["sales"] - test_results["predicted"]
plt.figure()
sns.histplot(residuals, bins=60, kde=True, color="slateblue")
plt.title("Distribution of Prediction Errors (Actual - Predicted)")
plt.xlabel("Error"); plt.ylabel("Frequency")
save_fig("12_prediction_error_distribution.png")

# %% 14. Feature importance
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=importance.values[:15], y=importance.index[:15], color="seagreen")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance"); plt.ylabel("Feature")
save_fig("13_feature_importance.png")
importance.round(4).to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), header=["importance"])

# %% 15. Final summary
summary_lines = [
    "=" * 60,
    "FINAL SUMMARY OF MODEL PERFORMANCE",
    "=" * 60,
    f"Dataset period            : {df['date'].min().date()} to {df['date'].max().date()}",
    f"Stores / Product families : {df['store_nbr'].nunique()} stores aggregated over all product families",
    "Model used                : Random Forest Regressor (Scikit-learn)",
    f"Training period           : {train_df['date'].min().date()} to {train_df['date'].max().date()}",
    f"Testing period            : {test_df['date'].min().date()} to {test_df['date'].max().date()}",
    "-" * 60,
    f"Test MAE  (store-day)     : {te_mae:,.2f}",
    f"Test RMSE (store-day)     : {te_rmse:,.2f}",
    f"Test R2   (store-day)     : {te_r2:.4f}",
    f"Test R2   (total daily)   : {d_r2:.4f}",
    "-" * 60,
    "Top 5 most important features:",
]
for i, (feat, val) in enumerate(importance.head(5).items(), 1):
    summary_lines.append(f"  {i}. {feat} ({val:.4f})")
summary_lines += [
    "-" * 60,
    "Observations from the data:",
    f"  - Average store-day sales on holidays    : {df[df['is_holiday'] == 1]['sales'].mean():,.2f}",
    f"  - Average store-day sales on non-holidays: {df[df['is_holiday'] == 0]['sales'].mean():,.2f}",
    f"  - Average store-day sales on weekends    : {df[df['is_weekend'] == 1]['sales'].mean():,.2f}",
    f"  - Average store-day sales on weekdays    : {df[df['is_weekend'] == 0]['sales'].mean():,.2f}",
    f"  - Best-selling product family            : {family_sales.idxmax()}",
    "-" * 60,
    "Limitations: the model uses same-day transactions and does not use past sales (lag) features;",
    "future improvements can include lag/rolling features, per-family modelling and hyperparameter tuning.",
    "=" * 60,
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(os.path.join(OUTPUT_DIR, "final_summary.txt"), "w") as f:
    f.write(summary_text)

# %% 16. Zip all outputs for download (Colab only)
if IN_COLAB:
    shutil.make_archive("store_sales_outputs", "zip", OUTPUT_DIR)
    files.download("store_sales_outputs.zip")
