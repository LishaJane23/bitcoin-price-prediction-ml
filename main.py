import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="darkgrid")

csv_path = r"C:\Users\Lisha\Desktop\Bitcoin_Price_Prediction\data\bitcoin.csv"
df = pd.read_csv(csv_path)

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# FIX 1: Removed the deprecated fillna method to clear the warning
df.ffill(inplace=True)

# Technical Indicators
df["SMA_14"] = df["Close"].rolling(window=14).mean()
df["EMA_14"] = df["Close"].ewm(span=14, adjust=False).mean()

delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# FIX 2: Predict the PRICE CHANGE (Return) instead of absolute price
df["Price_Change"] = df["Close"].diff()
df["Target_Next_Change"] = df["Price_Change"].shift(-1)
df.dropna(inplace=True)

# Use recent changes and technicals as features
features = ["Open", "High", "Low", "Close", "Volume", "Price_Change", "SMA_14", "EMA_14", "RSI_14"]
X = df[features]
y = df["Target_Next_Change"]

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train models on changes
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred_change = rf_model.predict(X_test)

xgb_model = XGBRegressor(learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred_change = xgb_model.predict(X_test)

# Reconstruct absolute prices: Yesterday's Close + Predicted Change
actual_last_close = X_test["Close"].values
y_test_prices = actual_last_close + y_test.values

rf_preds = actual_last_close + rf_pred_change
xgb_preds = actual_last_close + xgb_pred_change

def evaluate_predictions(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{model_name} Final Results] MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R2 Score: {r2:.4f}")

evaluate_predictions(y_test_prices, rf_preds, "Random Forest")
evaluate_predictions(y_test_prices, xgb_preds, "XGBoost")

plt.figure(figsize=(14, 6))
plt.plot(X_test.index, y_test_prices, label="Actual Bitcoin Price", color="black", alpha=0.6)
plt.plot(X_test.index, rf_preds, label="Random Forest Predictions", color="blue", linestyle="--")
plt.plot(X_test.index, xgb_preds, label="XGBoost Predictions", color="red", linestyle=":")
plt.title("Bitcoin Price Prediction (Stationary Return Transformation)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()