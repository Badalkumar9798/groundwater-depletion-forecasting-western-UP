"""
Step 2: Forecasting Model
---------------------------
For each district we build two models on the quarterly mean depth-to-water series:

1. LINEAR TREND (baseline, easy to explain in a viva):
   Fit depth_m ~ time_index using ordinary least squares. The slope tells us
   metres of decline (or recovery) per year. This is the simplest possible
   "AI/stats" approach and is fully interpretable.

2. RANDOM FOREST with lag features (the main model):
   Features = depth at t-1, t-2, t-3, t-4 quarters + quarter-of-year (seasonality).
   Target = depth at t.
   We forecast recursively: predict t+1, feed it back in as a lag to predict t+2, etc.,
   for 12 quarters (3 years) ahead.

Risk level is derived from the LINEAR slope (metres/year of decline), since it's
the more stable/interpretable signal for a "risk label":
    slope >= 0.30 m/year  -> High risk
    slope >= 0.10 m/year  -> Moderate risk
    slope <  0.10 m/year  -> Low risk
(slope <= 0 means water table is stable or recovering)

Output: output/forecast_results.json, consumed directly by the dashboard.
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

IN_PATH = "../output/district_quarterly_clean.csv"
OUT_PATH = "../output/forecast_results.json"
N_LAGS = 4
FORECAST_QUARTERS = 12  # 3 years

def make_lag_features(series_df):
    df = series_df.copy().reset_index(drop=True)
    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = df["mean_depth"].shift(lag)
    df["quarter_of_year"] = df["date"].dt.quarter
    df = df.dropna().reset_index(drop=True)
    return df

def train_and_forecast(district_df):
    district_df = district_df.sort_values("date").reset_index(drop=True)

    # ---- Linear trend (interpretable baseline) ----
    t = np.arange(len(district_df)).reshape(-1, 1)
    y = district_df["mean_depth"].values
    lin = LinearRegression().fit(t, y)
    slope_per_quarter = lin.coef_[0]
    slope_per_year = slope_per_quarter * 4  # ~4 quarters/year

    # ---- Random Forest on lag features ----
    feat_df = make_lag_features(district_df)
    feature_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)] + ["quarter_of_year"]
    X = feat_df[feature_cols]
    y_rf = feat_df["mean_depth"]

    # simple holdout validation: last 8 quarters as test
    split = max(len(X) - 8, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_rf.iloc[:split], y_rf.iloc[split:]

    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    mae = None
    if len(X_test) > 0:
        preds_test = rf.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds_test))

    # refit on ALL data for the actual forward forecast
    rf_full = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    rf_full.fit(X, y_rf)

    # recursive forecast forward
    history = list(district_df["mean_depth"].values[-N_LAGS:])
    last_date = district_df["date"].max()
    forecast_rows = []
    cur_date = last_date
    for step in range(1, FORECAST_QUARTERS + 1):
        cur_date = cur_date + pd.DateOffset(months=4)  # approx next quarter (3 surveys/yr->our resample is quarterly)
        lags = history[-N_LAGS:]
        x_input = pd.DataFrame([{
            **{f"lag_{i+1}": lags[-(i+1)] for i in range(N_LAGS)},
            "quarter_of_year": cur_date.quarter
        }])[feature_cols]
        pred = float(rf_full.predict(x_input)[0])
        history.append(pred)
        forecast_rows.append({"date": cur_date.strftime("%Y-%m-%d"), "depth_m": round(pred, 2)})

    # risk level from linear slope (m/year)
    if slope_per_year >= 0.30:
        risk = "High"
    elif slope_per_year >= 0.10:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "history": [
            {"date": d.strftime("%Y-%m-%d"), "depth_m": round(v, 2)}
            for d, v in zip(district_df["date"], district_df["mean_depth"])
        ],
        "forecast": forecast_rows,
        "slope_m_per_year": round(float(slope_per_year), 3),
        "risk_level": risk,
        "last_observed_depth_m": round(float(district_df["mean_depth"].iloc[-1]), 2),
        "well_count": int(district_df["well_count"].iloc[-1]),
        "rf_holdout_mae_m": round(mae, 3) if mae is not None else None,
    }

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"])
    results = {}
    for district, grp in df.groupby("DISTRICT"):
        print(f"Training model for {district}...")
        results[district] = train_and_forecast(grp)
        r = results[district]
        print(f"  slope={r['slope_m_per_year']} m/yr | risk={r['risk_level']} | "
              f"holdout MAE={r['rf_holdout_mae_m']} m | last depth={r['last_observed_depth_m']} m")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved forecast results: {OUT_PATH}")

if __name__ == "__main__":
    main()
