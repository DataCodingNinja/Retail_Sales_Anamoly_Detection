# sales_anomaly.py
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.offline as pyo

PERIOD = 7  # default weekly period

def load_data(path="sample_sales.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date").asfreq("D").reset_index()
    df["store_id"] = df["store_id"].ffill().bfill()
    df["sku_id"] = df["sku_id"].ffill().bfill()
    df["sales"] = df["sales"].fillna(method="ffill").fillna(0)
    return df

def feature_engineer(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["rolling_mean_7"] = df["sales"].rolling(7, min_periods=1, center=True).mean()
    df["rolling_std_7"] = df["sales"].rolling(7, min_periods=1, center=True).std().fillna(0)
    return df

def get_residuals(df, period=PERIOD):
    df = df.copy()
    series = df["sales"].interpolate().fillna(method="bfill").fillna(method="ffill")
    res = seasonal_decompose(series, period=period, model="additive", extrapolate_trend='freq')
    df["trend"] = res.trend
    df["seasonal"] = res.seasonal
    df["residual"] = (df["sales"] - df["trend"] - df["seasonal"]).fillna(0)
    return df

def iqr_flags(df, multiplier=1.5):
    df = df.copy()
    q1 = df["residual"].quantile(0.25)
    q3 = df["residual"].quantile(0.75)
    iqr = q3 - q1
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    df["anomaly_iqr"] = ((df["residual"] < low) | (df["residual"] > high))
    df["iqr_low"] = low
    df["iqr_high"] = high
    return df

def isolation_forest_flags(df, contamination=0.02):
    df = df.copy()
    feat_cols = ["sales", "rolling_mean_7", "rolling_std_7", "is_weekend"]
    X = df[feat_cols].fillna(0)
    iso = IsolationForest(n_estimators=50, max_samples=0.5, contamination=contamination, random_state=42)
    df["anomaly_iforest"] = iso.fit_predict(X) == -1
    return df

def aggregate_flags(df):
    df = df.copy()
    df["anomaly"] = df[["anomaly_iqr", "anomaly_iforest"]].any(axis=1)
    def reason(r):
        if r["anomaly_iqr"] and r["anomaly_iforest"]:
            return "both"
        if r["anomaly_iqr"]:
            return "iqr"
        if r["anomaly_iforest"]:
            return "iforest"
        return ""
    df["flag_reason"] = df.apply(reason, axis=1)
    return df

def make_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sales"], mode="lines", name="sales"))
    anomalies = df[df["anomaly"]]
    fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["sales"], mode="markers",
                             marker=dict(color="red", size=8), name="anomalies"))
    fig.update_layout(title="Daily Sales with Anomalies", xaxis_title="date", yaxis_title="sales")
    return fig
