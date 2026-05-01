# generate_sample_sales.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_daily_sales(start_date="2024-01-01", days=365, base=100):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    # weekly seasonality: higher on weekends
    weekday = dates.weekday
    weekly_effect = np.where(weekday >= 5, 1.2, 1.0)  # weekend +20%
    # monthly seasonality (simple)
    month = dates.month
    monthly_effect = 1 + 0.1 * np.sin((month - 1) / 12 * 2 * np.pi)
    # trend (small upward)
    trend = 1 + np.linspace(0, 0.05, days)
    # random noise
    noise = np.random.normal(0, 5, size=days)
    sales = base * weekly_effect * monthly_effect * trend + noise
    sales = np.round(np.clip(sales, 0, None), 2)
    df = pd.DataFrame({
        "date": dates,
        "store_id": "S1",
        "sku_id": "SKU-001",
        "sales": sales
    })
    # inject some anomalies (spikes and drops)
    anomaly_indices = [30, 90, 150, 220, 300]
    df.loc[anomaly_indices, "sales"] *= [0.2, 3.0, 0.1, 4.0, 0.3]  # drops and spikes
    return df

if __name__ == "__main__":
    df = generate_daily_sales(start_date="2024-01-01", days=365, base=120)
    df.to_csv("sample_sales.csv", index=False)
    print("sample_sales.csv generated (365 rows).")
