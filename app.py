# app.py
import streamlit as st
import pandas as pd
from generate_sample_sales import generate_daily_sales
import sales_anomoly as sa
import plotly.io as pio

st.set_page_config(layout="centered", page_title="Retail Anomaly Demo")

st.title("Retail Sales Anomaly Detection")

# sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Generate sample_sales.csv (365 days)"):
    df = generate_daily_sales(start_date="2024-01-01", days=365, base=120)
    df.to_csv("sample_sales.csv", index=False)
    st.sidebar.success("sample_sales.csv generated")

uploaded = st.file_uploader("Or upload sample_sales.csv", type=["csv"])
if uploaded is not None:
    uploaded.save = lambda path: open(path,"wb").write(uploaded.getvalue())
    uploaded.save("sample_sales.csv")
    st.success("Uploaded sample_sales.csv")

if st.sidebar.button("Run detection"):
    df = sa.load_data("sample_sales.csv")
    df = sa.feature_engineer(df)
    df = sa.get_residuals(df, period=st.sidebar.number_input("seasonal period", value=7, min_value=2, max_value=30))
    df = sa.iqr_flags(df, multiplier=st.sidebar.slider("IQR multiplier", 0.5, 3.0, 1.5))
    df = sa.isolation_forest_flags(df, contamination=st.sidebar.slider("IF contamination", 0.0, 0.2, 0.02))
    df = sa.aggregate_flags(df)
    st.write("Total rows:", len(df), " | Anomalies:", int(df["anomaly"].sum()))
    st.dataframe(df[df["anomaly"]][["date","sales","residual","flag_reason"]].reset_index(drop=True))
    fig = sa.make_plot(df)
    st.plotly_chart(fig, use_container_width=True)
    df[df["anomaly"]][["date","store_id","sku_id","sales","residual","flag_reason"]].to_csv("flagged_anomalies.csv", index=False)
    st.success("Saved flagged_anomalies.csv")

st.info("Use sidebar to generate data, tune sensitivity, and run detection.")
