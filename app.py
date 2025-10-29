import streamlit as st
import pandas as pd

from modules.data_loader import load_all_data
from modules.data_analysis import prepare_metrics, compute_kpis
from modules.delay_predictor import train_delay_model, train_delay_classifier, delay_feature_importance
from modules.cost_intelligence import train_cost_model, cost_feature_importance, detect_cost_anomalies
from modules.optimization import compute_route_risk, recommend_alternatives
from modules.visualization import (
    show_kpi_summary,
    show_delivery_performance,
    show_route_efficiency,
    show_vehicle_status,
    show_cost_breakdown,
    show_customer_feedback,
    show_warehouse_status,
    show_route_risk_scatter,
    show_cost_model_insights
)
from modules.utils import estimate_co2_per_order

st.set_page_config(page_title="Predictive Delivery & Cost Intelligence", layout="wide")
st.title("🚚 NexGen — Predictive Delivery & Cost Intelligence")

# Load data
data = load_all_data(data_dir="data")

# Basic validation
if data["orders"].empty:
    st.error("Orders data not loaded or empty. Check data files in /data.")
    st.stop()

# Prepare and merge data
metrics = prepare_metrics(data)  # returns merged dataframe with derived fields

# Sidebar filters
st.sidebar.header("Filters")
date_min = pd.to_datetime(metrics["Order_Date"]).min()
date_max = pd.to_datetime(metrics["Order_Date"]).max()
date_range = st.sidebar.date_input("Order Date Range", value=(date_min.date(), date_max.date()))
region_choices = ["All"] + sorted(metrics["Origin"].dropna().unique().tolist())
selected_origin = st.sidebar.selectbox("Origin (or All)", options=region_choices, index=0)
vehicle_choices = ["All"] + sorted(data["vehicle_fleet"]["Vehicle_Type"].dropna().unique().tolist())
selected_vehicle = st.sidebar.selectbox("Vehicle Type (or All)", options=vehicle_choices, index=0)

# Apply filters
start_date, end_date = date_range
metrics["Order_Date"] = pd.to_datetime(metrics["Order_Date"], errors="coerce")
filtered = metrics[
    (metrics["Order_Date"].dt.date >= pd.to_datetime(start_date).date()) &
    (metrics["Order_Date"].dt.date <= pd.to_datetime(end_date).date())
].copy()
if selected_origin != "All":
    filtered = filtered[filtered["Origin"] == selected_origin]
if selected_vehicle != "All":
    # vehicle matching via joined Vehicle_Type if present
    if "Vehicle_Type" in filtered.columns:
        filtered = filtered[filtered["Vehicle_Type"] == selected_vehicle]

# KPI summary
kpis = compute_kpis(filtered)
show_kpi_summary(kpis)

# Left column: main analytics
st.markdown("## Operational Insights")
col1, col2 = st.columns(2)
with col1:
    show_delivery_performance(filtered)
    show_cost_breakdown(filtered[[
        "Order_ID",
        "Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance","Packaging_Cost",
        "Technology_Platform_Fee","Other_Overhead","Total_Cost_INR","Cost_per_KM"
    ]].drop_duplicates(subset=["Order_ID"]))
with col2:
    show_route_efficiency(data["routes_distance"])
    show_vehicle_status(data["vehicle_fleet"])

# Route Risk and Optimization
st.markdown("## Route Risk & Recommendations")
route_risk_df = compute_route_risk(filtered)
show_route_risk_scatter(route_risk_df)
st.write("Top risky routes (highest composite risk):")
st.dataframe(route_risk_df.sort_values("Route_Risk", ascending=False).head(10))
alt_recs = recommend_alternatives(route_risk_df, data["vehicle_fleet"])
st.write("Suggested lower-risk route/vehicle alternatives (heuristic):")
st.dataframe(alt_recs.head(10))

# Cost Model (predictive)
st.markdown("## Cost Model & Insights")
cost_model, X_test_cost, y_test_cost = train_cost_model(filtered)
show_cost_model_insights(cost_model, X_test_cost, y_test_cost)
st.write("Top cost feature importances:")
st.dataframe(cost_feature_importance(cost_model))

# Delay predictor (regression) and classifier
st.markdown("## Delay Predictor")
delay_model = train_delay_model(filtered)
st.write("Delay regression model trained on historical data (simple linear/regression).")
clf, X_test_clf, y_test_clf = train_delay_classifier(filtered)
st.write("Delay classification model (Delayed vs On-time).")
st.dataframe(delay_feature_importance(clf))

# Sustainability
st.markdown("## Sustainability Insights")
if {"Fuel_Cost","Fuel_Consumption_L","Distance_KM"}.issubset(filtered.columns):
    filtered["CO2_kg_est"] = filtered.apply(estimate_co2_per_order, axis=1)
    st.metric("Estimated Total CO2 (kg)", f"{filtered['CO2_kg_est'].sum():,.0f}")
    st.write("Per-order CO2 sample:")
    st.dataframe(filtered[["Order_ID","Distance_KM","Fuel_Consumption_L","CO2_kg_est"]].head())

# Customer feedback and warehouse
st.markdown("## Customer & Warehouse")
show_customer_feedback(data["customer_feedback"])
show_warehouse_status(data["warehouse_inventory"])

st.markdown("---")
st.write("Download processed metrics or models from individual visualizations where available.")
