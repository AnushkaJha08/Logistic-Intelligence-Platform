import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud

def _download_button_df(df, prefix):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Download {prefix} (CSV)", data=csv, file_name=f"{prefix}.csv", mime="text/csv")

def show_kpi_summary(kpis: dict):
    st.header("Summary KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Delay (days)", f"{kpis.get('avg_delay_days',0):.2f}")
    col2.metric("On-time Rate (%)", f"{kpis.get('on_time_rate_pct',0):.2f}")
    col3.metric("Avg Cost / Order (INR)", f"{kpis.get('avg_cost_per_order',0):.2f}")
    col4.metric("Avg Customer Rating", f"{kpis.get('avg_customer_rating',0):.2f}")

def show_delivery_performance(df):
    st.subheader("Delivery Performance")
    if "Delivery_Delay_Days" in df.columns:
        fig = px.histogram(df, x="Delivery_Delay_Days", nbins=30, title="Delivery Delay Distribution")
        st.plotly_chart(fig, use_container_width=True)
        _download_button_df(df[["Order_ID","Delivery_Delay_Days"]], "delivery_delays")

def show_route_efficiency(routes_df):
    st.subheader("Route Efficiency")
    df = routes_df.copy()
    if {"Distance_KM","Fuel_Consumption_L"}.issubset(df.columns):
        df["Efficiency_Score"] = df["Distance_KM"] / df["Fuel_Consumption_L"]
        fig = px.scatter(df, x="Distance_KM", y="Fuel_Consumption_L", size="Efficiency_Score", color="Efficiency_Score", title="Fuel Efficiency by Route")
        st.plotly_chart(fig, use_container_width=True)
        _download_button_df(df, "routes_efficiency")

def show_vehicle_status(vehicle_df):
    st.subheader("Fleet Overview")
    if "Status" in vehicle_df.columns:
        counts = vehicle_df["Status"].value_counts().reset_index()
        counts.columns = ["Status","Count"]
        fig = px.pie(counts, names="Status", values="Count", title="Vehicle Status")
        st.plotly_chart(fig, use_container_width=True)
        _download_button_df(vehicle_df, "vehicle_fleet")

def show_cost_breakdown(cost_df):
    st.subheader("Cost Breakdown")
    df = cost_df.copy()
    # melt costs into categories if columns exist
    cost_cols = [c for c in ["Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance","Packaging_Cost","Technology_Platform_Fee","Other_Overhead"] if c in df.columns]
    if cost_cols:
        melt = df.melt(id_vars=["Order_ID"], value_vars=cost_cols, var_name="Cost_Type", value_name="Amount")
        summary = melt.groupby("Cost_Type")["Amount"].mean().reset_index()
        fig = px.bar(summary, x="Cost_Type", y="Amount", title="Avg Cost by Category")
        st.plotly_chart(fig, use_container_width=True)
        _download_button_df(summary, "cost_breakdown_summary")
    else:
        st.info("Cost columns not found.")

def show_customer_feedback(feedback_df):
    st.subheader("Customer Feedback")
    if "Rating" in feedback_df.columns:
        fig = px.histogram(feedback_df, x="Rating", nbins=5, title="Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)
    if "Feedback_Text" in feedback_df.columns:
        text = " ".join(feedback_df["Feedback_Text"].dropna().tolist())
        if text.strip():
            wc = WordCloud(width=800, height=300, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    _download_button_df(feedback_df, "customer_feedback")

def show_warehouse_status(warehouse_df):
    st.subheader("Warehouse Inventory")
    if {"Warehouse_ID","Current_Stock_Units","Reorder_Level"}.issubset(warehouse_df.columns):
        fig = px.bar(warehouse_df, x="Warehouse_ID", y=["Current_Stock_Units","Reorder_Level"], barmode="group", title="Stock vs Reorder Level")
        st.plotly_chart(fig, use_container_width=True)
    _download_button_df(warehouse_df, "warehouse_inventory")

def show_route_risk_scatter(route_risk_df):
    st.subheader("Route Risk Scatter")
    if route_risk_df.empty:
        st.info("No route risk data available.")
        return
    fig = px.scatter(route_risk_df, x="avg_costpkm", y="avg_delay", size="count_orders", color="Route_Risk",
                     hover_data=["Route"], title="Route Risk: cost vs delay (size=orders)")
    st.plotly_chart(fig, use_container_width=True)

def show_cost_model_insights(model, X_test, y_test):
    st.subheader("Cost Model Insights")
    if model is None:
        st.info("Cost model not available (insufficient data).")
        return
    preds = model.predict(X_test)
    df = X_test.copy()
    df["Actual_Cost"] = y_test.values
    df["Predicted_Cost"] = preds
    st.write("Sample predictions (Actual vs Predicted):")
    st.dataframe(df.head(10))
    # simple scatter
    fig = px.scatter(df, x="Actual_Cost", y="Predicted_Cost", title="Actual vs Predicted Cost")
    st.plotly_chart(fig, use_container_width=True)
    _download_button_df(df, "cost_model_predictions")
