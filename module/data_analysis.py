import pandas as pd
import numpy as np

def prepare_metrics(data_dict):
    """
    Merge orders + delivery + cost + routes + feedback into a single dataframe.
    Uses your column names: Order_ID, Distance_KM, etc.
    """
    orders = data_dict.get("orders", pd.DataFrame()).copy()
    perf = data_dict.get("delivery_performance", pd.DataFrame()).copy()
    cost = data_dict.get("cost_breakdown", pd.DataFrame()).copy()
    routes = data_dict.get("routes_distance", pd.DataFrame()).copy()
    feedback = data_dict.get("customer_feedback", pd.DataFrame()).copy()
    fleet = data_dict.get("vehicle_fleet", pd.DataFrame()).copy()
    warehouse = data_dict.get("warehouse_inventory", pd.DataFrame()).copy()

    # Merge left on Orders
    df = orders.merge(perf, on="Order_ID", how="left", suffixes=("","_perf"))
    df = df.merge(cost, on="Order_ID", how="left", suffixes=("","_cost"))
    df = df.merge(routes, on="Order_ID", how="left", suffixes=("","_route"))
    df = df.merge(feedback[["Order_ID","Rating","Issue_Category"]], on="Order_ID", how="left")

    # Derived fields
    # Delivery delay days
    if {"Promised_Delivery_Days","Actual_Delivery_Days"}.issubset(df.columns):
        df["Delivery_Delay_Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    else:
        df["Delivery_Delay_Days"] = np.nan

    # Total cost INR from cost breakdown columns
    cost_cols = [c for c in ["Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance","Packaging_Cost","Technology_Platform_Fee","Other_Overhead"] if c in df.columns]
    if cost_cols:
        df["Total_Cost_INR"] = df[cost_cols].sum(axis=1)
    else:
        df["Total_Cost_INR"] = df.get("Delivery_Cost_INR", np.nan)

    # Cost per km (safe)
    df["Cost_per_KM"] = df.apply(lambda r: (r["Total_Cost_INR"] / r["Distance_KM"]) if pd.notnull(r.get("Distance_KM")) and r.get("Distance_KM")>0 else np.nan, axis=1)

    # Delay flag
    df["Is_Delayed"] = df["Delivery_Delay_Days"].apply(lambda x: 1 if pd.notnull(x) and x > 0 else 0)

    # Merge fleet info if Vehicle_ID available
    if "Vehicle_ID" in df.columns and not fleet.empty:
        df = df.merge(fleet, on="Vehicle_ID", how="left")

    # Return enriched dataframe
    return df

def compute_kpis(df):
    kpis = {}
    kpis["avg_delay_days"] = df["Delivery_Delay_Days"].mean(skipna=True)
    kpis["on_time_rate_pct"] = (1 - df["Is_Delayed"].mean(skipna=True)) * 100 if "Is_Delayed" in df else None
    kpis["avg_cost_per_order"] = df["Total_Cost_INR"].mean(skipna=True)
    kpis["avg_cost_per_km"] = df["Cost_per_KM"].mean(skipna=True)
    kpis["avg_customer_rating"] = df["Rating"].mean(skipna=True) if "Rating" in df else None
    return kpis
