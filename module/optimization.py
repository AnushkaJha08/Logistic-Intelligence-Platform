import pandas as pd
import numpy as np

def compute_route_risk(df):
    """
    Composite Route Risk = normalized delay factor + cost factor + traffic factor
    Returns dataframe with Route and Route_Risk score
    """
    df2 = df.copy()
    # require Route and metrics
    df2 = df2.dropna(subset=["Route","Cost_per_KM","Delivery_Delay_Days","Traffic_Delay_Minutes"])
    if df2.empty:
        return pd.DataFrame(columns=["Route","Route_Risk"])
    grp = df2.groupby("Route").agg(
        avg_delay=("Delivery_Delay_Days","mean"),
        avg_costpkm=("Cost_per_KM","mean"),
        avg_traffic=("Traffic_Delay_Minutes","mean"),
        count_orders=("Order_ID","count")
    ).reset_index()
    # normalize columns
    for c in ["avg_delay","avg_costpkm","avg_traffic"]:
        v = grp[c]
        grp[c + "_n"] = (v - v.min()) / (v.max() - v.min() + 1e-9)
    # composite score (weights can be tuned)
    grp["Route_Risk"] = 0.5*grp["avg_delay_n"] + 0.3*grp["avg_costpkm_n"] + 0.2*grp["avg_traffic_n"]
    return grp[["Route","Route_Risk","avg_delay","avg_costpkm","avg_traffic","count_orders"]]

def recommend_alternatives(route_risk_df, vehicle_df):
    """
    Heuristic: For high-risk routes, recommend top active vehicles sorted by fuel efficiency
    """
    v = vehicle_df.copy()
    if v.empty:
        return pd.DataFrame()
    v_active = v[v["Status"].str.lower()=="active"].copy()
    if v_active.empty:
        v_active = v.copy()
    v_active = v_active.sort_values("Fuel_Efficiency_KM_per_L", ascending=False)
    # Pair each route with top 3 vehicles (simple)
    recs = []
    for _, r in route_risk_df.sort_values("Route_Risk", ascending=False).head(20).iterrows():
        top_vehicles = v_active.head(3)
        for _, veh in top_vehicles.iterrows():
            recs.append({
                "Route": r["Route"],
                "Route_Risk": r["Route_Risk"],
                "Recommended_Vehicle_ID": veh.get("Vehicle_ID"),
                "Vehicle_Type": veh.get("Vehicle_Type"),
                "Vehicle_Fuel_Eff_kmpl": veh.get("Fuel_Efficiency_KM_per_L")
            })
    return pd.DataFrame(recs)
