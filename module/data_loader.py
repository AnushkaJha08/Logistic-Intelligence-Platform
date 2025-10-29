import pandas as pd
import os

def _read_csv_safe(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return df

def load_all_data(data_dir="data"):
    files = {
        "cost_breakdown": "cost_breakdown.csv",
        "customer_feedback": "customer_feedback.csv",
        "delivery_performance": "delivery_performance.csv",
        "orders": "orders.csv",
        "routes_distance": "routes_distance.csv",
        "vehicle_fleet": "vehicle_fleet.csv",
        "warehouse_inventory": "warehouse_inventory.csv"
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        data[key] = _read_csv_safe(path)
    return data
