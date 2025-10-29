def estimate_co2_per_order(row):
    """
    Estimate CO2 kg for an order.
    Prefer fuel consumption if available, else use a rough factor per km by vehicle type.
    """
    try:
        if "Fuel_Consumption_L" in row and not pd.isna(row["Fuel_Consumption_L"]):
            # approx kg CO2 per liter diesel ~ 2.31
            return float(row["Fuel_Consumption_L"]) * 2.31
        # fallback: use per-km estimate based on vehicle type if available
        if "Vehicle_Type" in row and "Distance_KM" in row:
            factor = {"Van":0.2, "Truck":0.6, "Bike":0.05}
            return float(row["Distance_KM"]) * factor.get(row.get("Vehicle_Type","Van"), 0.2)
    except Exception:
        pass
    return 0.0
