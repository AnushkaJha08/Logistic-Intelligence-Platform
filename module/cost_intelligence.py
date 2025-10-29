import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def detect_cost_anomalies(df):
    """
    Flag cost anomalies where cost_per_km > mean + 2*std
    """
    out = df.copy()
    if "Cost_per_KM" not in out.columns:
        out["Cost_per_KM"] = out["Total_Cost_INR"] / out["Distance_KM"].replace({0:np.nan})
    mean = out["Cost_per_KM"].mean(skipna=True)
    std = out["Cost_per_KM"].std(skipna=True)
    out["Cost_Anomaly"] = out["Cost_per_KM"] > (mean + 2*std)
    return out

def train_cost_model(df):
    """
    Predict Total_Cost_INR using route and order features.
    Returns (model, X_test, y_test)
    """
    required = ["Total_Cost_INR","Distance_KM","Order_Value_INR"]
    if not set(required).issubset(df.columns):
        return None, pd.DataFrame(), pd.Series()
    df2 = df.dropna(subset=required)
    features = ["Distance_KM","Order_Value_INR"]
    X = df2[features].fillna(0)
    y = df2["Total_Cost_INR"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def cost_feature_importance(model):
    import pandas as pd
    if model is None:
        return pd.DataFrame()
    try:
        feats = model.feature_importances_
        names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"f{i}" for i in range(len(feats))]
        return pd.DataFrame({"feature": names, "importance": feats}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame()
