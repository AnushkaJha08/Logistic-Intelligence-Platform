import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

def train_delay_model(df):
    """
    Regression model predicting delay days (numeric). Returns trained regressor and test sets.
    """
    df2 = df.dropna(subset=["Delivery_Delay_Days","Promised_Delivery_Days"])
    if df2.empty:
        return None
    X = df2[["Promised_Delivery_Days","Distance_KM"]].fillna(0)
    y = df2["Delivery_Delay_Days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    return reg

def train_delay_classifier(df):
    """
    Classifier: Delayed (1) vs On-time (0).
    Returns (clf, X_test, y_test)
    """
    df2 = df.dropna(subset=["Is_Delayed","Distance_KM","Total_Cost_INR"])
    if df2.empty:
        return None, pd.DataFrame(), pd.Series()
    features = ["Distance_KM","Total_Cost_INR","Traffic_Delay_Minutes"] if "Traffic_Delay_Minutes" in df2.columns else ["Distance_KM","Total_Cost_INR"]
    X = df2[features].fillna(0)
    y = df2["Is_Delayed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def delay_feature_importance(clf):
    if clf is None:
        return pd.DataFrame()
    try:
        import pandas as pd
        feats = clf.feature_importances_
        names = clf.feature_names_in_ if hasattr(clf, "feature_names_in_") else [f"f{i}" for i in range(len(feats))]
        return pd.DataFrame({"feature": names, "importance": feats}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame()
