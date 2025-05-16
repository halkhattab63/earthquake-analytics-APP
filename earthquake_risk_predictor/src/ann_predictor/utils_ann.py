# ann_predictor/utils_ann.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split


def load_data(
    data_path="earthquake_risk_predictor/data/processed_earthquakes.csv",
    scaler_path="earthquake_risk_predictor/data/scaler.pkl",
    test_size=0.2,
    random_state=42
):
    df = pd.read_csv(data_path)

    features = [
        "depth_km", "magnitude", "latitude", "longitude",
        "month", "dayofweek", "year", "is_night",
        "energy", "severity_ratio", "dist_fault"
    ]

    X = df[features]
    y_class = df["severity_class"]
    y_reg = df[["magnitude", "depth_km"]]

    # Load pre-fitted scaler
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # Convert classification labels to one-hot encoding
    y_class_cat = pd.get_dummies(y_class).values

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_class_cat, y_reg.values,
        test_size=test_size, stratify=y_class, random_state=random_state
    )

    return (X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test)
