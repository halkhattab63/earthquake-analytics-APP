# ann_predictor/predict_ann.py

import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')
# إعداد المسارات
MODEL_PATH = "ann_predictor/outputs/model_ann.h5"
SCALER_PATH = "earthquake_risk_predictor/data/scaler.pkl"

# تحميل النموذج والسكالر
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"❌ Scaler not found at {SCALER_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# التسميات للفئات
LABELS = ["Hafif", "Orta", "Güçlü", "Şiddetli", "Felaket"]

# الأعمدة المطلوبة
FEATURES = [
    "depth_km", "magnitude", "latitude", "longitude",
    "month", "dayofweek", "year", "is_night",
    "energy", "severity_ratio", "dist_fault"
]

def predict(input_df: pd.DataFrame) -> pd.DataFrame:
    if not all(col in input_df.columns for col in FEATURES):
        raise ValueError(f"Input DataFrame must contain the following columns: {FEATURES}")

    X_scaled = scaler.transform(input_df[FEATURES])
    y_pred_class_logits, y_pred_reg = model.predict(X_scaled)

    class_indices = y_pred_class_logits.argmax(axis=1)
    class_labels = [LABELS[i] for i in class_indices]

    result_df = input_df.copy()
    result_df["Predicted_Class"] = class_labels
    result_df["Pred_Magnitude"] = y_pred_reg[:, 0]
    result_df["Pred_Depth_km"] = y_pred_reg[:, 1]

    return result_df

# مثال مباشر
# if __name__ == "__main__":
#     sample = pd.DataFrame([{
#         "depth_km": 10.0,
#         "magnitude": 5.8,
#         "latitude": 38.0,
#         "longitude": 39.5,
#         "month": 5,
#         "dayofweek": 2,
#         "year": 2024,
#         "hour": 2,
#         "is_night": 1,
#         "energy": 10 ** (1.5 * 5.8 + 4.8),
#         "severity_ratio": 5.8 / (10.0 + 1),
#         "dist_fault": 75.3
#     }])

    # result = predict(sample)
    # print("\n📊 Prediction Result:")
    # print(result)

    # # (اختياري) حفظ التنبؤات
    # result.to_csv("earthquake_risk_predictor/scr/ann_predictor/outputs/prediction_sample.csv", index=False)
    # print("✅ Saved to earthquake_risk_predictor/scr/ann_predictor/outputs/prediction_sample.csv")
    # print("📦 [Keras] Outputs saved in earthquake_risk_predictor/scr/ann_predictor/outputs/")