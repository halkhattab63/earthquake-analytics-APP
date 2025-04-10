import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.model import EarthquakeModel

reg_scaler = StandardScaler()

def train_model(model, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg, save_path='model.pt'):


    global reg_scaler
    reg_scaler = StandardScaler()
    y_train_reg_scaled = reg_scaler.fit_transform(y_train_reg)
    y_test_reg_scaled = reg_scaler.transform(y_test_reg)
    joblib.dump(reg_scaler, "data/reg_scaler.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_class_tensor = torch.tensor(y_train_class.values, dtype=torch.long).to(device)
    y_train_reg_tensor = torch.tensor(y_train_reg_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out_class, out_reg = model(X_train_tensor)

        loss_class = criterion_class(out_class, y_train_class_tensor)

        try:
            loss_reg = criterion_reg(out_reg, y_train_reg_tensor)
            if torch.isnan(loss_reg) or torch.isinf(loss_reg):
                print(f"⚠️ Epoch {epoch+1}: Regression loss is unstable. Skipping regression.")
                loss = loss_class
            else:
                loss = loss_class + 2.0 * loss_reg
        except Exception as e:
            print(f"⚠️ Regression loss error: {e}")
            loss = loss_class

        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to '{save_path}'")

    model.eval()
    with torch.no_grad():
        preds_class, preds_reg = model(X_test_tensor)
        preds_class_labels = torch.argmax(preds_class, axis=1).cpu().numpy()

        try:
            preds_reg_np = preds_reg.cpu().numpy()
            preds_reg_inv = reg_scaler.inverse_transform(preds_reg_np)
        except Exception as e:
            print(f"⚠️ Regression prediction scaling failed: {e}")
            preds_reg_inv = np.zeros_like(y_test_reg)

        print("\n📊 Classification Report:")
        print(classification_report(
            y_test_class,
            preds_class_labels,
            target_names=['Hafif', 'Orta', 'Güçlü', 'Şiddetli', 'Felaket']
        ))

        try:
            mse = mean_squared_error(y_test_reg, preds_reg_inv)
            print("\n📉 Regression MSE:")
            print(mse)
            
            # 📊 Confusion Matrix
            cm = confusion_matrix(y_test_class, preds_class_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Hafif', 'Orta', 'Güçlü', 'Şiddetli', 'Felaket'])
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title("Confusion Matrix")
            plt.tight_layout()
            os.makedirs("data", exist_ok=True)
            plt.savefig("data/confusion_matrix.png")
            print("📊 Saved confusion matrix to 'data/confusion_matrix.png'")
            
            errors = np.abs(y_test_reg.values - preds_reg_inv)
            plt.figure(figsize=(10, 4))
            plt.hist(errors[:, 0], bins=30, alpha=0.6, label='Magnitude Error')
            plt.hist(errors[:, 1], bins=30, alpha=0.6, label='Depth Error')
            plt.title("Regression Error Distribution")
            plt.legend()
            plt.tight_layout()
            os.makedirs("data", exist_ok=True)
            plt.savefig("data/regression_errors.png")
            print("📉 Saved regression error distribution to 'data/regression_errors.png'")
        except Exception as e:
            print(f"⚠️ Error calculating regression metrics: {e}")


def predict_and_save_outputs(model, X_test, y_test_class, y_test_reg, output_path="earthquake_risk_predictor/data/predictions.csv"):
    global reg_scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_class, preds_reg = model(X_test_tensor)
        class_labels = torch.argmax(preds_class, axis=1).cpu().numpy()
        preds_reg = reg_scaler.inverse_transform(preds_reg.cpu().numpy())

    df_out = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(X_test.shape[1])])
    df_out["true_class"] = y_test_class.values
    df_out["pred_class"] = class_labels
    df_out["true_magnitude"] = y_test_reg.iloc[:, 0].values
    df_out["pred_magnitude"] = preds_reg[:, 0]
    df_out["true_depth_km"] = y_test_reg.iloc[:, 1].values
    df_out["pred_depth_km"] = preds_reg[:, 1]

    df_out.to_csv(output_path, index=False)
    print(f"📄 Predictions saved to {output_path}")


def run_xgboost_regressor(X_train, X_test, y_train_reg, y_test_reg):
    os.makedirs("data", exist_ok=True)
    results = {}

    for i, label in enumerate(["magnitude", "depth_km"]):
        model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train_reg.iloc[:, i])
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test_reg.iloc[:, i], preds)
        r2 = r2_score(y_test_reg.iloc[:, i], preds)
        mae = mean_absolute_error(y_test_reg.iloc[:, i], preds)

        results[label] = {"MSE": mse, "MAE": mae, "R2": r2}
        plt.figure()
        plt.scatter(y_test_reg.iloc[:, i], preds, alpha=0.5)
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"XGBoost Prediction: {label}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"earthquake_risk_predictor/data/xgboost_results_{label}.png")

    pd.DataFrame(results).T.to_csv("earthquake_risk_predictor/data/xgboost_metrics.csv")
    print("📊 XGBoost metrics saved to 'data/xgboost_metrics.csv'")


import geopandas as gpd
import matplotlib.pyplot as plt

def plot_geographic_distribution(df, lat_col="latitude", lon_col="longitude", border_file="tr.json"):
    """
    Plots earthquakes on a Turkey map using GeoJSON borders.
    """
    # تحويل DataFrame إلى GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

    # تحميل حدود تركيا
    if os.path.exists(border_file):
        turkey = gpd.read_file(border_file)
    else:
        print("⚠️ GeoJSON sınır dosyası bulunamadı.")
        turkey = gpd.GeoDataFrame()

    # رسم الخريطة
    fig, ax = plt.subplots(figsize=(10, 8))
    if not turkey.empty:
        turkey.boundary.plot(ax=ax, color="black", linewidth=0.8)

    gdf.plot(
        ax=ax,
        column='magnitude',
        cmap='Reds',
        markersize=30,
        legend=True,
        alpha=0.6
    )

    plt.title("Earthquake Geographic Distribution (Turkey)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("earthquake_risk_predictor/data/geographic_distribution.png")
    print("🗺️ Saved geographic map with borders to 'data/geographic_distribution.png'")