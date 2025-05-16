import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# joblib: لحفظ الكائنات مثل StandardScaler.
# torch.*: لبناء وتدريب الشبكة العصبية.
# pandas, numpy: لمعالجة البيانات.
# matplotlib: للرسوم البيانية.
# os: لإدارة الملفات والمجلدات.
# من sklearn.metrics: لتقييم النموذج.
# من sklearn.preprocessing: لتحجيم البيانات.
import geopandas as gpd
import matplotlib.pyplot as plt


from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
# أدوات تقييم للنماذج التصنيفية والانحدارية.
# XGBRegressor: نموذج انحدار من مكتبة XGBoost.


from earthquake_risk_predictor.src.PyTorch_predictor.model import EarthquakeModel
# استيراد النموذج العصبي.


# هذا السكالر يستخدم لاحقًا لتوحيد y_train_reg ثم يُحفظ.
reg_scaler = StandardScaler()



#  دالة تدريب النموذج
def train_model(model, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg, save_path='model.pt'):

# توحيد أهداف الانحدار
# توحيد القيم في y_train_reg (magnitude و depth).
#     يتم حفظ السكالر لاستخدامه لاحقًا في inference.
#     1. توحيد القيم باستخدام StandardScaler.
#     2. حفظ السكالر في ملف pickle لاستخدامه لاحقًا.
    global reg_scaler
    reg_scaler = StandardScaler()
    y_train_reg_scaled = reg_scaler.fit_transform(y_train_reg)
    y_test_reg_scaled = reg_scaler.transform(y_test_reg)
    joblib.dump(reg_scaler, "earthquake_risk_predictor/data/reg_scaler.pkl")

# نقل النموذج إلى الجهاز المناسب (CPU أو GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# حويل البيانات إلى Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_class_tensor = torch.tensor(y_train_class.values, dtype=torch.long).to(device)
    y_train_reg_tensor = torch.tensor(y_train_reg_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# إعداد التدريب
    # خوارزمية تدريب: Adam
    # دوال الخسارة: تصنيف وانحدار.
    # إعداد خوارزمية التدريب
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    
#  تدريب 100 حقبة (epoch)
    model.train()
    for epoch in range(250):
        
        # نموذج في وضع التدريب.
        # تصفير التدرج.
        # تنفيذ forward pass.
        optimizer.zero_grad()
        out_class, out_reg = model(X_train_tensor)

        loss_class = criterion_class(out_class, y_train_class_tensor)

# حساب الخسارة
        try:
            loss_reg = criterion_reg(out_reg, y_train_reg_tensor)
            if torch.isnan(loss_reg) or torch.isinf(loss_reg):
                print(f"⚠️ Epoch {epoch+1}: Regression loss is unstable. Skipping regression.")
                loss = loss_class
            else:
                # ندمج الخسارتين، مع تعزيز الانحدار بمعامل 2.0.
                loss = loss_class + 0.2 * loss_reg
        except Exception as e:
            print(f"⚠️ Regression loss error: {e}")
            loss = loss_class
            
# التدرج والتحديث
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/250 - Loss: {loss.item():.4f}")

# حفظ النموذج
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to '{save_path}'")
    
#  التقييم بعد التدريب
    
    # إيقاف التدريب.
    model.eval()
    with torch.no_grad():
        # تحويل بيانات الاختبار إلى Tensor.
        # تنفيذ forward pass.
        # حساب التوقعات.استخراج فئات التصنيف.
        # تحويل التوقعات إلى numpy.
        # حساب دقة التصنيف.
        # استخراج فئات التصنيف.
        preds_class, preds_reg = model(X_test_tensor)
        # إعادة القيم الأصلية للانحدار
        # استرجاع magnitude, depth بالقيم الأصلية.
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
 
            # حساب MSE للانحدار.
            # حساب MSE باستخدام sklearn.metrics.
            mse = mean_squared_error(y_test_reg, preds_reg_inv)
            print("\n📉 Regression MSE:")
            print(mse)
            
            # 📊 Confusion Matrix
            #  حفظ المصفوفة والتوزيع
            cm = confusion_matrix(y_test_class, preds_class_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Hafif', 'Orta', 'Güçlü', 'Şiddetli', 'Felaket'])
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title("Confusion Matrix")
            plt.tight_layout()
            os.makedirs("data", exist_ok=True)
            plt.savefig("earthquake_risk_predictor/data/confusion_matrix.png")
            print("📊 Saved confusion matrix to 'earthquake_risk_predictor/data/confusion_matrix.png'")
            
            # رسم توزيع الخطأ.
            errors = np.abs(y_test_reg.values - preds_reg_inv)
            plt.figure(figsize=(10, 4))
            plt.hist(errors[:, 0], bins=30, alpha=0.6, label='Magnitude Error')
            plt.hist(errors[:, 1], bins=30, alpha=0.6, label='Depth Error')
            plt.title("Regression Error Distribution")
            plt.legend()
            plt.tight_layout()
            os.makedirs("data", exist_ok=True)
            plt.savefig("earthquake_risk_predictor/data/regression_errors.png")
            print("📉 Saved regression error distribution to 'earthquake_risk_predictor/data/regression_errors.png'")
        except Exception as e:
            print(f"⚠️ Error calculating regression metrics: {e}")



# حفظ التوقعات
# نفس فكرة predict لكن تحفظ النتائج في CSV.
def predict_and_save_outputs(model, X_test, y_test_class, y_test_reg, output_path="earthquake_risk_predictor/data/predictions.csv"):
    global reg_scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

# التحويل والتنبؤ:
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_class, preds_reg = model(X_test_tensor)
        class_labels = torch.argmax(preds_class, axis=1).cpu().numpy()
        preds_reg = reg_scaler.inverse_transform(preds_reg.cpu().numpy())


# حفظ CSV:
    df_out = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(X_test.shape[1])])
    df_out["true_class"] = y_test_class.values
    df_out["pred_class"] = class_labels
    df_out["true_magnitude"] = y_test_reg.iloc[:, 0].values
    df_out["pred_magnitude"] = preds_reg[:, 0]
    df_out["true_depth_km"] = y_test_reg.iloc[:, 1].values
    df_out["pred_depth_km"] = preds_reg[:, 1]

    df_out.to_csv(output_path, index=False)
    print(f"📄 Predictions saved to {output_path}")



# انحدار XGBoost
def run_xgboost_regressor(X_train, X_test, y_train_reg, y_test_reg):
    os.makedirs("data", exist_ok=True)
    results = {}

# يدرب نموذج XGBRegressor لكل من magnitude, depth_km.
    for i, label in enumerate(["magnitude", "depth_km"]):
        model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train_reg.iloc[:, i])
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test_reg.iloc[:, i], preds)
        r2 = r2_score(y_test_reg.iloc[:, i], preds)
        mae = mean_absolute_error(y_test_reg.iloc[:, i], preds)

        results[label] = {"MSE": mse, "MAE": mae, "R2": r2}
        plt.figure()
        #  حساب الأداء والرسم البياني.
        # رسم التوقع مقابل الحقيقي.
        plt.scatter(y_test_reg.iloc[:, i], preds, alpha=0.5)
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"XGBoost Prediction: {label}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"earthquake_risk_predictor/data/xgboost_results_{label}.png")

# حفظ الأداء في CSV.
    pd.DataFrame(results).T.to_csv("earthquake_risk_predictor/data/xgboost_metrics.csv")
    print("📊 XGBoost metrics saved to 'earthquake_risk_predictor/data/xgboost_metrics.csv'")


#  خريطة تركيا
def plot_geographic_distribution(df, lat_col="latitude", lon_col="longitude", border_file="tr.json"):
    """
    Plots earthquakes on a Turkey map using GeoJSON borders.
    """
    # تحويل DataFrame إلى GeoDataFrame
    # تحويل بيانات الزلازل إلى خريطة.
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



    # رسم الزلازل على الخريطة.
    # تلوين الزلازل حسب magnitude.
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