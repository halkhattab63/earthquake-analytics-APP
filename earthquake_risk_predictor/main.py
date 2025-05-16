# import os
# import sys
# import pandas as pd

# from src.train import train_model, predict_and_save_outputs, run_xgboost_regressor, plot_geographic_distribution
# from src.model import EarthquakeModel
# from src.preprocessing import load_data, export_processed_data, plot_severity_distribution
# from src.evaluation import  generate_pdf_report, plot_regression_differences


# sys.stdout.reconfigure(encoding='utf-8')
# from src.evaluation import generate_html_report


# # def clean_old_outputs():
# #     files_to_remove = [
# #         "predictions.csv", "processed_earthquakes.csv",
# #         "severity_distribution.png", "regression_errors.png",
# #         "geographic_distribution.png", "xgboost_metrics.csv",
# #         "xgboost_results_magnitude.png", "xgboost_results_depth_km.png"
# #     ]
# #     for file in files_to_remove:
# #         if os.path.exists(file):
# #             os.remove(file)
# #             print(f"🧹 Removed: {file}")

# def main():
#     print("🚀 Starting Earthquake Smart System Pipeline...\n")
#     os.makedirs("data", exist_ok=True)
#     # clean_old_outputs()

#     # 1. Load and preprocess data
#     X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = load_data(
#         "D:/books/4.1 donem/Bitirme projesi/codes/earthquake analytics APP/earthquake_risk_predictor/data/dataset/usgs_earthquakes_turkey2020.csv", use_smote=True)
#     # earthquake_risk_predictor/data/dataset/usgs_earthquakes_turkey2020.csv

#     print("🔍 Distribution after SMOTE:")
#     print(y_train_class.value_counts(), "\n")

#     # 2. Initialize and train model
#     model = EarthquakeModel(input_dim=X_train.shape[1])
#     train_model(model, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)

#     # 3. XGBoost Regression for Benchmarking
#     run_xgboost_regressor(X_train, X_test, y_train_reg, y_test_reg)

#     # 4. Save processed data and plots
#     export_processed_data()
#     plot_severity_distribution()
#     plot_regression_differences(y_test_reg)

#     # 5. Predict and save output
#     predict_and_save_outputs(model, X_test, y_test_class, y_test_reg)
#     # في نهاية main()



#     # 7. Plot geographic distribution
#     if os.path.exists("earthquake_risk_predictor/data/processed_earthquakes.csv"):
#         df = pd.read_csv("earthquake_risk_predictor/data/processed_earthquakes.csv")
#         plot_geographic_distribution(df)
#     else:
#         print("⚠️ Processed data not found. Skipping geographic plot.")

#     print("\n✅ All steps completed successfully.")
#     print("📦 Model and results saved in the '' folder.")
    
#     generate_pdf_report("earthquake_risk_predictor\\data")
#     generate_html_report("earthquake_risk_predictor\\data")
# if __name__ == '__main__':
    
#     main() 




# main.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("ann_predictor")

from earthquake_risk_predictor.src.PyTorch_predictor.train import train_model, predict_and_save_outputs, run_xgboost_regressor, plot_geographic_distribution
from earthquake_risk_predictor.src.PyTorch_predictor.model import EarthquakeModel
from earthquake_risk_predictor.src.PyTorch_predictor.preprocessing import load_data, export_processed_data, plot_severity_distribution
from earthquake_risk_predictor.src.PyTorch_predictor.evaluation import generate_pdf_report, plot_regression_differences, generate_html_report

import tensorflow as tf
from src.ann_predictor.utils_ann import load_data as load_ann_data
from src.ann_predictor.ann_model import build_ann_model
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def train_keras_model():
    print("\n🔁 [Keras] Training ANN model...")

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_ann_data()

    model = build_ann_model(input_dim=X_train.shape[1], num_classes=y_class_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "class_output": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            "reg_output": tf.keras.losses.MeanSquaredError(),
        },
        metrics={
            "class_output": tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            "reg_output": tf.keras.metrics.MeanAbsoluteError(name="mae"),
        }
    )

    os.makedirs("ann_predictor/outputs", exist_ok=True)

    model.fit(
        X_train,
        {"class_output": y_class_train, "reg_output": y_reg_train},
        validation_data=(X_test, {"class_output": y_class_test, "reg_output": y_reg_test}),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("ann_predictor/outputs/model_ann.keras", save_best_only=True)
        ],
        verbose=2
    )

    print("\n✅ [Keras] Model saved to model_ann.keras")

    model = tf.keras.models.load_model("ann_predictor/outputs/model_ann.keras")
    y_pred_class_logits, y_pred_reg = model.predict(X_test)

    y_true_class = y_class_test.argmax(axis=1)
    y_pred_class = y_pred_class_logits.argmax(axis=1)
    labels = ["Hafif", "Orta", "Güçlü", "Şiddetli", "Felaket"]
    y_pred_labels = [labels[i] for i in y_pred_class]

    print("\n[Keras] Classification Report:")
    print(classification_report(y_true_class, y_pred_class, target_names=labels))

    cm = confusion_matrix(y_true_class, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ann_predictor/outputs/confusion_matrix.png")

    mse = mean_squared_error(y_reg_test, y_pred_reg)
    mae = mean_absolute_error(y_reg_test, y_pred_reg)
    print(f"\n📉 Regression MSE: {mse:.4f}")
    print(f"📉 Regression MAE: {mae:.4f}")

    errors = np.abs(y_pred_reg - y_reg_test)
    plt.figure(figsize=(10, 4))
    plt.hist(errors[:, 0], bins=30, alpha=0.6, label='Magnitude Error')
    plt.hist(errors[:, 1], bins=30, alpha=0.6, label='Depth Error')
    plt.title("Regression Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("-ann_predictor/outputs/regression_errors.png")

    df_preds = pd.DataFrame({
        "Predicted_Class_Index": y_pred_class,
        "Predicted_Class_Label": y_pred_labels,
        "Predicted_Magnitude": y_pred_reg[:, 0],
        "Predicted_Depth_km": y_pred_reg[:, 1],
    })
    df_preds.to_csv("ann_predictor/outputs/predictions_ann.csv", index=False)
    print("📁 predictions_ann.csv saved.")

def main():
    print("Starting Earthquake Smart System Pipeline...\n")

    os.makedirs("data", exist_ok=True)

    print("\n================ PyTorch Model =================")
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = load_data(
        "earthquake_risk_predictor/data/dataset/usgs_earthquakes_turkey2020.csv", use_smote=True)

    print("🔍 Distribution after SMOTE:")
    print(y_train_class.value_counts(), "\n")

    model = EarthquakeModel(input_dim=X_train.shape[1])
    train_model(model, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)

    run_xgboost_regressor(X_train, X_test, y_train_reg, y_test_reg)
    export_processed_data()
    plot_severity_distribution()
    plot_regression_differences(y_test_reg)
    predict_and_save_outputs(model, X_test, y_test_class, y_test_reg)

    if os.path.exists("earthquake_risk_predictor/data/processed_earthquakes.csv"):
        df = pd.read_csv("earthquake_risk_predictor/data/processed_earthquakes.csv")
        plot_geographic_distribution(df)

    generate_pdf_report("earthquake_risk_predictor/data")
    generate_html_report("earthquake_risk_predictor/data")

    print("\n================ Keras ANN Model =================")
    train_keras_model()

    print("\n✅ All models completed successfully.")

if __name__ == '__main__':
    main()
