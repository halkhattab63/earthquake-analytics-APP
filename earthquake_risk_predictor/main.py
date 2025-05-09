import os
import sys
import pandas as pd

from src.train import train_model, predict_and_save_outputs, run_xgboost_regressor, plot_geographic_distribution
from src.model import EarthquakeModel
from src.preprocessing import load_data, export_processed_data, plot_severity_distribution
from src.evaluation import  generate_pdf_report, plot_regression_differences

sys.stdout.reconfigure(encoding='utf-8')
from src.evaluation import generate_html_report


# def clean_old_outputs():
#     files_to_remove = [
#         "predictions.csv", "processed_earthquakes.csv",
#         "severity_distribution.png", "regression_errors.png",
#         "geographic_distribution.png", "xgboost_metrics.csv",
#         "xgboost_results_magnitude.png", "xgboost_results_depth_km.png"
#     ]
#     for file in files_to_remove:
#         if os.path.exists(file):
#             os.remove(file)
#             print(f"🧹 Removed: {file}")

def main():
    print("🚀 Starting Earthquake Smart System Pipeline...\n")
    os.makedirs("data", exist_ok=True)
    # clean_old_outputs()

    # 1. Load and preprocess data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = load_data(
        "D:/books/4.1 donem/Bitirme projesi/codes/earthquake analytics APP/earthquake_risk_predictor/data/dataset/usgs_earthquakes_turkey2020.csv", use_smote=True)
    # earthquake_risk_predictor/data/dataset/usgs_earthquakes_turkey2020.csv

    print("🔍 Distribution after SMOTE:")
    print(y_train_class.value_counts(), "\n")

    # 2. Initialize and train model
    model = EarthquakeModel(input_dim=X_train.shape[1])
    train_model(model, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)

    # 3. XGBoost Regression for Benchmarking
    run_xgboost_regressor(X_train, X_test, y_train_reg, y_test_reg)

    # 4. Save processed data and plots
    export_processed_data()
    plot_severity_distribution()
    plot_regression_differences(y_test_reg)

    # 5. Predict and save output
    predict_and_save_outputs(model, X_test, y_test_class, y_test_reg)
    # في نهاية main()



    # 7. Plot geographic distribution
    if os.path.exists("earthquake_risk_predictor/data/processed_earthquakes.csv"):
        df = pd.read_csv("earthquake_risk_predictor/data/processed_earthquakes.csv")
        plot_geographic_distribution(df)
    else:
        print("⚠️ Processed data not found. Skipping geographic plot.")

    print("\n✅ All steps completed successfully.")
    print("📦 Model and results saved in the '' folder.")
    
    generate_pdf_report("earthquake_risk_predictor\\data")
    generate_html_report("earthquake_risk_predictor\\data")
if __name__ == '__main__':
    main()
