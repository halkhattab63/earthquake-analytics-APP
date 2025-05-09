import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from fpdf import FPDF
import os

def generate_pdf_report(data_folder, output_path="earthquake_risk_predictor/data/earthquake_report.pdf"):
    os.makedirs(data_folder, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_title(text, size=16):
        pdf.set_font("Arial", 'B', size)
        pdf.cell(0, 10, txt=text, ln=True, align='L')

    def add_image(img_path, w=180, h=0):
        if os.path.exists(img_path):
            pdf.image(img_path, w=w, h=h)
        else:
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"[Missing Image] {img_path}", ln=True)

    pdf.add_page()
    add_title("Earthquake Risk Analysis Report", size=20)

    add_title("\nConfusion Matrix / ROC Curve:")
    add_image(os.path.join(data_folder, "confusion_matrix.png"))
    add_image(os.path.join(data_folder, "roc_curve.png"))

    add_title("\nClass Distribution:")
    add_image(os.path.join(data_folder, "severity_distribution.png"))

    add_title("\nRegression Error Distribution:")
    add_image(os.path.join(data_folder, "regression_errors.png"))

    add_title("\nGeographic Risk Map:")
    add_image(os.path.join(data_folder, "geographic_distribution.png"))

    add_title("\nXGBoost Results:")
    add_image(os.path.join(data_folder, "xgboost_results_magnitude.png"))
    add_image(os.path.join(data_folder, "xgboost_results_depth_km.png"))

    preds_path = os.path.join(data_folder, "predictions.csv")
    if os.path.exists(preds_path):
        df = pd.read_csv(preds_path)
        add_title("\nPrediction Summary Statistics:")
        stats = df.describe().round(2)
        pdf.set_font("Courier", size=8)
        for line in stats.to_string().split("\n"):
            pdf.cell(0, 5, line, ln=True)
    else:
        pdf.cell(0, 10, "predictions.csv not found", ln=True)

    pdf.output(output_path)
    print(f"✅ PDF report generated: {output_path}")


def generate_html_report(data_folder, output_path="earthquake_risk_predictor/data/report.html"):
    os.makedirs(data_folder, exist_ok=True)
    preds_path = os.path.join(data_folder, "predictions.csv")
    img_files = [
        "confusion_matrix.png",
        "roc_curve.png",
        "severity_distribution.png",
        "regression_errors.png",
        "geographic_distribution.png",
        "xgboost_results_magnitude.png",
        "xgboost_results_depth_km.png"
    ]

    html = '''
    <html>
    <head>
        <meta charset="utf-8">
        <title>Earthquake Risk Report</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #fdfdfd; padding: 30px; color: #333; }
            h1 { font-size: 26px; margin-bottom: 10px; }
            h2 { font-size: 20px; margin-top: 40px; color: #444; }
            img { max-width: 100%; margin: 10px 0; border: 1px solid #ccc; }
            table { border-collapse: collapse; width: 100%; background-color: #fff; }
            th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: center; }
            th { background-color: #eee; }
            .missing { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Earthquake Risk Analysis - HTML Report</h1>
    '''

    for img in img_files:
        full_path = os.path.join(data_folder, img)
        if os.path.exists(full_path):
            section = img.replace("_", " ").replace(".png", "").title()
            html += f"<h2>{section}</h2><img src='{img}' alt='{img}'>"
        else:
            html += f"<p class='missing'>[Missing Image] {img}</p>"

    if os.path.exists(preds_path):
        df = pd.read_csv(preds_path)
        html += "<h2>Prediction Results (sample)</h2>"
        html += df.head(50).to_html(index=False, border=1)
    else:
        html += "<p class='missing'>[Missing File] predictions.csv not found</p>"

    html += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"🌐 HTML report created: {output_path}")


def plot_regression_differences(y_test_reg, y_pred_reg=None, output_path="earthquake_risk_predictor/data/regression_errors.png"):
    if y_pred_reg is None or isinstance(y_pred_reg, str):
        if os.path.exists("earthquake_risk_predictor/data/predictions.csv"):
            df = pd.read_csv("earthquake_risk_predictor/data/predictions.csv")
            y_pred_reg = df[['pred_magnitude', 'pred_depth_km']].values
            y_test_reg = df[['true_magnitude', 'true_depth_km']].values
        else:
            print("⚠️ File predictions.csv not found to plot errors.")
            return

    errors = y_pred_reg - y_test_reg
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(errors[:, 0], bins=40, color='skyblue', edgecolor='black')
    axs[0].set_title("Magnitude Prediction Error")
    axs[0].set_xlabel("Error (Magnitude)")
    axs[0].set_ylabel("Frequency")
    axs[0].grid(True)

    axs[1].hist(errors[:, 1], bins=40, color='salmon', edgecolor='black')
    axs[1].set_title("Depth Prediction Error")
    axs[1].set_xlabel("Error (Depth_km)")
    axs[1].set_ylabel("Frequency")
    axs[1].grid(True)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(output_path)
    print(f"📉 Saved regression error distribution to '{output_path}'")
