# ann_predictor/evaluate_ann.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error
)
from utils_ann import load_data

LABELS = ["Hafif", "Orta", "Güçlü", "Şiddetli", "Felaket"]

# تحميل النموذج
model = tf.keras.models.load_model("ann_predictor/outputs/model_ann.keras")

# تحميل البيانات
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_data()

# التنبؤ
y_pred_class_logits, y_pred_reg = model.predict(X_test)
y_prob = tf.nn.softmax(y_pred_class_logits).numpy()
y_true_class = y_class_test.argmax(axis=1)
y_pred_class = y_prob.argmax(axis=1)

# 📊 Classification Report
report = classification_report(y_true_class, y_pred_class, target_names=LABELS, output_dict=True)
print("\n📊 Classification Report:")
print(classification_report(y_true_class, y_pred_class, target_names=LABELS))

# 🔥 Identify Weak Classes
weak_classes = [LABELS[i] for i, scores in enumerate(report.values()) if isinstance(scores, dict) and scores["f1-score"] < 0.7]
print(f"\n⚠️ Weak classes (F1 < 0.7): {weak_classes}")

# 🎯 Confusion Matrix (%)
cm = confusion_matrix(y_true_class, y_pred_class, normalize="true") * 100
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix (%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
os.makedirs("ann_predictor/outputs", exist_ok=True)
plt.savefig("ann_predictor/outputs/confusion_matrix.png")
print("✅ Confusion matrix saved.")

# 📈 ROC Curves
plt.figure(figsize=(8, 6))
for i in range(len(LABELS)):
    fpr, tpr, _ = roc_curve(y_class_test[:, i], y_prob[:, i])
    auc = roc_auc_score(y_class_test[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("ann_predictor/outputs/roc_curves.png")
print("✅ ROC curves saved.")

# 📊 F1 Score per Class Bar Chart
f1_scores = [report[label]["f1-score"] for label in LABELS]
plt.figure(figsize=(7, 4))
sns.barplot(x=LABELS, y=f1_scores)
plt.ylim(0, 1)
plt.title("F1 Score per Class")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("ann_predictor/outputs/f1_scores.png")
print("✅ F1 scores chart saved.")

# 📉 Regression Evaluation
mse = mean_squared_error(y_reg_test, y_pred_reg)
mae = mean_absolute_error(y_reg_test, y_pred_reg)
print(f"\n📉 Regression MSE: {mse:.4f}")
print(f"📉 Regression MAE: {mae:.4f}")

# 🧭 Regression Error Distribution
errors = np.abs(y_pred_reg - y_reg_test)
plt.figure(figsize=(10, 4))
plt.hist(errors[:, 0], bins=30, alpha=0.6, label='Magnitude Error')
plt.hist(errors[:, 1], bins=30, alpha=0.6, label='Depth Error')
plt.title("Regression Error Distribution")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("ann_predictor/outputs/regression_errors.png")
print("✅ Regression error histogram saved.")
