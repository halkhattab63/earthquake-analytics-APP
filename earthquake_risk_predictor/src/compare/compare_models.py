import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve, auc, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

sys.stdout.reconfigure(encoding='utf-8')

# إعداد المسارات
BASE_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(BASE_OUTPUT, exist_ok=True)

# تضمين مجلد src لتمكين import صحيح
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_DIR)

from ann_predictor.utils_ann import load_data

keras_preds = pd.read_csv(os.path.join(SRC_DIR, "ann_predictor", "outputs", "predictions_ann.csv"))
pytorch_preds = pd.read_csv(os.path.join(SRC_DIR, "..", "data", "predictions.csv"))

LABELS = ["Hafif", "Orta", "Güçlü", "Şiddetli", "Felaket"]

_, X_test, _, y_class_test, _, y_reg_test = load_data()
y_true_class = y_class_test.argmax(axis=1)
y_true_bin = label_binarize(y_true_class, classes=list(range(len(LABELS))))

pytorch_preds_test = pytorch_preds.iloc[:len(y_true_class)].reset_index(drop=True)

# F1 Score Comparison
keras_f1 = classification_report(y_true_class, keras_preds["Predicted_Class_Index"], target_names=LABELS, output_dict=True)
pytorch_f1 = classification_report(y_true_class, pytorch_preds_test["pred_class"], target_names=LABELS, output_dict=True)

plt.figure(figsize=(8, 4))
keras_scores = [keras_f1[label]["f1-score"] for label in LABELS]
pytorch_scores = [pytorch_f1[label]["f1-score"] for label in LABELS]
x = range(len(LABELS))
plt.bar(x, keras_scores, width=0.4, label="Keras ANN", align="center")
plt.bar([i + 0.4 for i in x], pytorch_scores, width=0.4, label="PyTorch/XGBoost", align="center")
plt.xticks([i + 0.2 for i in x], LABELS)
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("F1 Score per Class")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT, "compare_f1_scores.png"))

# Confusion Matrices
keras_cm = confusion_matrix(y_true_class, keras_preds["Predicted_Class_Index"], normalize="true")
pytorch_cm = confusion_matrix(y_true_class, pytorch_preds_test["pred_class"], normalize="true")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(keras_cm, annot=True, fmt=".2f", ax=axes[0], xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
axes[0].set_title("Keras ANN Confusion Matrix")
sns.heatmap(pytorch_cm, annot=True, fmt=".2f", ax=axes[1], xticklabels=LABELS, yticklabels=LABELS, cmap="Purples")
axes[1].set_title("PyTorch/XGBoost Confusion Matrix")
for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT, "compare_confusion_matrices.png"))

# Regression
keras_mae = mean_absolute_error(y_reg_test, keras_preds[["Predicted_Magnitude", "Predicted_Depth_km"]])
pytorch_mae = mean_absolute_error(y_reg_test, pytorch_preds_test[["pred_magnitude", "pred_depth_km"]])
keras_mse = mean_squared_error(y_reg_test, keras_preds[["Predicted_Magnitude", "Predicted_Depth_km"]])
pytorch_mse = mean_squared_error(y_reg_test, pytorch_preds_test[["pred_magnitude", "pred_depth_km"]])

print("\n📉 Regression Comparison:")
print(f"Keras ANN        - MAE: {keras_mae:.4f}, MSE: {keras_mse:.4f}")
print(f"PyTorch/XGBoost  - MAE: {pytorch_mae:.4f}, MSE: {pytorch_mse:.4f}")

# ROC-AUC + softmax
keras_logits = keras_preds["Predicted_Class_Index"]
keras_probs = tf.one_hot(keras_logits, depth=len(LABELS)).numpy()
pytorch_probs = pd.get_dummies(pytorch_preds_test["pred_class"]).reindex(columns=range(len(LABELS)), fill_value=0)

def plot_roc_curves(y_true_bin, y_scores, model_name, save_name):
    plt.figure(figsize=(8, 6))
    for i in range(len(LABELS)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{LABELS[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, save_name))

plot_roc_curves(y_true_bin, keras_probs, "Keras ANN", "roc_keras_ann.png")
plot_roc_curves(y_true_bin, pytorch_probs.values, "PyTorch", "roc_pytorch.png")

# Brier Scores
print("\n📉 Brier Score per class:")
for i in range(len(LABELS)):
    keras_brier = brier_score_loss(y_true_bin[:, i], keras_probs[:, i])
    pytorch_brier = brier_score_loss(y_true_bin[:, i], pytorch_probs.values[:, i])
    print(f"🔸 {LABELS[i]}: Keras={keras_brier:.4f}, PyTorch={pytorch_brier:.4f}")

# Reliability Diagrams

def plot_reliability_diagram(probs, y_true_bin, model_name, save_name):
    plt.figure(figsize=(7, 5))
    for i in range(len(LABELS)):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, i], probs[:, i], n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker='o', label=f"{LABELS[i]}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.title(f"Reliability Diagram - {model_name}")
    plt.xlabel("Confidence")
    plt.ylabel("Actual Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, save_name))

plot_reliability_diagram(keras_probs, y_true_bin, "Keras ANN", "reliability_keras.png")
plot_reliability_diagram(pytorch_probs.values, y_true_bin, "PyTorch", "reliability_pytorch.png")
