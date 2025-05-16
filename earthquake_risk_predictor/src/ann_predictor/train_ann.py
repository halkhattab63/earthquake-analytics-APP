# ann_predictor/train_ann.py

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ann_model import build_ann_model
from earthquake_risk_predictor.src.ann_predictor.focal_loss import FocalLoss
from utils_ann import load_data

# تحميل البيانات
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_data()

# ⚖️ استخراج فئات التصنيف كـ أرقام
y_train_labels = np.argmax(y_class_train, axis=1)

# ✅ حساب class weights تلقائيًا للفئات غير المتوازنة
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights_arr))
print("📊 Class Weights:", class_weights)

# بناء النموذج
model = build_ann_model(input_dim=X_train.shape[1], num_classes=y_class_train.shape[1])

# تجميع النموذج

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "class_output": FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
        "reg_output": tf.keras.losses.MeanSquaredError(),
    },
    metrics={
        "class_output": tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        "reg_output": tf.keras.metrics.MeanAbsoluteError(name="mae"),
    }
)

# إعداد المسارات
os.makedirs("ann_predictor/logs", exist_ok=True)
os.makedirs("ann_predictor/outputs", exist_ok=True)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint("ann_predictor/outputs/model_ann.keras", save_best_only=True)
]

# تدريب النموذج مع أوزان الفئات
model.fit(
    X_train,
    {"class_output": y_class_train, "reg_output": y_reg_train},
    validation_data=(X_test, {"class_output": y_class_test, "reg_output": y_reg_test}),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
    class_weight={"class_output": class_weights}
)
