# ann_predictor/ann_model.py

import tensorflow as tf


def build_ann_model(input_dim: int, num_classes: int = 5) -> tf.keras.Model:
    """
    Builds a Multi-Task ANN for:
      - Classification of earthquake severity (5 classes)
      - Regression of [magnitude, depth_km]
    Shared layers: 3 Dense layers with LeakyReLU, BatchNorm, Dropout.
    """

    input_layer = tf.keras.Input(shape=(input_dim,), name="input")

    # Shared feature extractor
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Classification head (logits)
    class_output = tf.keras.layers.Dense(num_classes, name="class_output")(x)

    # Regression head (positive outputs)
    reg_output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softplus, name="reg_output")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[class_output, reg_output], name="EarthquakeANN")
    return model
