# ann_predictor/focal_loss.py

import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1. - y_pred, self.gamma)
        loss = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # يرجع scalar لكل batch
