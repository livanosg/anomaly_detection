import numpy as np
from keras import Sequential, layers
from sklearn.metrics import precision_recall_curve

from config import INPUT_SHAPE


def get_model():
    """
    Constructs and returns a Keras Sequential model for anomaly detection.

    Returns:
        tf.keras.Model: A Sequential model for anomaly detection.
    """
    return Sequential([
        layers.InputLayer(input_shape=INPUT_SHAPE),
        layers.SeparableConv2D(16, 3, depth_multiplier=3, padding='valid', activation='relu'),  #, input_shape=INPUT_SHAPE
        layers.SeparableConv2D(8, 5, depth_multiplier=2, padding='valid', activation='relu'),
        layers.MaxPooling2D(),
        layers.SeparableConv2D(8, 3, depth_multiplier=1, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2,  activation="softmax"),
    ])


def calculate_threshold(y_true, probas_pred):
    """
        Calculates a threshold for anomaly detection using precision-recall curve.

        Args:
            y_true (np.ndarray): True labels.
            probas_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: The calculated threshold.
        """

    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=probas_pred)
    euclidean_distance = np.sqrt(np.power(1. - recall, 2) + np.power(1. - precision, 2))  # Distance from (1, 1)
    return thresholds[np.argmin(euclidean_distance)]
