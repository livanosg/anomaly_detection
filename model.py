import numpy as np
from keras import Sequential, layers
from sklearn.metrics import precision_recall_curve


class ModelHandler:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.SeparableConv2D(16, 3, depth_multiplier=3, padding='valid', activation='relu'),
            layers.SeparableConv2D(8, 5, depth_multiplier=2, padding='valid', activation='relu'),
            layers.MaxPooling2D(),
            layers.SeparableConv2D(8, 3, depth_multiplier=1, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(2,  activation="softmax")
        ])


def calculate_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_prob)
    euclidean_distance = np.sqrt(np.power(1. - recall, 2) + np.power(1. - precision, 2))  # Distance from (1, 1)
    return thresholds[np.argmin(euclidean_distance)]
