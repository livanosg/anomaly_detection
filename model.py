import numpy as np
from keras import Sequential, layers
from sklearn.metrics import precision_recall_curve

from config import INPUT_SHAPE
from dataset import get_dataset_predictions


def get_model():
    return Sequential([
        layers.SeparableConv2D(8, 3, depth_multiplier=3, padding='same', activation='relu', input_shape=INPUT_SHAPE),
        layers.MaxPooling2D(),
        layers.SeparableConv2D(16, 7, depth_multiplier=5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.SeparableConv2D(32, 9, depth_multiplier=7, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, "softmax"),
    ])


def calculate_threshold(y_true, probas_pred):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=probas_pred)
    eucl_dist = np.sqrt(np.power(1. - recall, 2) + np.power(1. - precision, 2))  # Distance from (1, 1)
    return thresholds[np.argmin(eucl_dist)]
    # f1 = 2. * (precision * recall) / (precision + recall)
    # return thresholds[np.argmax(f1)]
    # return thresholds[np.argmin(x)]
