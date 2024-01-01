import os

import numpy as np
from keras import Sequential, Input
from keras.layers import Dense, SeparableConv2D, Flatten, MaxPooling2D, Convolution2DTranspose
from sklearn.metrics import precision_recall_curve


def supervised_anomaly_detector(input_shape):
    return Sequential([
        Input(shape=input_shape),
        SeparableConv2D(16, 3, depth_multiplier=3, padding='valid', activation='relu'),
        SeparableConv2D(8, 5, depth_multiplier=2, padding='valid', activation='relu'),
        MaxPooling2D(),
        SeparableConv2D(8, 3, depth_multiplier=1, padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation="sigmoid")
    ])


def unsupervised_anomaly_detector(input_shape):
    return Sequential(
        [Input(shape=input_shape),
         SeparableConv2D(128, 7, depth_multiplier=5, padding="same", activation="relu"),  # (224, 224)
         MaxPooling2D(),  # (112, 112)
         SeparableConv2D(64, 5, depth_multiplier=4, padding="same", activation="relu"),  # (112, 112)
         SeparableConv2D(32, 3, depth_multiplier=3, padding="same", activation="relu"),  # (112, 112)
         MaxPooling2D(),  # (56, 56)
         SeparableConv2D(32, 3, depth_multiplier=2, padding="same", activation="relu"),  # (56, 56)
         MaxPooling2D(),  # (28, 28)
         SeparableConv2D(16, 3, depth_multiplier=1, padding="same", activation="relu"),  # (28, 28)
         Convolution2DTranspose(32, 3, (2, 2), padding="same", activation="relu"),  # (56, 56)
         Convolution2DTranspose(32, 3, (2, 2), padding="same", activation="relu"),  # (112, 112)
         SeparableConv2D(64, 5, depth_multiplier=2, padding="same", activation="relu"),  # (112, 112)
         Convolution2DTranspose(128, 7, (2, 2), padding="same", activation="relu"),  # (224, 224)
         SeparableConv2D(3, 3, depth_multiplier=3, padding="same", activation="sigmoid")  # (224, 224)
         ])


def pr_threshold(y_true, probas_pred, conf, save=True):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=probas_pred)
    euclidean_distance = np.sqrt(np.power(1. - recall, 2) + np.power(1. - precision, 2))  # Distance from (1, 1)
    threshold = thresholds[np.argmin(euclidean_distance)]
    if save:
        np.save(os.path.join(conf.model_dir, "threshold.npy"), threshold)
    return threshold
