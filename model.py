import keras
import numpy as np
from keras import Sequential, Input
from keras.layers import Dense, SeparableConv2D, Flatten, MaxPooling2D, Convolution2DTranspose, UpSampling2D
from sklearn.metrics import precision_recall_curve

from datasets import split_inputs_labels


def supervised_anomaly_detector(conf):
    return Sequential([
        Input(shape=conf.input_shape),
        SeparableConv2D(16, 3, depth_multiplier=3, padding='valid', activation='relu'),
        SeparableConv2D(8, 5, depth_multiplier=2, padding='valid', activation='relu'),
        MaxPooling2D(),
        SeparableConv2D(8, 3, depth_multiplier=1, padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation="softmax")
    ])


def unsupervised_anomaly_detector(conf):
    return Sequential(
        [Input(shape=conf.input_shape),
         SeparableConv2D(8, 7, depth_multiplier=3, padding="valid", activation="relu"),
         SeparableConv2D(8, 5, depth_multiplier=2, padding="valid", activation="relu"),
         MaxPooling2D(),
         SeparableConv2D(16, 3, depth_multiplier=1, padding="valid", activation="relu"),
         MaxPooling2D(),
         Convolution2DTranspose(16, 3, padding="valid"),
         UpSampling2D(size=(2, 2), interpolation="nearest"),
         Convolution2DTranspose(8, 5, padding="valid"),
         UpSampling2D(size=(2, 2), interpolation="nearest"),
         Convolution2DTranspose(6, 7, padding="valid"),
         SeparableConv2D(3, 7, depth_multiplier=3, padding="valid", activation="sigmoid"),
         ])


def calculate_threshold(model, dataset, conf):
    x_input, y_true = split_inputs_labels(dataset)
    y_true = np.concatenate(list(y_true.as_numpy_iterator()))
    y_prob = model.predict(x_input).squeeze()
    np.save(conf.y_true, y_true)
    np.save(conf.y_prob, y_prob)
    threshold = None
    if conf.method == "supervised":
        precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_prob)
        euclidean_distance = np.sqrt(np.power(1. - recall, 2) + np.power(1. - precision, 2))  # Distance from (1, 1)
        threshold = thresholds[np.argmin(euclidean_distance)]

    if conf.method == "unsupervised":
        loss = keras.losses.mean_squared_error(y_true, y_prob)
        threshold = np.mean(loss) + np.std(loss)

    np.save(conf.threshold_path, threshold)
