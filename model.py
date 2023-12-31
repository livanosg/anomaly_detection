import keras
import numpy as np
from keras import Sequential, Input
from keras.layers import Dense, SeparableConv2D, Flatten, MaxPooling2D, Convolution2DTranspose, UpSampling2D
from sklearn.metrics import precision_recall_curve

from datasets import split_inputs_labels, get_dataset
from icecream import ic

from project_manage import Config


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
         SeparableConv2D(64, 3, depth_multiplier=3, padding="same", activation="relu"),  # (180, 180)
         MaxPooling2D(),  # (90, 90) /2
         SeparableConv2D(32, 3, depth_multiplier=1, padding="same", activation="relu"),  # (90, 90) -2
         SeparableConv2D(16, 3, depth_multiplier=1, padding="same", activation="relu"),  # (90, 90) -2
         MaxPooling2D(),  # (45, 45) /2
         SeparableConv2D(8, 3, depth_multiplier=1, padding="same", activation="relu"),  # (45, 45) -2
         Convolution2DTranspose(16, 3, (2, 2), padding="same", activation="relu"),  # (90, 90)
         SeparableConv2D(32, 3, depth_multiplier=1, padding="same", activation="relu"),  # (90, 90) -2
         Convolution2DTranspose(64, 3, (2, 2), padding="same", activation="relu"),  # (180, 180)
         SeparableConv2D(3, 3, depth_multiplier=1, padding="same", activation="sigmoid")  # (180, 180)
         ])


def calculate_threshold(model, dataset, conf):
    x_input, y_true = split_inputs_labels(dataset)
    y_true = np.concatenate(list(y_true.as_numpy_iterator()))
    y_prob = model.predict(x_input).squeeze()
    threshold = None
    if conf.method == "supervised":
        precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_prob)
        euclidean_distance = np.sqrt(np.power(1. - recall, 2) + np.power(1. - precision, 2))  # Distance from (1, 1)
        threshold = thresholds[np.argmin(euclidean_distance)]
    if conf.method == "unsupervised":
        losses = np.power(np.concatenate(list(x_input.as_numpy_iterator())) - y_prob, 2)
        mean = np.mean(losses)
        std = np.std(losses)
        threshold = mean + std
    np.save(conf.threshold_path, threshold)


if __name__ == '__main__':
    import  tensorflow as tf
    threshold = np.load(
        "/home/meph103/PycharmProjects/anomaly_detection/trials/unsupervised/20231231075300/model/threshold.npy")
    model = keras.models.load_model(
        "/home/meph103/PycharmProjects/anomaly_detection/trials/unsupervised/20231231075300/model/model.keras")

    conf = Config(trial_id=20231231075300, method="unsupervised", mode="validation")
    train_ds = get_dataset(directory=conf.train_dir, conf=conf, shuffle=False, get_labels=False)
    train_ds = tf.data.Dataset.zip(train_ds, train_ds)
    calculate_threshold(model, train_ds, conf)
    # np.mean(loss):0.00039602164 0.00045541607
    # np.std(loss):0.002014422 0.0022966806
    ic(threshold)
