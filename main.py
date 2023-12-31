import os

import fire
import numpy as np
import tensorflow as tf
import keras.models

from project_manage import Config
from training import training_pipeline
from model import supervised_anomaly_detector, unsupervised_anomaly_detector
from datasets import get_dataset, split_inputs_labels
from metrics import plot_metrics

conf = fire.Fire(Config)

if __name__ == '__main__':
    conf.mode = "train"
    with conf.strategy.scope():
        if conf.mode == "train":
            train_ds = get_dataset(mode="train", shuffle=True, **conf.dataset_conf)
            if conf.method == "unsupervised":
                train_ds = train_ds.unbatch().filter(lambda x, y: tf.equal(y, 0), name="Keep_normal_class")
                train_ds = train_ds.batch(conf.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            x_train, y_train = split_inputs_labels(train_ds)

        if conf.mode in ("train", "validation"):
            val_ds = get_dataset(mode="validation", shuffle=False, **conf.dataset_conf)
            x_val, y_val = split_inputs_labels(val_ds)
        if conf.mode in ("train", "test"):
            test_ds = get_dataset(mode="test", shuffle=False, **conf.dataset_conf)
            x_test, y_test = split_inputs_labels(test_ds)

    if conf.mode == "train":
        with conf.strategy.scope():
            if conf.method == "supervised":
                model = supervised_anomaly_detector(input_shape=conf.input_shape)
                train_data = train_ds
                val_data = val_ds
            elif conf.method == "unsupervised":
                model = unsupervised_anomaly_detector(input_shape=conf.input_shape)
                train_data = tf.data.Dataset.zip(x_train, x_train)
                val_data = tf.data.Dataset.zip(x_val, x_val)
            else:
                raise ValueError(f"Unknown method {conf.method}")
            training_pipeline(model=model, train_data=train_data, val_data=val_data, **conf.train_conf)

    threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
    model = keras.models.load_model(os.path.join(conf.model_dir, "model.keras"))

    if conf.mode in ("train", "validation"):
        y_val = np.concatenate(list(y_val.as_numpy_iterator()))
        output = model.predict(x_val).squeeze()
        if conf.method == "unsupervised":
            losses = np.power(np.concatenate(list(x_val.as_numpy_iterator())) - output, 2)
            reconstruction_error = np.mean(losses, axis=tuple(range(1, len(losses.shape))))
            y_prob = reconstruction_error
        else:
            y_prob = output
        plot_metrics(title="Validation", y_true=y_val, y_prob=y_prob, threshold=threshold, method=conf.method)

    if conf.mode in ("train", "test"):
        y_test = np.concatenate(list(y_test.as_numpy_iterator()))
        output = model.predict(x_test).squeeze()
        if conf.method == "unsupervised":
            losses = np.power(np.concatenate(list(x_test.as_numpy_iterator())) - output, 2)
            reconstruction_error = np.mean(losses, axis=tuple(range(1, len(losses.shape))))
            y_prob = reconstruction_error
        else:
            y_prob = output
        plot_metrics(title="Test", y_true=y_test, y_prob=y_prob, threshold=threshold, method=conf.method)

    if conf.mode == "predict":
        pred_ds = get_dataset(mode="all", shuffle=False, **conf.dataset_conf)
        x_all, y_all = split_inputs_labels(pred_ds)
        y_all = np.concatenate(list(y_all.as_numpy_iterator()))
        output = model.predict(x_all).squeeze()

        if conf.method == "unsupervised":
            losses = np.power(np.concatenate(list(x_all.as_numpy_iterator())) - output, 2)
            y_prob = np.mean(losses, axis=tuple(range(1, len(losses.shape))))
        else:
            y_prob = output
        y_pred = np.greater_equal(y_prob, threshold).squeeze().astype(int)
        map(lambda x: print("Anomaly" if x == 1 else "Not Anomaly"), y_pred)
    exit(0)
