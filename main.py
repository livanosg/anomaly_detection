import fire
import numpy as np
import tensorflow as tf
import keras.models
import pandas as pd
from icecream import ic

from project_manage import Config
from training import training_pipeline
from model import supervised_anomaly_detector, unsupervised_anomaly_detector
from datasets import get_dataset, split_inputs_labels
from metrics import plot_metrics, plot_training_summary

if __name__ == '__main__':
    conf = fire.Fire(Config)
    # Config(unsupervised=False)
    if conf.mode == "train":
        train_ds = get_dataset(directory=conf.train_dir, conf=conf, shuffle=True, get_labels=False)
        val_ds = get_dataset(directory=conf.val_dir, conf=conf, shuffle=True, get_labels=False)
        if conf.method == "unsupervised":
            # train_input, _ = split_inputs_labels(train_ds)
            # val_input, _ = split_inputs_labels(val_ds)

            train_ds = tf.data.Dataset.zip((train_ds, train_ds))
            val_ds = tf.data.Dataset.zip((val_ds, val_ds))
        with conf.strategy.scope():
            if conf.method == "supervised":
                model = supervised_anomaly_detector(conf=conf)
            else:
                model = unsupervised_anomaly_detector(conf=conf)
            with open(conf.summary_path, 'w') as f:
                model.build(input_shape=[None] + conf.input_shape)
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            history = training_pipeline(model=model, train_data=train_ds, val_data=val_ds, conf=conf)
        plot_training_summary(pd.read_csv(conf.history_path))
    threshold = np.load(conf.threshold_path)
    model = keras.models.load_model(conf.model_path)
    if conf.mode in ("train", "validation"):
        val_ds = get_dataset(directory=conf.val_dir, conf=conf, shuffle=False, get_labels=True)
        x_input, y_true = split_inputs_labels(val_ds)
        y_true = np.concatenate(list(y_true.as_numpy_iterator()))
        output = model.predict(x_input).squeeze()
        if conf.method == "unsupervised":
            losses = np.power(np.concatenate(list(x_input.as_numpy_iterator())) - output, 2)
            y_prob = ic(np.mean(losses, axis=tuple(range(1, len(losses.shape)))))
        else:
            y_prob = output
        plot_metrics(title="Validation", y_true=y_true, y_prob=y_prob, threshold=threshold, method=conf.method)

    if conf.mode in ("train", "test"):
        test_ds = get_dataset(directory=conf.test_dir, conf=conf, shuffle=False, get_labels=True)
        x_input, y_true = split_inputs_labels(test_ds)
        y_true = np.concatenate(list(y_true.as_numpy_iterator()))
        output = model.predict(x_input).squeeze()
        if conf.method == "unsupervised":
            losses = np.power(np.concatenate(list(x_input.as_numpy_iterator())) - output, 2)
            y_prob = np.mean(losses, axis=tuple(range(1, len(losses.shape))))
        else:
            y_prob = output
        plot_metrics(title="Test", y_true=y_true, y_prob=y_prob, threshold=threshold, method=conf.method)

    if conf.mode == "predict":
        input_images = get_dataset(directory=conf.test_dir, shuffle=False, conf=conf, get_labels=False)
        output = model.predict(input_images).squeeze()
        if conf.method == "unsupervised":
            losses = np.power(np.concatenate(list(input_images.as_numpy_iterator())) - output, 2)
            y_prob = np.mean(losses, axis=tuple(range(1, len(losses.shape))))
        else:
            y_prob = output

        y_pred = np.greater_equal(y_prob, threshold).squeeze().astype(int)
        map(lambda x: print("Anomaly" if x == 1 else "Not Anomaly"), y_pred)
    exit(0)
