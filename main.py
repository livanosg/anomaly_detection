import os
import fire
import numpy as np
import keras.models
import pandas as pd

from project_manage import Config
from training import training_pipeline
from model import supervised_anomaly_detector, unsupervised_anomaly_detector
from datasets import get_dataset, split_inputs_labels
from metrics import plot_metrics, plot_training_summary

if __name__ == '__main__':
    conf = fire.Fire(Config)
    # Config(unsupervised=False)
    if conf.mode == "train":

        if conf.method == "supervised":
            get_labels = True
        else:
            get_labels = False
        train_ds = get_dataset(directory=conf.train_dir, conf=conf, shuffle=True, get_labels=get_labels)
        val_ds = get_dataset(directory=conf.val_dir, conf=conf, shuffle=True, get_labels=get_labels)

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
    y_true = np.load(conf.y_true)
    y_prob = np.load(conf.y_prob)
    threshold = np.load(conf.threshold_path)

    if conf.mode in ("train", "validation"):
        plot_metrics(title="Validation", y_true=y_true, y_prob=y_prob, threshold=threshold)
    model = keras.models.load_model(conf.model_path)
    if conf.mode in ("train", "test"):
        test_ds = get_dataset(directory=conf.test_dir, conf=conf, shuffle=False)
        x_input, y_true = split_inputs_labels(test_ds)
        y_true = np.concatenate(list(y_true.as_numpy_iterator()))
        y_prob = model.predict(x_input).squeeze()
        plot_metrics(title="Test", y_true=y_true, y_prob=y_prob, threshold=threshold)
    if conf.mode == "predict":
        threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
        test_ds = get_dataset(directory=conf.test_dir, shuffle=False, conf=conf)
        y_pred_prob = model.predict(test_ds).squeeze()
        if conf.method == "supervised":
            y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)
            map(lambda x: print("Anomaly" if x == 1 else "Not Anomaly"), y_pred)
        if conf.method == "unsupervised":
            y_pred = model.evaluate(y_pred_prob, test_ds)
            map(lambda x: print("Anomaly" if x == 1 else "Not Anomaly"), y_pred)
    exit(0)
