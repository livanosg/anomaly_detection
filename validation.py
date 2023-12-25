import os
from glob import glob

import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from matplotlib import pyplot as plt

from config import CLASS_NAMES, VALIDATION_DIR
from hyper_parameters import INPUT_SHAPE
from local_utils import get_latest_trial_id, trial_dirs
from model import classifier


def calculate_threshold(y_true, probas_pred, threshold_file):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=probas_pred)
    threshold_idx = np.argmax(np.sqrt(np.power(precision, 2) + np.power(recall, 2)))  # fix
    threshold = thresholds[threshold_idx]
    if not os.path.isfile(threshold_file):
        np.save(str(threshold_file), threshold)
        print(f"Threshold saved to : {threshold_file}")
    return threshold


def validation_metrics(y_true, y_output, history_file_name, name):
    if os.path.isfile(history_file_name):
        history = pd.read_csv(str(history_file_name))
        epochs_trained = range(len(history))
    else:
        raise FileNotFoundError(f"History file {history_file_name} does not exist")
    nrows = 1
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    plt.show()

    display = PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_output, name="CNN",
                                                      ax=axs[0],
                                                      plot_chance_level=True)

    display.plot(name="Precision-Recall curve")
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, float(y[45]) + 0.02))

    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    if os.path.isfile(history_file_name):
        idx = 0
        for metric in history.columns:
            if metric.startswith("val_"):
                continue
            print(metric)
            metric_values = history[metric]
            axs[1 + idx].plot(epochs_trained, metric_values, label=f"Training {metric.capitalize()}")
            if not metric == "lr":
                val_metric_values = history["val_" + metric]
                axs[1 + idx].plot(epochs_trained, val_metric_values, label=f"Validation {metric.capitalize()}")
            axs[1 + idx].legend()
            axs[1 + idx].set_title(f"{metric.capitalize()}")
            idx += 1
    else:
        print(f"History file {history_file_name} does not exist")

    fig.show()
    # plt.savefig()


def validate_model(model, data_dir, history_path, threshold_path, name):

    data_size = len(glob(os.path.join(data_dir, "**", "*.png"), recursive=True))
    data = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=data_size,
        image_size=INPUT_SHAPE[:-1])

    x_input = data.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_output = model.predict(x_input)
    y_true = data.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    y_true = np.asarray([label for label in y_true.as_numpy_iterator()])

    y_output = y_output.squeeze()
    y_true = y_true.squeeze()
    if os.path.isfile(threshold_path):
        threshold = np.load(threshold_path)
    else:
        threshold = calculate_threshold(y_true, y_output, threshold_path)

    validation_metrics(y_true, y_output, history_file_name=history_path, name=name)

if __name__ == '__main__':
    trial_id = get_latest_trial_id()
    trial_dir, model_path, history_path, threshold_path = trial_dirs(trial_id)
    model = keras.models.load_model(model_path)

    validate_model(model=model, data_dir=VALIDATION_DIR, history_path=history_path, threshold_path=threshold_path, name="validation")
