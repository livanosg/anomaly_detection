import os
from datetime import datetime

import keras.models
import numpy as np

from config import TRIALS_DIR
from training import model_training

from dataset import get_dataset, split_inputs_labels
from metrics import plot_history, plot_pr_cure, print_metrics, plot_confusion_matrix
from model import calculate_threshold

if __name__ == '__main__':

    # trial_id = "latest"
    # mode = "test"
    trial_id = None
    mode = "train"
    if trial_id is None:
        trial_id = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    elif trial_id == "latest":
        trial_id = sorted(os.listdir(TRIALS_DIR), key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        print("latest_trial_id", trial_id)

    trial_dir = os.path.join(TRIALS_DIR, str(trial_id))
    metrics_dir = os.path.join(str(trial_dir), "metrics")
    model_dir = os.path.join(str(trial_dir), "model")

    if mode == "train":
        model = model_training(model_dir=model_dir)
        # model.save_model(model_dir)

    if mode in ("train", "validation"):
        model = keras.models.load_model(os.path.join(model_dir, "model.keras"))

        x_val, y_val_true = split_inputs_labels(get_dataset(dataset="validation"))
        y_val_true = np.concatenate(list(y_val_true.as_numpy_iterator()))

        y_val_pred_prob = model.predict(x_val).squeeze()

        y_true = y_val_true[:, 1]
        y_pred_prob = y_val_pred_prob[:, 1]

        threshold = calculate_threshold(y_true, y_pred_prob)
        np.save(os.path.join(model_dir, "threshold.npy"), threshold)

        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)

        plot_history(model_dir)
        plot_pr_cure(y_true, y_pred_prob, os.path.join(metrics_dir, "validation"))
        print_metrics(y_true, y_pred, os.path.join(metrics_dir, "validation"))
        plot_confusion_matrix(y_true, y_pred, os.path.join(metrics_dir, "validation"))

    if mode in ("train", "test"):
        model = keras.models.load_model(os.path.join(model_dir, "model.keras"))
        threshold = np.load(os.path.join(model_dir, "threshold.npy"))

        x_val, y_val_true = split_inputs_labels(get_dataset(dataset="test"))
        y_val_true = np.concatenate(list(y_val_true.as_numpy_iterator()))

        y_val_pred_prob = model.predict(x_val).squeeze()

        y_true = y_val_true[:, 1]
        y_pred_prob = y_val_pred_prob[:, 1]
        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)

        plot_pr_cure(y_true, y_pred_prob, os.path.join(metrics_dir, "test"))
        print_metrics(y_true, y_pred, os.path.join(metrics_dir, "test"))
        plot_confusion_matrix(y_true, y_pred, os.path.join(metrics_dir, "test"))

    elif mode == "predict":
        model = keras.models.load_model(os.path.join(model_dir, "model.keras"))
        threshold = np.load(os.path.join(model_dir, "threshold.npy"))

        x_input = get_dataset(dataset="validation", get_labels=False)

        y_pred_prob = model.predict(x_input).squeeze()
        y_pred_prob = y_pred_prob[:, 1]
        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)
        print(y_pred)
