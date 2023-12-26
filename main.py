import os
from datetime import datetime

import keras.models
import numpy as np

from config import TRIALS_DIR
from dataset import get_dataset, split_inputs_labels
from metrics import plot_history, plot_pr_cure, print_metrics, plot_confusion_matrix
from model import calculate_threshold
from training import model_training

if __name__ == '__main__':

    # trial_id = "latest"
    # mode = "validation"
    trial_id = None
    mode = "train"
    if trial_id is None:
        trial_id = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    elif trial_id == "latest":
        trial_id = sorted(os.listdir(TRIALS_DIR), key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        print("latest_trial_id", trial_id)

    trial_dir = os.path.join(TRIALS_DIR, str(trial_id))
    model_path = os.path.join(str(trial_dir), "model.keras")
    history_path = os.path.join(str(trial_dir), "history.csv")
    threshold_path = os.path.join(str(trial_dir), "threshold.npy")
    backup_dir = os.path.join(str(trial_dir), "backup")
    threshold = None
    if mode == "train":
        model = model_training(model_path=model_path,
                               backup_dir=backup_dir,
                               history_path=history_path)
        keras.models.save_model(model_path)

    if mode in ("train", "validation"):
        model = keras.models.load_model(model_path)

        x_val, y_val_true = split_inputs_labels(get_dataset(dataset="validation"))
        y_val_true = np.concatenate(list(y_val_true.as_numpy_iterator()))

        y_val_pred_prob = model.predict(x_val).squeeze()

        y_true = y_val_true[:, 1]
        y_pred_prob = y_val_pred_prob[:, 1]
        threshold = calculate_threshold(y_true, y_pred_prob)
        np.save(threshold_path, threshold)
        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)
        plot_history(history_path)
        plot_pr_cure(y_true, y_pred_prob)

        print_metrics(y_true, y_pred)
        plot_confusion_matrix(y_true, y_pred)
    if mode in ("train", "test"):
        model = keras.models.load_model(model_path)

        x_val, y_val_true = split_inputs_labels(get_dataset(dataset="validation"))
        y_val_true = np.concatenate(list(y_val_true.as_numpy_iterator()))

        y_val_pred_prob = model.predict(x_val).squeeze()

        y_true = y_val_true[:, 1]
        y_pred_prob = y_val_pred_prob[:, 1]
        threshold = np.load(threshold_path)
        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)
        plot_pr_cure(y_true, y_pred_prob)

        print_metrics(y_true, y_pred)
        plot_confusion_matrix(y_true, y_pred)

    elif mode == "predict":
        model = keras.models.load_model(model_path)

        x_val, y_val_true = split_inputs_labels(get_dataset(dataset="validation"))
        y_val_true = np.concatenate(list(y_val_true.as_numpy_iterator()))

        y_val_pred_prob = model.predict(x_val).squeeze()

        y_true = y_val_true[:, 1]
        y_pred_prob = y_val_pred_prob[:, 1]
        threshold = np.load(threshold_path)
        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)
        print(y_pred)
