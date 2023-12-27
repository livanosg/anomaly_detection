import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import CLASS_NAMES


def plot_history(model_path):
    """
    Plots the training history of a model.

    Args:
        model_path (str): Path to the directory containing the model's training history.
    """

    history_df = pd.read_csv(os.path.join(model_path, "history.csv"))
    epochs_trained = range(len(history_df))
    fig, axs = plt.subplots(4)
    idx = 0
    for metric in history_df.columns:
        if metric.startswith("val_"):
            continue
        axs[idx].plot(epochs_trained, history_df[metric], label=f"Training {metric.capitalize()}")
        if metric == "lr":
            axs[idx].set_yscale("log")
        else:
            axs[idx].plot(epochs_trained, history_df["val_" + metric], label=f"Validation {metric.capitalize()}")
        axs[idx].legend()
        axs[idx].set_title(f"{metric.capitalize()}")
        if metric.endswith("precision") or metric.endswith("recall"):
            axs[idx].set_ylim(0, 1)
        if metric.endswith("loss"):
            axs[idx].set_ylim(0, history_df[metric].mean() + history_df[metric].std())
        idx += 1
    fig.savefig(os.path.join(model_path, "training_history.png"))
    plt.show()


def plot_pr_cure(y_true, y_pred_prob, metrics_dir):
    """
    Plots the precision-recall curve for a model's predictions.

    Args:
        y_true (np.ndarray): True labels.
        y_pred_prob (np.ndarray): Predicted probabilities.
        metrics_dir (str): Directory to save the precision-recall curve plot.
    """
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob)}")

    display = PrecisionRecallDisplay.from_predictions(y_true=y_true,
                                                      y_pred=y_pred_prob,
                                                      name="CNN",
                                                      plot_chance_level=True)
    # display.plot()
    pr_path = os.path.join(metrics_dir, "pr_curve.png")
    os.makedirs(os.path.dirname(pr_path), exist_ok=True)
    display.figure_.savefig(pr_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, metrics_dir):
    """
    Plots the confusion matrix for a model's predictions.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        metrics_dir (str): Directory to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=CLASS_NAMES)
    cm_path = os.path.join(metrics_dir, "confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    display.plot()
    display.figure_.savefig(cm_path)
    plt.show()


def print_metrics(y_true, y_pred, metrics_dir):
    """
    Prints various classification metrics for model evaluation.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        metrics_dir (str): Directory to save the evaluation metrics.
    """
    metrics = {"accuracy": [accuracy_score(y_true, y_pred)],
               "precision": [precision_score(y_true, y_pred)],
               "recall": [recall_score(y_true, y_pred)],
               "f1": [f1_score(y_true, y_pred)]
               }

    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_file = os.path.join(metrics_dir, "metrics.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    metrics_df.to_csv(metrics_file, index=False)

    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_file = os.path.join(metrics_dir, "report.csv")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report_df.to_csv(report_file, index=False)
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=CLASS_NAMES))

    # Note that in binary classification, recall of the positive class is also known as "sensitivity";
    # recall of the negative class is "specificity"
    # classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
