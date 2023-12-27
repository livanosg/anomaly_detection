import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import CLASS_NAMES


def plot_history(model_path):
    history_df = pd.read_csv(os.path.join(model_path, "history.csv"))
    epochs_trained = range(len(history_df))
    fig, axs = plt.subplots(3)
    idx = 0
    for metric in history_df.columns:
        if metric.startswith("val_"):
            continue
        axs[idx].plot(epochs_trained, history_df[metric], label=f"Training {metric.capitalize()}")
        if not metric == "lr":
            axs[idx].plot(epochs_trained, history_df["val_" + metric], label=f"Validation {metric.capitalize()}")
        axs[idx].legend()
        axs[idx].set_title(f"{metric.capitalize()}")
        idx += 1
    fig.show()
    fig.savefig(os.path.join(model_path, "training_history.png"))


def plot_pr_cure(y_true, y_pred_prob, metrics_dir):
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob)}")

    display = PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_pred_prob, name="CNN",
                                                      plot_chance_level=True)
    display.plot()
    display.ax_.set_title('Precision-Recall curve')
    pr_path = os.path.join(metrics_dir, "pr_curve.png")
    os.makedirs(os.path.dirname(pr_path), exist_ok=True)
    display.figure_.savefig(pr_path)


def plot_confusion_matrix(y_true, y_pred, metrics_dir):
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    cm_path = os.path.join(metrics_dir, "confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    display.plot()
    display.figure_.savefig(cm_path)


def print_metrics(y_true, y_pred, metrics_dir):
    metrics = {"accuracy": [accuracy_score(y_true, y_pred)],
               "precision": [precision_score(y_true, y_pred)],
               "recall": [recall_score(y_true, y_pred)],
               "f1": [f1_score(y_true, y_pred)]
               }

    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_file = os.path.join(metrics_dir, "metrics.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    metrics_df.to_csv(metrics_file, index=False)

    print(f"Accuracy Score: {metrics['accuracy']}")
    print(f"Precision Score: {metrics['precision']}")
    print(f"Recall Score: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=CLASS_NAMES,output_dict=True)
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=CLASS_NAMES))
    report_df = pd.DataFrame(report).transpose()
    report_file = os.path.join(metrics_dir, "report.csv")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report_df.to_csv(report_file, index=False)

    # Note that in binary classification, recall of the positive class is also known as "sensitivity";
    # recall of the negative class is "specificity"
    # classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
