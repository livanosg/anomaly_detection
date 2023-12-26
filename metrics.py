import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import CLASS_NAMES


def plot_history(history_path):
    history_df = pd.read_csv(history_path)
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
    plt.show()
    # plt.savefig()


def plot_pr_cure(y_true, y_pred_prob):
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob)}")

    display = PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_pred_prob, name="CNN",
                                                      plot_chance_level=True)
    display.ax_.set_title('Precision-Recall curve')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot()
    plt.show()


def print_metrics(y_true, y_pred):
    print(f"Accuracy Score Before Thresholding: {accuracy_score(y_true, y_pred)}")
    print(f"Precision Score Before Thresholding: {precision_score(y_true, y_pred)}")
    print(f"Recall Score Before Thresholding: {recall_score(y_true, y_pred)}")
    print(f"F1 Score Before Thresholding: {f1_score(y_true, y_pred)}")
    print(classification_report(y_true=y_true, y_pred=y_pred,target_names=CLASS_NAMES))
    # Note that in binary classification, recall of the positive class is also known as "sensitivity";
    # recall of the negative class is "specificity"
    # classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
