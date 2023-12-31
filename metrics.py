import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
from config import CLASS_NAMES


def plot_metrics(title, y_true, y_prob, threshold, method):
    plt.style.use("ggplot")
    mpl.rcParams["figure.dpi"] = 200
    fontsize = 7
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle(title, fontsize=fontsize * 2)
    y_pred = np.greater_equal(y_prob, threshold).squeeze().astype(int)
    axs[0, 0].set_title("Precision Recall curve")
    PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_prob, ax=axs[0, 0], plot_chance_level=True)
    axs[0, 0].legend(loc="lower left", fontsize=fontsize, shadow=True)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlim(0, 1)

    axs[0, 1].set_title("Calibration curve")
    CalibrationDisplay.from_predictions(y_true=y_true, y_prob=y_prob, ax=axs[0, 1])
    axs[0, 1].legend(loc="lower right", fontsize=fontsize, shadow=True)
    # axs[0, 1].set_ylim(0, 1.1)
    # axs[0, 1].set_xlim(0, 1.1)

    if method == "supervised":
        subtitle = "Probability"
    else:
        subtitle = "Loss"
    axs[1, 0].set_title(f"{subtitle} distribution")
    axs[1, 0].hist([y_prob[y_true == 0], y_prob[y_true == 1]])
    axs[1, 0].legend(CLASS_NAMES, loc="upper left", fontsize=fontsize, shadow=True)
    axs[1, 0].set_ylabel("N Samples")
    axs[1, 0].set_xlabel(f"Output {subtitle}")

    axs[1, 1].set_title("Confusion matrix")
    ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, ax=axs[1, 1], display_labels=CLASS_NAMES)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    history_df = pd.DataFrame.from_dict(history)
    fontsize = 7
    metrics = [metric for metric in history_df.columns if not (metric.startswith("val_") or metric.startswith("epoch"))]
    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig2, axs2 = plt.subplots(ncols=cols, nrows=rows, figsize=(10, 10))
    for metric, ax2 in zip(metrics, axs2.reshape(-1)):
        ax2.set_title(metric.capitalize())
        ax2.plot(metric, data=history_df)
        if metric == "lr":
            ax2.legend(loc="upper right", fontsize=fontsize, shadow=True)
            ax2.set_ylim(history_df[metric].min() * 0.1, history_df[metric].max() * 10)
            ax2.set_yscale("log")
        if "val_" + metric in history_df.columns:
            ax2.plot(history_df["epoch"], "val_" + metric, data=history_df)
        ax2.legend(loc="lower left", fontsize=fontsize, shadow=True)
    plt.tight_layout()
    plt.show()
