import os

import numpy as np
import matplotlib as mpl
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, RocCurveDisplay, classification_report


def plot_metrics(title, y_true, y_prob, y_pred, threshold, conf, save=True):
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=conf.class_names, digits=3)
    print(report)
    plt.style.use("ggplot")
    mpl.rcParams["figure.dpi"] = 100
    fontsize = 7
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), tight_layout=True)
    fig.suptitle(title.capitalize(), fontsize=fontsize * 2)

    PrecisionRecallDisplay.from_predictions(y_true=y_true, y_pred=y_prob, ax=axs[0, 0],
                                            plot_chance_level=True, pos_label=conf.pos_label)
    axs[0, 0].set_title("Precision Recall curve")
    axs[0, 0].legend(loc="lower left", fontsize=fontsize, shadow=True)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlim(0, 1)

    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_prob, ax=axs[0, 1],
                                     plot_chance_level=True, label=conf.pos_label)
    axs[0, 1].set_title("ROC curve")
    axs[0, 1].legend(loc="lower right", fontsize=fontsize, shadow=True)

    axis_title = "Probability"
    if conf.model_type == "autoencoder":
        axis_title = "Reconstruction error"

    axs[1, 0].set_title(f"{axis_title} distribution")
    axs[1, 0].set_ylabel("N Samples")
    axs[1, 0].set_xlabel(f"{axis_title}")
    ic(np.asarray(y_prob)[np.logical_and(y_true == 0, y_pred == 0)])
    ic(len(np.logical_and(y_true == 0, y_pred == 1)))
    ic(len(np.logical_and(y_true == 1, y_pred == 0)))
    ic(len(np.logical_and(y_true == 1, y_pred == 1)))
    axs[1, 0].hist([y_prob[np.logical_and(y_true == 0, y_pred == 0)],
                    y_prob[np.logical_and(y_true == 0, y_pred == 1)],
                    y_prob[np.logical_and(y_true == 1, y_pred == 0)],
                    y_prob[np.logical_and(y_true == 1, y_pred == 1)]
                    ],
                   bins=50,
                   label=["normal", "miss_normal", "miss_anomaly", "anomaly"])
    axs[1, 0].axvline(threshold, ls="--", color="black", label="Threshold")
    axs[1, 0].legend(loc="upper right", fontsize=fontsize, shadow=True)  # ["threshold"] + conf.class_names

    ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, ax=axs[1, 1],
                                            display_labels=conf.class_names)
    axs[1, 1].set_title("Confusion matrix")
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    if save:
        cr_path = os.path.join(conf.metrics_dir, f"{title.lower()}_report.txt")
        with open(cr_path, "a" if os.path.isfile(cr_path) else "w") as file:
            file.write(report)
        fig.savefig(os.path.join(conf.metrics_dir, f"{title.lower()}_plots.png"))
        print(f"Metrics saved at {conf.metrics_dir}")
    else:
        plt.show()


def plot_training_history(history, conf, save=True):
    history_df = pd.DataFrame.from_dict(history)
    metrics = [metric for metric in history_df.columns if not (metric.startswith("val_") or metric.startswith("epoch"))]

    plt.style.use("ggplot")
    mpl.rcParams["figure.dpi"] = 200
    fontsize = 7

    cols = 2
    rows = max([1, int(np.ceil(len(metrics) / cols))])
    fig, axs = plt.subplots(ncols=cols, nrows=rows, tight_layout=True)
    fig.suptitle("Training graphs", fontsize=fontsize * 2)

    for metric, ax2 in zip(metrics, axs.reshape(-1)):
        ax2.set_title(metric.capitalize())
        ax2.plot(metric, data=history_df)
        if "val_" + metric in history_df.columns:
            ax2.plot("val_" + metric, data=history_df)
        ax2.legend(loc="lower left", fontsize=fontsize, shadow=True)
        ax2.set_xlabel("epochs")

        if metric == "lr":
            ax2.legend(loc="upper right", fontsize=fontsize, shadow=True)
            ax2.set_ylim(history_df[metric].min() * 0.1, history_df[metric].max() * 10)
            ax2.set_yscale("log")
    if save:
        fig.savefig(os.path.join(conf.model_dir, f"training_history.png"))
        print(f"Training information saved at: {conf.model_dir}")
    else:
        plt.show()
