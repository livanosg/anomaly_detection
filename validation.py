import os

import numpy as np
import tensorflow as tf
from keras.losses import MeanAbsoluteError
from sklearn.metrics import precision_recall_curve

from datasets import get_dataset
from metrics import plot_metrics


def get_outputs_labels(dataset, model):
    outputs, labels = [], []
    [(outputs.append(model.predict(image, verbose=False)), labels.append(label)) for (image, label) in dataset]
    return np.concatenate(outputs), np.concatenate(labels)


def get_reconstruction_errors_labels(dataset, model):
    rec_errs = []
    labels = []
    mae = MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    for image, label in dataset:
        output = model.predict(image, verbose=False)
        rec_errs.append(np.mean(mae(image, output), axis=tuple(range(1, output.ndim - 1))))
        labels.append(label)
    return np.concatenate(rec_errs), np.concatenate(labels)


def f1_threshold(precision, recall, thresholds):
    return thresholds[np.argmax(2 * (precision * recall) / (precision + recall))]


def eu_threshold(precision, recall, thresholds):
    euclidean_distance = np.sqrt(np.sum(np.square(1 - np.column_stack([precision, recall])), axis=-1))
    return thresholds[np.argmin(euclidean_distance)]


def get_pred_labels(dataset_name, model, conf):
    if conf.model_type == "classifier":
        y_out, y_true = get_outputs_labels(get_dataset(dataset_name, conf=conf), model)
        y_out = y_out[..., conf.pos_label]
    elif conf.model_type == "autoencoder":
        rec_err, y_true = get_reconstruction_errors_labels(get_dataset(dataset_name, conf=conf), model)
        y_out = rec_err
    else:
        raise ValueError(f"Unknown type: {conf.model_type} ")
    y_true = np.argmax(y_true, axis=-1)
    return y_out, y_true


def get_threshold(dataset_name, model, conf, save=True):
    y_out, y_true = get_pred_labels(dataset_name, model, conf)
    threshold = 0.5
    if conf.threshold_type:
        prec, rec, thresh = precision_recall_curve(y_true=y_true, probas_pred=y_out, pos_label=conf.pos_label)
        if conf.threshold_type == "f1":
            threshold = f1_threshold(prec, rec, thresh)
        if conf.threshold_type == "euclidean":
            threshold = eu_threshold(prec, rec, thresh)
    if save:
        np.save(os.path.join(conf.model_dir, "threshold.npy"), threshold)
    return threshold


def validate_model(dataset_name, model, threshold, conf, save):
    y_out, y_true = get_pred_labels(dataset_name, model, conf)
    y_pred = np.greater_equal(y_out, threshold)
    plot_metrics(title=dataset_name, y_true=y_true, y_prob=y_out, y_pred=y_pred, threshold=threshold,
                 conf=conf, save=save)


if __name__ == '__main__':
    precision = np.random.rand(100000000)
    recall = np.random.rand(100000000)
