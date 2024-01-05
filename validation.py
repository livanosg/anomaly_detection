import os

import numpy as np
from icecream import ic
from sklearn.metrics import precision_recall_curve
from metrics import plot_metrics


def get_outputs_labels(dataset, model):
    outputs, labels = [], []
    [(outputs.append(model.predict(image, verbose=False)), labels.append(label)) for (image, label) in dataset]
    return np.concatenate(outputs), np.concatenate(labels)


def get_reconstruction_errors_labels(dataset, model):
    rec_errs = []
    labels = []
    for image, label in dataset.unbatch().batch(1):
        output = model.evaluate(image, image, verbose=False)[0]
        rec_errs.append(output)
        labels.append(label)
    rec_errs = np.asarray(rec_errs)
    labels = np.argmax(np.concatenate(labels), axis=-1)
    return rec_errs, labels


def get_pred_labels(dataset, model, conf):
    y_out, y_true = get_outputs_labels(dataset=dataset, model=model)
    y_out = y_out[..., conf.pos_label]
    y_true = np.argmax(y_true, axis=-1)
    return y_out, y_true


def f1_threshold(precision, recall, thresholds):
    return thresholds[np.argmax(2 * (precision * recall) / (precision + recall))]


def eu_threshold(precision, recall, thresholds):
    euclidean_distance = np.sqrt(np.sum(np.square(1 - np.column_stack([precision, recall])), axis=-1))
    return thresholds[np.argmin(euclidean_distance)]


def get_threshold(dataset, model, conf, save=True):
    if conf.model_type == "autoencoder":
        rec_err, y_true = get_reconstruction_errors_labels(dataset=dataset, model=model)
        threshold = np.mean(rec_err)
    elif conf.model_type == "classifier":
        y_out, y_true = get_pred_labels(dataset, model, conf)
        prec, rec, thresh = precision_recall_curve(y_true=y_true, probas_pred=y_out, pos_label=conf.pos_label)
        if conf.threshold_type == "f1":
            threshold = f1_threshold(prec, rec, thresh)
        elif conf.threshold_type == "euclidean":
            threshold = eu_threshold(prec, rec, thresh)
        else:
            raise ValueError(f"Unknown threshold type: {conf.threshold_type}.")
    else:
        raise ValueError(f"Unknown model type: {conf.model_type}.")
    if save:
        np.save(os.path.join(conf.model_dir, "threshold.npy"), threshold)
    return threshold


def validate_model(dataset_name, dataset, model, threshold, conf, save):
    if conf.model_type == "classifier":
        y_out, y_true = get_pred_labels(dataset, model, conf)
        y_pred = np.greater_equal(y_out, threshold).astype(int)
    elif conf.model_type == "autoencoder":
        rec_err, y_true = get_reconstruction_errors_labels(dataset=dataset, model=model)
        y_pred = np.greater_equal(rec_err, threshold).astype(int)
        y_out = rec_err
    else:
        raise ValueError(f"Unknown model type {conf.model_type}.")
    plot_metrics(title=dataset_name, y_true=y_true, y_prob=y_out, y_pred=y_pred, threshold=threshold,
                 conf=conf, save=save)
