import os

import numpy as np
import tensorflow as tf
from icecream import ic
from keras.optimizers import Adam
from keras.losses import BinaryFocalCrossentropy, CategoricalFocalCrossentropy, MeanSquaredError
from keras.metrics import Precision, Recall, MeanAbsoluteError
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger, EarlyStopping

from datasets import split_inputs_labels
from model import supervised_anomaly_detector, unsupervised_anomaly_detector
from metrics import plot_metrics, plot_training_history, pr_threshold


def supervised_training(train_ds, val_ds, conf):
    train_input, train_label = split_inputs_labels(train_ds)
    train_label = np.concatenate(list(train_label.as_numpy_iterator()))

    model = supervised_anomaly_detector(input_shape=conf.input_shape)

    metrics = [Precision(), Recall()]
    callbacks = [ModelCheckpoint(filepath=os.path.join(conf.model_dir, "model.keras"), save_best_only=True),
                 BackupAndRestore(backup_dir=os.path.join(conf.model_dir, "backup"), delete_checkpoint=False),
                 CSVLogger(filename=os.path.join(conf.model_dir, "history.csv"), append=True),
                 ReduceLROnPlateau(factor=0.5, patience=10, cooldown=5),
                 EarlyStopping(patience=20, restore_best_weights=True)]
    if conf.label_type == "categorical":
        loss = CategoricalFocalCrossentropy()
    else:
        loss = BinaryFocalCrossentropy()
    optimizer = Adam(learning_rate=conf.learning_rate)
    print(model.summary())

    with open(os.path.join(conf.model_dir, "summary.txt"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(x=train_ds, validation_data=val_ds, epochs=conf.epochs, callbacks=callbacks)
    plot_training_history(history.history, conf, save=True)

    train_probas_pred = model.predict(train_input).squeeze()
    if conf.label_type == "categorical":
        print(train_label.shape)
        train_label = train_label[..., 1]
        train_probas_pred = train_probas_pred[..., 1]
    pr_threshold(y_true=train_label, probas_pred=train_probas_pred, conf=conf, save=True)


def unsupervised_training(train_ds, val_ds, conf):
    train_normal = train_ds.unbatch().filter(lambda image, label: tf.math.equal(label, [0])[0])
    train_normal = train_normal.batch(conf.batch_size)
    train_normal_input, train_normal_label = split_inputs_labels(train_normal)

    train_normal_data = tf.data.Dataset.zip(train_normal_input, train_normal_input).prefetch(
        buffer_size=tf.data.AUTOTUNE)
    train_normal_data = train_normal_data

    val_input, val_label = split_inputs_labels(val_ds)
    val_data = tf.data.Dataset.zip(val_input, val_input)

    model = unsupervised_anomaly_detector(input_shape=conf.input_shape)

    metrics = [MeanAbsoluteError()]
    callbacks = [ModelCheckpoint(filepath=os.path.join(conf.model_dir, "model.keras"), save_best_only=True),
                 BackupAndRestore(os.path.join(conf.model_dir, "backup"), delete_checkpoint=False),
                 CSVLogger(os.path.join(conf.model_dir, "history.csv"), append=True),
                 ReduceLROnPlateau(patience=10, cooldown=5, monitor='loss', mode="min"),
                 EarlyStopping(patience=30, restore_best_weights=True, monitor='loss', mode="min")]

    loss = MeanSquaredError()
    optimizer = Adam(learning_rate=conf.learning_rate)
    print(model.summary())
    with open(os.path.join(conf.model_dir, "summary"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x=train_normal_data, validation_data=val_data, epochs=conf.epochs, callbacks=callbacks)

    train_reconstruction_errors = []
    for train_input, _ in train_ds:
        train_reconstruction_errors.append(reconstruction_error(inpt=train_input, model=model))

    train_reconstruction_errors = np.concatenate(train_reconstruction_errors)
    ic(train_reconstruction_errors)
    ic(train_reconstruction_errors.mean())
    ic(train_reconstruction_errors.std())

    threshold = train_reconstruction_errors.mean() + train_reconstruction_errors.std()
    # if save:
    np.save(os.path.join(conf.model_dir, "threshold.npy"), threshold)


def validate_supervised_model(title, dataset, model, threshold, conf, save=False):
    inpts, labels = split_inputs_labels(dataset)
    labels = np.concatenate(list(labels.as_numpy_iterator()))
    probas = model.predict(inpts).squeeze()
    if conf.label_type == "categorical":
        labels = labels[..., 1]
        probas = probas[..., 1]
    ic(probas.mean())
    ic(probas.std())
    preds = np.greater_equal(probas, threshold).astype(int)

    plot_metrics(title=title, y_true=labels, y_prob=probas, y_pred=preds, threshold=threshold, conf=conf, save=save)


def validate_unsupervised_model(title, dataset, model, threshold, conf, save=False):
    labels = []
    rec_errors = []
    for inpt, label in dataset:
        rec_errors.append(reconstruction_error(inpt, model))
        labels.append(label)
    rec_errors = np.concatenate(rec_errors)
    labels = np.concatenate(labels)
    preds = np.greater_equal(rec_errors, threshold).astype(int)
    plot_metrics(title=title, y_true=labels, y_prob=rec_errors, y_pred=preds, threshold=threshold, conf=conf, save=save)


def reconstruction_error(inpt, model):
    return np.mean(np.absolute(inpt.numpy() - model.predict(inpt, verbose=0)), axis=(1, 2, 3))
