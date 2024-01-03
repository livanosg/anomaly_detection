import os
from contextlib import redirect_stdout

import keras.models
import numpy as np
import tensorflow as tf
from icecream import ic

from keras.metrics import MeanSquaredError, F1Score
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger, EarlyStopping

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy, MeanAbsoluteError

from datasets import split_inputs_labels
from model import supervised_anomaly_detector, unsupervised_anomaly_detector
from metrics import plot_metrics, plot_training_history, f1_threshold


def supervised_training(train_ds, val_ds, conf):
    if os.path.isfile(os.path.join(conf.model_dir, "model.keras")):
        model = keras.models.load_model(os.path.join(conf.model_dir, "model.keras"))
    else:
        model = supervised_anomaly_detector(input_shape=conf.input_shape)

    metrics = [F1Score(average="macro")]
    callbacks = [
        ModelCheckpoint(filepath=os.path.join(conf.model_dir, "model.keras"), save_best_only=True),
        # BackupAndRestore(backup_dir=os.path.join(conf.model_dir, "backup"), delete_checkpoint=False),
        CSVLogger(filename=os.path.join(conf.model_dir, "history.csv"), append=True),
        ReduceLROnPlateau(factor=0.5, patience=6, cooldown=5, monitor="val_loss", mode="min"),
        EarlyStopping(start_from_epoch=5, patience=10, restore_best_weights=True, monitor="val_loss", mode="min")
    ]
    if conf.label_type == "categorical":
        loss = CategoricalCrossentropy()  # CategoricalFocalCrossentropy()
    else:
        loss = BinaryCrossentropy()  # BinaryFocalCrossentropy()
    optimizer = Adam(learning_rate=conf.learning_rate)
    model.summary()
    with open(os.path.join(conf.model_dir, "summary.txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(x=train_ds, validation_data=val_ds, epochs=conf.epochs, callbacks=callbacks)
    plot_training_history(history.history, conf, save=True)
    model = keras.models.load_model(os.path.join(conf.model_dir, "model.keras"))

    probas, labels = get_predictions_labels(val_ds, model, conf)
    # pr_threshold(y_true=val_label, probas_pred=val_probas, conf=conf, save=True)
    f1_threshold(y_true=labels, probas_pred=probas, conf=conf, save=True)


def unsupervised_training(train_ds, val_ds, conf):
    train_norm = train_ds.unbatch().filter(lambda image, label: tf.math.equal(label, [0])[0])
    train_norm = train_norm.batch(conf.batch_size)
    train_norm_inpt, _ = split_inputs_labels(train_norm)

    train_normal_data = tf.data.Dataset.zip(train_norm_inpt, train_norm_inpt).prefetch(buffer_size=tf.data.AUTOTUNE)
    train_normal_data = train_normal_data

    val_input, _ = split_inputs_labels(val_ds)
    val_data = tf.data.Dataset.zip(val_input, val_input)

    model = unsupervised_anomaly_detector(input_shape=conf.input_shape)

    metrics = [MeanSquaredError()]
    callbacks = [ModelCheckpoint(filepath=os.path.join(conf.model_dir, "model.keras"), save_best_only=True),
                 BackupAndRestore(os.path.join(conf.model_dir, "backup"), delete_checkpoint=False),
                 CSVLogger(os.path.join(conf.model_dir, "history.csv"), append=True),
                 ReduceLROnPlateau(patience=10, cooldown=5, monitor='loss', mode="min"),
                 EarlyStopping(patience=30, restore_best_weights=True, monitor='loss', mode="min")]

    loss = MeanAbsoluteError()
    optimizer = Adam()  # learning_rate=conf.learning_rate)
    model.summary()
    with open(os.path.join(conf.model_dir, "summary"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(x=train_normal_data, validation_data=val_data, epochs=conf.epochs, callbacks=callbacks)
    plot_training_history(history.history, conf, save=True)

    train_reconstruction_errors = []
    for train_input, _ in train_ds:
        train_reconstruction_errors.append(reconstruction_error(inpt=train_input, model=model))

    train_reconstruction_errors = np.concatenate(train_reconstruction_errors)
    threshold = train_reconstruction_errors.mean() + train_reconstruction_errors.std()
    # if save:
    np.save(os.path.join(conf.model_dir, "threshold.npy"), threshold)


def validate_supervised_model(title, dataset, model, threshold, conf, save=False):
    probas, labels = get_predictions_labels(dataset, model, conf)
    plot_metrics(title=title, y_true=labels, y_prob=probas, threshold=threshold, conf=conf, save=save)


def validate_unsupervised_model(title, dataset, model, threshold, conf, save=False):
    labels = []
    rec_errors = []
    for inpt, label in dataset:
        rec_errors.append(reconstruction_error(inpt, model))
        labels.append(label)
    rec_errors = np.concatenate(rec_errors)
    labels = np.concatenate(labels)
    plot_metrics(title=title, y_true=labels, y_prob=rec_errors, threshold=threshold, conf=conf, save=save)


def reconstruction_error(inpt, model):
    return np.mean(np.absolute(inpt.numpy() - model.predict(inpt, verbose=0)), axis=(1, 2, 3))


def get_predictions_labels(dataset, model, conf):
    images, labels = split_inputs_labels(dataset)
    labels = np.concatenate(list(labels.as_numpy_iterator()))
    probas = model.predict(images)
    if conf.label_type == "categorical":
        probas = probas[..., 1]
        labels = labels[..., 1]
    return probas, labels