import os

import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger

from config import BATCH_SIZE, INPUT_SHAPE, LEARNING_RATE, EPOCHS
from metrics import plot_metrics
from project_manage import Project
from datasets import DatasetHandler
from model import calculate_threshold, ModelHandler

if __name__ == '__main__':
    pr = Project(trial_id="latest")
    mode = "validation"
    if mode == "train":
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
        global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
        with strategy.scope():
            mh = ModelHandler(input_shape=INPUT_SHAPE)
            train_dh = DatasetHandler("train", batch_size=global_batch_size, get_labels=True,
                                      image_shape=INPUT_SHAPE[:-1])
            val_dh = DatasetHandler("validation", batch_size=global_batch_size, get_labels=True,
                                    image_shape=INPUT_SHAPE[:-1])
        with open(os.path.join(pr.metrics_dir, 'summary.txt'), 'w') as f:
            mh.model.summary(print_fn=lambda x: f.write(x + '\n'))
        metrics = [keras.metrics.Precision(), keras.metrics.Recall()]
        callbacks = [EarlyStopping(patience=20, restore_best_weights=True),
                     ReduceLROnPlateau(patience=10, cooldown=5),
                     ModelCheckpoint(filepath=pr.model_path, save_best_only=True),
                     BackupAndRestore(pr.backup_dir, delete_checkpoint=False),
                     CSVLogger(os.path.join(pr.metrics_dir, "history.csv"), append=True)]

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss = keras.losses.CategoricalFocalCrossentropy()

        mh.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = mh.model.fit(x=train_dh.dataset, validation_data=val_dh.dataset, epochs=EPOCHS, callbacks=callbacks)
        mh.model.save(os.path.join(pr.model_dir, "model.keras"))

        x_input, y_true = val_dh.split_inputs_labels()
        y_true = np.concatenate(list(y_true.as_numpy_iterator()))
        y_prob = mh.model.predict(x_input).squeeze()

        threshold = calculate_threshold(y_true[:, 1], y_prob[:, 1])

        np.save(os.path.join(pr.model_dir, "threshold.npy"), threshold)
        np.save(os.path.join(pr.metrics_dir, "y_true.npy"), y_true)
        np.save(os.path.join(pr.metrics_dir, "y_prob.npy"), y_prob)
    if mode in ("train", "validation"):
        model = keras.models.load_model(os.path.join(pr.model_dir, "model.keras"))
        threshold = np.load(os.path.join(pr.model_dir, "threshold.npy"))
        history_df = pd.read_csv(os.path.join(pr.metrics_dir, "history.csv"))
        y_true = np.load(os.path.join(pr.metrics_dir, "y_true.npy"))
        y_prob = np.load(os.path.join(pr.metrics_dir, "y_prob.npy"))

        plot_metrics(y_true[:, 1], y_prob[:, 1], threshold, history_df=history_df)

    if mode in ("train", "test"):
        model = keras.models.load_model(os.path.join(pr.model_dir, "model.keras"))
        threshold = np.load(os.path.join(pr.model_dir, "threshold.npy"))
        history_df = pd.read_csv(os.path.join(pr.metrics_dir, "history.csv"))
        test_dh = DatasetHandler("test", batch_size=100, get_labels=True, image_shape=INPUT_SHAPE[:-1])
        x_input, y_true = test_dh.split_inputs_labels()
        y_true = np.concatenate(list(y_true.as_numpy_iterator()))
        y_prob = model.predict(x_input).squeeze()
        plot_metrics(y_true, y_prob, threshold, history_df)
    elif mode == "predict":
        model = keras.models.load_model(pr.model_path)
        threshold = np.load(os.path.join(pr.metrics_dir, "threshold.npy"))
        test_dh = DatasetHandler("test", batch_size=200, get_labels=False, image_shape=INPUT_SHAPE[:-1])
        y_pred_prob = model.predict(test_dh.dataset).squeeze()
        y_pred_prob = y_pred_prob[:, 1]
        y_pred = np.greater_equal(y_pred_prob, threshold).squeeze().astype(int)
        print(y_pred)
