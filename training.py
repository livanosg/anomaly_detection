import os

import tensorflow as tf
import keras.losses
import pandas as pd

from config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from model import get_model
from dataset import get_dataset


def model_training(model_dir):
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    with strategy.scope():
        train_data = get_dataset("train", batch_size=global_batch_size)
        validation_data = get_dataset("validation", batch_size=global_batch_size)

        training_model = get_model()
        training_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                               loss=keras.losses.CategoricalFocalCrossentropy(), metrics=["accuracy"])
    training_history = training_model.fit(x=train_data,
                                          validation_data=validation_data,
                                          epochs=EPOCHS,
                                          callbacks=[
                                              # keras.callbacks.EarlyStopping(patience=20,
                                              #                               restore_best_weights=True,
                                              #                               verbose=1),
                                              keras.callbacks.ReduceLROnPlateau(patience=10,
                                                                                cooldown=5,
                                                                                verbose=1),
                                              keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, "model.keras"),
                                                                              save_best_only=True,
                                                                              verbose=1),
                                              keras.callbacks.BackupAndRestore(os.path.join(model_dir, "backup"),
                                                                               delete_checkpoint=False)],
                                          verbose=1)
    if len(training_history.history["loss"]) > 0:
        pd.DataFrame.from_dict(training_history.history).to_csv(os.path.join(model_dir, "history.csv"), index=False)

    return training_model
