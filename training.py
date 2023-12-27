import os

import tensorflow as tf
import keras.losses
import pandas as pd

from config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from model import get_model
from datasets import get_dataset


def model_training(model_dir):
    """
    Trains a convolutional neural network (CNN) model with specified hyperparameters.

    Args:
        model_dir (str): Directory to save the trained model and related files.

    Returns:
        tf.keras.Model: The trained CNN model.
    """
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
                               loss=keras.losses.CategoricalFocalCrossentropy(),
                               metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

    os.makedirs(os.path.join(model_dir), exist_ok=True)
    with open(os.path.join(model_dir, 'summary.txt'), 'w') as f:
        training_model.summary(print_fn=lambda x: f.write(x + '\n'))

    training_history = training_model.fit(x=train_data,
                                          validation_data=validation_data,
                                          epochs=EPOCHS,
                                          callbacks=[
                                              keras.callbacks.EarlyStopping(patience=20,
                                                                            restore_best_weights=True,
                                                                            verbose=0),
                                              keras.callbacks.ReduceLROnPlateau(patience=10,
                                                                                cooldown=5,
                                                                                verbose=0),
                                              keras.callbacks.ModelCheckpoint(
                                                  filepath=os.path.join(model_dir, "model.keras"),
                                                  save_best_only=True,
                                                  verbose=0),
                                              keras.callbacks.BackupAndRestore(os.path.join(model_dir, "backup"),
                                                                               delete_checkpoint=False)],
                                          verbose=0)

    if len(training_history.history["loss"]) > 0:
        pd.DataFrame.from_dict(training_history.history).to_csv(os.path.join(model_dir, "history.csv"), index=False)

    return training_model


if __name__ == '__main__':
    for k, v in tf.keras.metrics.__dict__.items():
        if not k[0].isupper() and not k[0] == "_":
            print(k)
