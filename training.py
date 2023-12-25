import os
from datetime import datetime

import keras.losses
import pandas as pd
import tensorflow as tf

from config import TRIALS_DIR, TRAIN_DIR, VALIDATION_DIR, CLASS_NAMES, MODEL_NAME, HISTORY_NAME
from hyper_parameters import BATCH_SIZE, INPUT_SHAPE, SEED, LEARNING_RATE, EPOCHS
from model import model


def training(model_path, history_path):

    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    with strategy.scope():
        train_data = keras.utils.image_dataset_from_directory(
            directory=TRAIN_DIR,
            labels="inferred",
            label_mode="binary",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=global_batch_size,
            image_size=INPUT_SHAPE[:-1],
            shuffle=False,
            seed=SEED)

        validation_data = keras.utils.image_dataset_from_directory(
            directory=VALIDATION_DIR,
            labels="inferred",
            label_mode="binary",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=global_batch_size,
            image_size=INPUT_SHAPE[:-1],
            shuffle=False,
            seed=SEED)

        train_data = train_data.cache().shuffle(train_data.cardinality()).prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_data = validation_data.cache().shuffle(train_data.cardinality()).prefetch(buffer_size=tf.data.AUTOTUNE)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=keras.losses.BinaryCrossentropy())

    history = model.fit(x=train_data,
                        validation_data=validation_data,
                        epochs=EPOCHS,
                        callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
                                   keras.callbacks.ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
                                   keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True,
                                                                   verbose=1)], verbose=1)

    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_csv(history_path, index=False)

    # model.save(os.path.join(TRIALS_DIR, f"{TRIAL_ID}", "model.keras"))
    return model


if __name__ == '__main__':
    TRIAL_ID = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    model_path = os.path.join(TRIALS_DIR, TRIAL_ID, MODEL_NAME)
    history_path = os.path.join(TRIALS_DIR, TRIAL_ID, HISTORY_NAME)
    training(model_path, history_path)
