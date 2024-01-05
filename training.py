import os
from contextlib import redirect_stdout

from keras.metrics import MeanSquaredError, F1Score
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger, EarlyStopping
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, MeanAbsoluteError

from metrics import plot_training_history


def training(train_ds, val_ds, model, conf):
    """
    Train a Keras model using the provided training and validation datasets and configurations.

    Parameters:
    - train_ds (tf.data.Dataset): Training dataset.
    - val_ds (tf.data.Dataset): Validation dataset.
    - model (keras.Model): Keras model to be trained.
    - conf (Config): Configuration object.

    Returns:
    - keras.Model: Trained model.
    """
    callbacks = [ModelCheckpoint(filepath=os.path.join(conf.model_dir, "model.keras"), save_best_only=True),
                 BackupAndRestore(os.path.join(conf.model_dir, "backup"), delete_checkpoint=False),
                 CSVLogger(os.path.join(conf.model_dir, "history.csv"), append=True),
                 ReduceLROnPlateau(patience=10, cooldown=5, monitor='loss', mode="min"),
                 EarlyStopping(patience=30, restore_best_weights=True, monitor='loss', mode="min")]
    metrics = []
    loss = None

    if conf.model_type == "classifier":
        metrics += [F1Score(average="macro")]
        loss = CategoricalCrossentropy()  # CategoricalFocalCrossentropy()
    if conf.model_type == "autoencoder":
        metrics += [MeanSquaredError()]
        loss = MeanAbsoluteError()

    optimizer = Adam(learning_rate=conf.learning_rate)
    model.summary()
    with open(os.path.join(conf.model_dir, "summary.txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(x=train_ds, validation_data=val_ds, epochs=conf.epochs, callbacks=callbacks)
    plot_training_history(history.history, conf, save=True)
    return model
