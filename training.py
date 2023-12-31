import os

import keras
from keras.src.callbacks import ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger, EarlyStopping

from metrics import plot_training_history
from model import calculate_threshold


def training_pipeline(model, train_data, val_data, **kwargs):
    callbacks = [ModelCheckpoint(filepath=os.path.join(kwargs["model_dir"], "model.keras"), save_best_only=True),
                 BackupAndRestore(os.path.join(kwargs["model_dir"], "backup"), delete_checkpoint=False),
                 CSVLogger(os.path.join(kwargs["model_dir"], "history.csv"), append=True)]

    if kwargs["method"] == "supervised":
        metrics = [keras.metrics.Precision(), keras.metrics.Recall()]
        callbacks += [ReduceLROnPlateau(patience=10, cooldown=5, monitor='val_loss', mode="min"),
                      EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', mode="min")]
        loss = keras.losses.BinaryFocalCrossentropy()
    else:
        metrics = [keras.metrics.MeanSquaredError()]
        callbacks += [
            # ReduceLROnPlateau(patience=10, cooldown=5, monitor='loss', mode="min"),
            EarlyStopping(patience=20, restore_best_weights=True, monitor='loss', mode="min")]
        loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=kwargs["learning_rate"])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)
    with open(os.path.join(kwargs["model_dir"], "summary"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    history = model.fit(x=train_data, validation_data=val_data, epochs=kwargs["epochs"], callbacks=callbacks)
    calculate_threshold(model, train_data, **kwargs)
    plot_training_history(history.history)

    return history
