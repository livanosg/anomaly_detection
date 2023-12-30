import keras
from keras.src.callbacks import ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger, EarlyStopping

from model import calculate_threshold


def training_pipeline(model, train_data, val_data, conf):
    callbacks = [ModelCheckpoint(filepath=conf.model_path, save_best_only=True),
                 BackupAndRestore(conf.backup_dir, delete_checkpoint=False),
                 CSVLogger(conf.history_path, append=True)]
    optimizer = keras.optimizers.Adam(learning_rate=conf.learning_rate)

    if conf.method == "supervised":
        metrics = [keras.metrics.Precision(), keras.metrics.Recall()]
        callbacks += [ReduceLROnPlateau(patience=10, cooldown=5, monitor='val_loss', mode="min"),
                      EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', mode="min")]
        loss = keras.losses.BinaryFocalCrossentropy()
    else:
        metrics = [keras.metrics.MeanSquaredError()]
        callbacks += [ReduceLROnPlateau(patience=10, cooldown=5, monitor='loss', mode="min"),
                      EarlyStopping(patience=20, restore_best_weights=True, monitor='loss', mode="min")]
        loss = keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(x=train_data, validation_data=val_data, epochs=conf.epochs, callbacks=callbacks)
    calculate_threshold(model, train_data, conf)
    return history
