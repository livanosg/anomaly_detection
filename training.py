import os
from contextlib import redirect_stdout

from keras.metrics import MeanSquaredError, F1Score
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger, EarlyStopping
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, MeanAbsoluteError

from datasets import get_autoencoder_dataset
from metrics import plot_training_history
from models import classifier


def training(train_ds, val_ds, model, conf):
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
        train_ds, _ = get_autoencoder_dataset(train_ds, conf, keep_label=0)  # Keep normal
        val_ds, _ = get_autoencoder_dataset(val_ds, conf)
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


if __name__ == '__main__':
    import keras
    model = classifier(input_shape=(224, 224, 3))
    model.load_weights("/home/meph103/PycharmProjects/anomaly_detection/trials/classifier/20240104205411/model/.weights.h5")
    print(model)