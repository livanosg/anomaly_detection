import os

import fire
import numpy as np
import keras.models

from config import MODE
from project_manage import Config
from training import supervised_training, unsupervised_training, validate_supervised_model, validate_unsupervised_model
from datasets import get_dataset
from utils import inspect_data
import tensorflow as tf

if __name__ == '__main__':
    conf = fire.Fire(Config)
    if MODE == "train":
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
            conf.batch_size = conf.batch_size * strategy.num_replicas_in_sync
        with strategy.scope():
            train_ds = get_dataset("train", shuffle=True, conf=conf)
            val_ds = get_dataset("validation", shuffle=True, conf=conf)
            if conf.method == "supervised":
                model, threshold = supervised_training(train_ds=train_ds, val_ds=val_ds, conf=conf)
                validate_supervised_model("train", dataset=train_ds, model=model, threshold=threshold,
                                          conf=conf, save=True)
                validate_supervised_model("validation", dataset=val_ds, model=model, threshold=threshold,
                                          conf=conf, save=True)

            if conf.method == "unsupervised":
                model, threshold = unsupervised_training(train_ds=train_ds, val_ds=val_ds, conf=conf)
                validate_unsupervised_model(title="train", dataset=train_ds, model=model, threshold=threshold,
                                            conf=conf, save=True)
                validate_unsupervised_model(title="validation", dataset=val_ds, model=model, threshold=threshold,
                                            conf=conf, save=True)

    else:
        model = keras.models.load_model(os.path.join(conf.model_dir, "model.keras"))
        threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
        if MODE == "validation":
            val_ds = get_dataset("validation", shuffle=False, conf=conf)
            if conf.method == "supervised":
                validate_supervised_model(title="validation", dataset=val_ds, model=model, threshold=threshold,
                                          conf=conf, save=False)
            if conf.method == "unsupervised":
                validate_unsupervised_model(title="validation", dataset=val_ds, model=model, threshold=threshold,
                                            conf=conf, save=False)
        if MODE == "predict":
            all_images = get_dataset("all", shuffle=False, conf=conf)
            if conf.method == "supervised":
                validate_supervised_model(title="validation", dataset=all_images, model=model, threshold=threshold,
                                          conf=conf, save=False)
            if conf.method == "unsupervised":
                validate_unsupervised_model("validation", all_images, model=model, threshold=threshold,
                                            conf=conf, save=False)
            a = inspect_data(dataset=all_images, model=model, threshold=threshold, conf=conf)
