import os

import fire
import numpy as np
import keras.models
import tensorflow as tf
from icecream import ic

from project_manage import Config
from training import supervised_training, unsupervised_training, validate_supervised_model, validate_unsupervised_model
from datasets import get_dataset
from utils import inspect_data

if __name__ == '__main__':
    conf = fire.Fire(Config)
    if conf.mode == "train":
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
            conf.batch_size = conf.batch_size * strategy.num_replicas_in_sync
        with strategy.scope():
            train_ds = get_dataset("train", shuffle=True, augment=True, conf=conf)
            val_ds = get_dataset("validation", conf=conf)
            test_ds = get_dataset("test", conf=conf)

            if conf.method == "supervised":
                supervised_training(train_ds=train_ds, val_ds=val_ds, conf=conf)

            if conf.method == "unsupervised":
                unsupervised_training(train_ds=train_ds, val_ds=val_ds, conf=conf)
        model = keras.models.load_model(os.path.join(conf.model_dir, "model.keras"))
        threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
        if conf.method == "supervised":
            validate_supervised_model("validation", conf=conf, save=True,
                                      dataset=val_ds,
                                      model=model,
                                      threshold=threshold)
            validate_supervised_model("test", conf=conf, save=True,
                                      dataset=test_ds,
                                      model=model,
                                      threshold=threshold)
        if conf.method == "unsupervised":
            validate_unsupervised_model("train", conf=conf, save=True,
                                        dataset=get_dataset("train", shuffle=False, augment=False, conf=conf),
                                        model=model,
                                        threshold=threshold)
            validate_unsupervised_model("validation", conf=conf, save=True,
                                        dataset=val_ds,
                                        model=model,
                                        threshold=threshold)
    else:
        model = keras.models.load_model(os.path.join(conf.model_dir, "model.keras"))
        threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
        if conf.mode == "validation":
            val_ds = get_dataset("validation", conf=conf)
            test_ds = get_dataset("test", conf=conf)
            if conf.method == "supervised":
                validate_supervised_model(title="test", dataset=test_ds, model=model, threshold=threshold,
                                          conf=conf, save=False)
            if conf.method == "unsupervised":
                validate_unsupervised_model(title="test", dataset=test_ds, model=model, threshold=threshold,
                                            conf=conf, save=False)
        if conf.mode == "predict":
            all_images = get_dataset("all", conf=conf)
            if conf.method == "supervised":
                validate_supervised_model(title="all", dataset=all_images, model=model, threshold=threshold,
                                          conf=conf, save=False)
            if conf.method == "unsupervised":
                validate_unsupervised_model("all", all_images, model=model, threshold=threshold,
                                            conf=conf, save=False)
            a = inspect_data(dataset=all_images, model=model, threshold=threshold, conf=conf)
