import os

import fire
import numpy as np
import tensorflow as tf
from keras.models import load_model

from project_manage import Config
from datasets import get_dataset
from models import classifier, autoencoder
from training import training
from validation import validate_model, get_threshold
from utils import inspect_data

# TF_CPP_MIN_LOG_LEVEL=3
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
        if os.path.exists(os.path.join(conf.model_dir, "model.keras")):
            model = load_model(os.path.join(conf.model_dir, "model.keras"))
        elif conf.model_type == "classifier":
            model = classifier(input_shape=conf.input_shape)
        elif conf.model_type == "autoencoder":
            model = autoencoder(input_shape=conf.input_shape)
        else:
            raise ValueError(f"Unknown type: {conf.model_type} ")

        trained_model = training(train_ds=train_ds, val_ds=val_ds, model=model, conf=conf)

    threshold = get_threshold("validation", trained_model, conf, save=True)

    for data_name in ["train", "validation", "test"]:
        print(f"Validate model on {data_name} dataset")
        validate_model(dataset_name=data_name, model=trained_model, threshold=threshold, conf=conf, save=True)

model = load_model(os.path.join(conf.model_dir, "model.keras"))
if os.path.exists(os.path.join(conf.model_dir, "threshold.npy")):
    threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
else:
    threshold = get_threshold("validation", model, conf, save=False)
if conf.mode == "validation":
    for data_name in ["validation", "test"]:
        validate_model(dataset_name=data_name, model=model, threshold=threshold, conf=conf, save=True)
if conf.mode == "predict":
    inspect_data(dataset=get_dataset("all", conf=conf), model=model, threshold=threshold, conf=conf)
