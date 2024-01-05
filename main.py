import os

import fire
import numpy as np
import tensorflow as tf
from icecream import ic
from keras.models import load_model

from project_manage import Config
from datasets import get_dataset, get_autoencoder_dataset, get_filtered_dataset
from models import classifier, autoencoder
from training import training
from validation import validate_model, get_threshold
from utils import inspect_data

conf = fire.Fire(Config)
model = None
# Load pre-trained model if available
if os.path.exists(os.path.join(conf.model_dir, "model.keras")):
    model = load_model(os.path.join(conf.model_dir, "model.keras"))

# Set up distributed training strategy
if conf.mode == "train":
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
        conf.batch_size = conf.batch_size * strategy.num_replicas_in_sync
    # Scope training within the distributed strategy
    with strategy.scope():
        if conf.model_type == "classifier":
            train_ds = get_dataset("train", shuffle=True, augment=True, conf=conf)
            val_ds = get_dataset("validation", conf=conf)
            ic(model)
            if model is None:
                model = classifier(input_shape=conf.input_shape)
            ic(model)
            model = training(train_ds=train_ds, val_ds=val_ds, model=model, conf=conf)

        elif conf.model_type == "autoencoder":
            train_ds = get_dataset("train", shuffle=True, conf=conf, keep_label=0)  # Keep normal
            val_ds = get_dataset("validation", conf=conf, keep_label=0)
            auto_train_ds, _ = get_autoencoder_dataset(train_ds)
            auto_val_ds, _ = get_autoencoder_dataset(val_ds)
            if model is None:
                model = autoencoder(input_shape=conf.input_shape)
            model = training(train_ds=auto_train_ds, val_ds=auto_val_ds, model=model, conf=conf)
    # Determine threshold for classification
    if conf.model_type == "classifier":
        threshold = get_threshold(val_ds, model, conf, save=True)
    elif conf.model_type == "autoencoder":
        normal_val_data = get_filtered_dataset(val_ds, conf, keep_label=0)
        threshold = get_threshold(normal_val_data, model, conf, save=True)
    else:
        raise ValueError(f"Unknown type: {conf.model_type} ")
    # Validate the model on different datasets
    for data_name in ["train", "validation", "test"]:
        print(f"Validate model on {data_name} dataset")
        dataset = get_dataset(data_name, conf=conf)
        validate_model(dataset_name=data_name, dataset=dataset, model=model, threshold=threshold, conf=conf, save=True)

# Load the threshold if available
if os.path.exists(os.path.join(conf.model_dir, "threshold.npy")):
    threshold = np.load(os.path.join(conf.model_dir, "threshold.npy"))
else:
    threshold = get_threshold("validation", model, conf, save=False)

# Validate the model on validation and test datasets
if conf.mode == "validation":
    for data_name in ["validation", "test"]:
        dataset = get_dataset(data_name, conf=conf)
        validate_model(dataset_name=data_name, dataset=dataset, model=model, threshold=threshold, conf=conf, save=True)
# Predict and inspect data in 'predict' mode
if conf.mode == "predict":
    if conf.model_type == "classifier":
        threshold = 0.8
    if conf.model_type == "autoencoder":
        threshold = 0.
    inspect_data(dataset=get_dataset("all", conf=conf), model=model, threshold=threshold, conf=conf)
