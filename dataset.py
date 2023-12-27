import os
import keras
import tensorflow as tf
from config import DATA_DIR, INPUT_SHAPE, CLASS_NAMES


def get_dataset(dataset, get_labels=True, batch_size=32):
    if not os.path.isdir(os.path.join(DATA_DIR, dataset)):
        raise NotADirectoryError(f"{os.path.join(DATA_DIR, dataset)} not a directory.")
    labels = None
    label_mode = None
    class_names = None
    if get_labels:
        labels = "inferred"
        label_mode = "categorical"
        class_names = CLASS_NAMES
    ds = keras.utils.image_dataset_from_directory(directory=os.path.join(DATA_DIR, dataset),
                                                  labels=labels,
                                                  label_mode=label_mode,
                                                  class_names=class_names,
                                                  image_size=(1080, 1920) if dataset == "images" else INPUT_SHAPE[:-1],
                                                  batch_size=batch_size,
                                                  shuffle=dataset is not "images",
                                                  seed=42)

    ds = ds.cache()
    if dataset == "train":
        ds = ds.shuffle(ds.cardinality())
    if get_labels:
        ds = ds.map(lambda image, label: (image / 255., label), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda image: image / 255., num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def split_inputs_labels(ds):
    x_input = ds.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_true = ds.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE)
    return x_input, y_true


def get_dataset_predictions(model, dataset, get_labels=True, batch_size=32):
    ds = get_dataset(dataset=dataset, get_labels=get_labels, batch_size=batch_size)
    if get_labels:
        x_input, y_true = split_inputs_labels(ds)
    else:
        x_input = ds
        y_true = None
    y_output = model.predict(x_input)
    return x_input, y_output, y_true
