import keras
import tensorflow as tf
from config import SEED


def get_dataset(directory, conf, shuffle=True, get_labels=True):
    dataset = keras.utils.image_dataset_from_directory(directory=directory,
                                                       labels="inferred",
                                                       label_mode="int" if get_labels else None,
                                                       class_names=conf.class_names,
                                                       image_size=conf.image_shape,
                                                       batch_size=conf.batch_size,
                                                       shuffle=shuffle,
                                                       seed=SEED)

    if get_labels:
        dataset = dataset.map(lambda image, label: (image / 255., label), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda image: image / 255., num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def split_inputs_labels(dataset):
    """
    Splits a dataset into input images and corresponding labels.
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Input images and labels datasets.
    """
    x_input = dataset.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_true = dataset.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE)
    return x_input, y_true
