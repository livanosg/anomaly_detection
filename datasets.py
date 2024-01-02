import keras
import tensorflow as tf
from icecream import ic

from setup_data import DATASETS


def image_data_augmentation(image, label, seed):
    seeds = (seed, seed)
    # Data augmentation
    image = tf.image.stateless_random_flip_left_right(image, seed=seeds)
    image = tf.image.stateless_random_flip_up_down(image, seed=seeds)
    image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seeds)
    image = tf.image.stateless_random_contrast(image, lower=0.5, upper=1.5, seed=seeds)
    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seeds)
    image = tf.image.stateless_random_hue(image, max_delta=0.2, seed=seeds)
    return image, label


def get_dataset(dataset, conf, shuffle=False, augm=False,):
    ds = keras.utils.image_dataset_from_directory(directory=DATASETS[dataset],
                                                  labels="inferred",
                                                  label_mode="int",
                                                  class_names=conf.class_names,
                                                  image_size=conf.input_shape[:-1],
                                                  batch_size=conf.batch_size,
                                                  shuffle=shuffle,
                                                  seed=conf.seed)

    if augm:
        ds = ds.map(lambda image, label: image_data_augmentation(image, label, conf.seed),
                    num_parallel_calls=tf.data.AUTOTUNE, name="image_augm")
    ds = ds.map(lambda image, label: (tf.cast(image, tf.float32) / 255., label),
                num_parallel_calls=tf.data.AUTOTUNE, name="normalize")
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def split_inputs_labels(dataset):
    """
    Splits a dataset into input images and corresponding labels.
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Input images and labels datasets.
    """
    x_input = dataset.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_true = dataset.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE)
    return x_input, y_true
