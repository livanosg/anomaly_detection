import keras
import tensorflow as tf
from config import SEED, TRAIN_DIR, VALIDATION_DIR, TEST_DIR, IMAGES_DIR


def get_dataset(mode, shuffle, **kwargs):
    datasets = {"train": TRAIN_DIR,
                "validation": VALIDATION_DIR,
                "test": TEST_DIR,
                "all": IMAGES_DIR
                }
    dataset = keras.utils.image_dataset_from_directory(directory=datasets[mode],
                                                       labels="inferred",
                                                       label_mode="int",
                                                       class_names=kwargs["class_names"],
                                                       image_size=kwargs["image_size"],
                                                       batch_size=kwargs["batch_size"],
                                                       shuffle=shuffle,
                                                       seed=SEED)

    dataset = dataset.map(lambda image, label: (image / 255., label), num_parallel_calls=tf.data.AUTOTUNE, name="Normalize")
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
