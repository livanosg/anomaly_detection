import os
import keras
import tensorflow as tf
from config import DATA_DIR, INPUT_SHAPE, CLASS_NAMES


def get_dataset(dataset, get_labels=True, batch_size=32):
    """
      Retrieves an image dataset from the specified directory.

      Args:
          dataset (str): The dataset type ("train", "validation", "test").
          get_labels (bool): Whether to include labels in the dataset.
          batch_size (int): Batch size for loading images.

      Returns:
          tf.data.Dataset: A TensorFlow Dataset containing images and labels (if specified).
      """
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
                                                  image_size=INPUT_SHAPE[:-1],
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
    """
    Splits a dataset into input images and corresponding labels.

    Args:
        ds (tf.data.Dataset): The input dataset.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Input images and labels datasets.
    """
    x_input = ds.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_true = ds.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE)
    return x_input, y_true
