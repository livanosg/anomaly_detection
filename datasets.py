import keras_cv
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from setup_data import DATASETS


def get_basic_augm_pipeline(seed):
    """
        Get a basic image augmentation pipeline using RandAugment and GridMask.

        Parameters:
        - seed (int): Seed for random number generation.

        Returns:
        - keras_cv.layers.RandomAugmentationPipeline: Augmentation pipeline.
        """
    layers = keras_cv.layers.RandAugment.get_standard_policy(value_range=(0, 255), magnitude=0.5, magnitude_stddev=0.2,
                                                             seed=seed)
    layers += [keras_cv.layers.GridMask((0., 0.3), seed=seed)]
    return keras_cv.layers.RandomAugmentationPipeline(layers=layers, augmentations_per_image=3, seed=seed)


def split_inputs_labels(dataset):
    """
        Split a dataset into input images and corresponding labels.

        Parameters:
        - dataset (tf.data.Dataset): Input dataset.

        Returns:
        - tuple: Two datasets containing input images and labels, respectively.
        """
    x_input = dataset.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_true = dataset.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE)
    return x_input.prefetch(buffer_size=tf.data.AUTOTUNE), y_true.prefetch(buffer_size=tf.data.AUTOTUNE)


def get_dataset(dataset_name, conf, shuffle=False, augment=False, keep_label=None):
    """
    Get a TensorFlow dataset for training or evaluation.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - conf (Config): Configuration object.
    - shuffle (bool): Whether to shuffle the dataset.
    - augment (bool): Whether to apply data augmentation.
    - keep_label (int or None): Class label to keep in the dataset.

    Returns:
    - tf.data.Dataset: TensorFlow dataset.
    """
    ds = image_dataset_from_directory(directory=DATASETS[dataset_name],
                                      labels="inferred",
                                      label_mode="categorical",
                                      class_names=conf.class_names,
                                      image_size=conf.input_shape[:-1],
                                      batch_size=conf.batch_size,
                                      shuffle=shuffle,
                                      seed=conf.seed)

    mixup = keras_cv.layers.MixUp(seed=conf.seed)
    fouriermix = keras_cv.layers.FourierMix(seed=conf.seed)
    pipeline = get_basic_augm_pipeline(conf.seed)

    # noinspection PyCallingNonCallable
    def _augm(image, label):
        inputs = {"images": image, "labels": label}
        inputs = mixup(inputs)
        inputs = fouriermix(inputs)
        inputs = pipeline(inputs)
        return inputs["images"], inputs["labels"]

    if keep_label:
        ds = get_filtered_dataset(ds, conf, keep_label=keep_label)
    ds = ds.cache()
    if augment:
        ds = ds.map(lambda image, label: _augm(image, label), num_parallel_calls=tf.data.AUTOTUNE, name="image_augm")
    ds = ds.map(lambda image, label: (image / 255., label), num_parallel_calls=tf.data.AUTOTUNE, name="normalize")
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def get_filtered_dataset(dataset, conf, keep_label=None):
    """
    Get a filtered dataset containing only samples with a specified label.

    Parameters:
    - dataset (tf.data.Dataset): Input dataset.
    - conf (Config): Configuration object.
    - keep_label (int): Class label to keep in the dataset.

    Returns:
    - tf.data.Dataset: Filtered TensorFlow dataset.
    """
    return dataset.unbatch().filter(lambda x, y: tf.math.equal(tf.argmax(y), keep_label)).batch(conf.batch_size)


def get_autoencoder_dataset(dataset):
    """
    Prepare a dataset for training an autoencoder.

    Parameters:
    - dataset (tf.data.Dataset): Input dataset.

    Returns:
    - tuple: Two datasets containing input images and labels, respectively.
    """
    dataset = dataset.map(lambda image, label: ((image, image), label), num_parallel_calls=tf.data.AUTOTUNE)
    ae_dataset, labels = split_inputs_labels(dataset)
    return ae_dataset.prefetch(buffer_size=tf.data.AUTOTUNE), labels.prefetch(buffer_size=tf.data.AUTOTUNE)
