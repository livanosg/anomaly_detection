import tensorflow as tf
from icecream import ic
from keras.utils import image_dataset_from_directory
from setup_data import DATASETS


def augmentations(image, seed):
    seeds = (seed, seed)
    # Data augmentation
    image = tf.image.stateless_random_flip_left_right(image, seed=seeds)
    image = tf.image.stateless_random_flip_up_down(image, seed=seeds)
    image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seeds)
    image = tf.image.stateless_random_contrast(image, lower=0.5, upper=1.5, seed=seeds)
    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seeds)
    image = tf.image.stateless_random_hue(image, max_delta=0.2, seed=seeds)
    return image


# rand_augment = keras_cv.layers.RandAugment(
#     value_range=(0, 255),
#     augmentations_per_image=3,
#     magnitude=0.3,
#     magnitude_stddev=0.2,
#     rate=1.0,
#     auto_vectorize=False,
# )
#
#
# def augmentations(image, label):
#     inputs = {"images": image, "labels": label}
#     # inputs = keras_cv.layers.CutMix()( training=True)
#     # inputs = keras_cv.layers.MixUp()(inputs, training=True)
#     inputs = rand_augment._augment(inputs)
#     print(inputs)
#     return inputs["images"], inputs["labels"]


def get_dataset(dataset, conf, shuffle=False, augment=False):
    ds = image_dataset_from_directory(directory=DATASETS[dataset],
                                      labels="inferred",
                                      label_mode="categorical" if conf.label_type == "categorical" else "int",
                                      class_names=conf.class_names,
                                      image_size=conf.input_shape[:-1],
                                      batch_size=conf.batch_size,
                                      shuffle=shuffle,
                                      seed=conf.seed)

    if augment:
        ds = ds.map(lambda image, label: (augmentations(image, conf.seed), label),
                    num_parallel_calls=tf.data.AUTOTUNE, name="image_augm")
    ds = ds.map(lambda image, label: (image / 255., label), num_parallel_calls=tf.data.AUTOTUNE, name="normalize")
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
