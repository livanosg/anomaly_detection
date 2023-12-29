import os
import keras
import tensorflow as tf
from config import DATA_DIR, CLASS_NAMES, SEED


class DatasetHandler:
    def __init__(self, dataset_name, batch_size, get_labels, image_shape):
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(DATA_DIR, self.dataset_name)
        self.class_names = CLASS_NAMES
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.seed = SEED
        self.get_labels = get_labels
        self.dataset = keras.utils.image_dataset_from_directory(directory=self.dataset_dir,
                                                                labels="inferred" if self.get_labels else None,
                                                                label_mode="categorical" if self.get_labels else None,
                                                                class_names=self.class_names,
                                                                image_size=self.image_shape,
                                                                batch_size=self.batch_size,
                                                                shuffle=self.dataset_name != "images",
                                                                seed=self.seed).cache()

        if self.dataset_name == "train":
            self.dataset = self.dataset.shuffle(self.dataset.cardinality())
        if self.get_labels:
            self.dataset = self.dataset.map(lambda image, label: (image / 255., label),
                                            num_parallel_calls=tf.data.AUTOTUNE)
        else:
            self.dataset = self.dataset.map(lambda image: image / 255., num_parallel_calls=tf.data.AUTOTUNE)

        self.dataset = self.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def split_inputs_labels(self):
        """
        Splits a dataset into input images and corresponding labels.
        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset]: Input images and labels datasets.
        """
        x_input = self.dataset.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
        y_true = self.dataset.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE)
        return x_input, y_true
