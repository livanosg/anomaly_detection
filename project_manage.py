import os
from datetime import datetime

import tensorflow as tf

from config import TRIALS_DIR, DATA_DIR, CLASS_NAMES, INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, EPOCHS


class Config:
    def __init__(self, trial_id=None, method="supervised", mode="train", class_names=CLASS_NAMES,
                 input_shape=INPUT_SHAPE,
                 batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS):

        self.method = method.lower()
        self.mode = mode.lower()

        if trial_id is None:
            self.trial_id = datetime.now().strftime("%Y%m%d%H%M%S")
        elif trial_id == "latest":
            self.trial_id = sorted(os.path.join(TRIALS_DIR, self.method),
                                   key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        else:
            self.trial_id = trial_id
        self.trial_dir = os.path.join(TRIALS_DIR, self.method, str(self.trial_id))
        self.model_dir = os.path.join(self.trial_dir, "model")
        self.metrics_dir = os.path.join(self.trial_dir, "metrics")
        self.backup_dir = os.path.join(self.trial_dir, "backup")

        [os.makedirs(_dir, exist_ok=True) for _dir in [self.model_dir, self.metrics_dir]]
        self.batch_size = batch_size
        if self.mode == "train":
            if tf.config.list_physical_devices('GPU'):
                self.strategy = tf.distribute.MirroredStrategy()
                self.batch_size = self.batch_size * self.strategy.num_replicas_in_sync

            else:
                self.strategy = tf.distribute.get_strategy()

        self.model_path = os.path.join(self.model_dir, "model.keras")
        self.threshold_path = os.path.join(self.model_dir, "threshold.npy")
        self.y_true = os.path.join(self.metrics_dir, "y_true.npy")
        self.y_prob = os.path.join(self.metrics_dir, "y_prob.npy")
        self.summary_path = os.path.join(self.metrics_dir, "summary.txt")
        self.history_path = os.path.join(self.metrics_dir, "history.csv")

        self.train_dir = os.path.join(DATA_DIR, "train")
        self.val_dir = os.path.join(DATA_DIR, "validation")
        self.test_dir = os.path.join(DATA_DIR, "test")
        self.class_names = class_names
        if self.method == "unsupervised":
            self.train_dir = os.path.join(self.train_dir, "normal")
            self.val_dir = os.path.join(self.val_dir, "normal")
            # self.test_dir = os.path.join(self.val_dir, "normal")
            self.class_names = []

        self.input_shape = input_shape
        self.image_shape = self.input_shape[:-1]
        self.learning_rate = learning_rate
        self.epochs = epochs
