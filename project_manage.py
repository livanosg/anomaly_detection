import os
from datetime import datetime

ROOT_DIR = os.path.dirname(__file__)
CLASS_NAMES = ("normal", "anomaly")


class Config:
    """
    Configuration class for training and experimenting with models.

    Attributes:
    - TRIALS_DIR (str): The root directory for storing experiment trials.
    - trial_id (str): Unique identifier for the current trial.
    - model_type (str): Type of the model (e.g., "autoencoder").
    - mode (str): Operating mode ("train" or other).
    - threshold_type (str): Type of threshold used for evaluation.
    - class_names (tuple): Tuple of class names.
    - pos_label (int): Positive label for binary classification.
    - input_shape (tuple): Shape of the input data (height, width, channels).
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for training.
    - epochs (int): Number of training epochs.
    - seed (int): Seed for random number generation.

    Methods:
    - __init__: Initialize the configuration with specified or default values.
    """
    TRIALS_DIR = os.path.join(ROOT_DIR, "trials")

    def __init__(self,
                 trial_id="latest",
                 model_type="classifier",
                 mode="predict",
                 threshold_type="f1",
                 class_names=CLASS_NAMES,
                 pos_label=1,
                 input_shape=(224, 224, 3),
                 batch_size=128,
                 learning_rate=1e-3,
                 epochs=50,
                 seed=1312):

        if trial_id is None:
            self.trial_id = datetime.now().strftime("%Y%m%d%H%M%S")
        elif trial_id == "latest":
            self.trial_id = sorted(os.listdir(os.path.join(str(self.TRIALS_DIR), model_type)),
                                   key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        else:
            self.trial_id = str(trial_id)
        self.model_type = model_type.lower()

        self.trial_dir = os.path.join(self.TRIALS_DIR, self.model_type, self.trial_id)
        self.model_dir = os.path.join(self.trial_dir, "model")
        self.metrics_dir = os.path.join(self.trial_dir, "metrics")

        self.mode = mode.lower()
        if self.mode == "train":
            [os.makedirs(_dir, exist_ok=True) for _dir in [self.model_dir, self.metrics_dir]]

        self.threshold_type = threshold_type
        self.class_names = class_names
        self.pos_label = pos_label
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
