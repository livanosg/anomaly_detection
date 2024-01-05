import os
from datetime import datetime

ROOT_DIR = os.path.dirname(__file__)
CLASS_NAMES = ("normal", "anomaly")


class Config:
    """
    Configuration class for machine learning or deep learning experiments.

    Attributes:
    - TRIALS_DIR (str): Directory path for storing experiment trials.

    Parameters:
    - trial_id (str or None): Identifier for the experiment. If None, a timestamp will be generated.
    - model_type (str): Type of the model, e.g., 'classifier'.
    - mode (str): Mode of operation, should be 'train' or another mode depending on the project.
    - threshold_type (str): Type of threshold used, e.g., 'f1'.
    - class_names (tuple): Tuple of class names, default is ("normal", "anomaly").
    - pos_label (int): Positive label for binary classification, default is 1.
    - input_shape (tuple): Shape of the input data, default is (224, 224, 3).
    - batch_size (int): Batch size for training, default is 32.
    - learning_rate (float): Learning rate for the optimizer, default is 1e-3.
    - epochs (int): Number of training epochs, default is 50.
    - seed (int): Seed for random number generation, default is 1312.

    Attributes:
    - model_type (str): Lowercased version of the provided model_type.
    - trial_id (str): Experiment identifier.
    - trial_dir (str): Directory path for the current experiment.
    - model_dir (str): Directory path for saving the trained model.
    - metrics_dir (str): Directory path for saving experiment metrics.
    - mode (str): Lowercased version of the provided mode.

    Methods:
    - No public methods are defined in this class.

    Usage:
    config = Config(trial_id="exp1", model_type="classifier", epochs=100)
    print(config.learning_rate)  # Accessing configuration parameters.
    """
    TRIALS_DIR = os.path.join(ROOT_DIR, "trials")

    def __init__(self,
                 trial_id=None,  # "latest",
                 model_type="autoencoder",
                 mode="train",
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
