import os
from datetime import datetime

ROOT_DIR = os.path.dirname(__file__)
TRIALS_DIR = os.path.join(ROOT_DIR, "trials")

CLASS_NAMES = ("normal", "anomaly")

TRIAL_ID = None
MODE = "train"
METHOD = "supervised"

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
SEED = 2


class Config:
    """Configuration class for the anomaly detection project."""
    def __init__(self, method=METHOD, trial_id=TRIAL_ID, mode=MODE, class_names=CLASS_NAMES,
                 input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, seed=SEED):
        """
        Initializes the Config instance.

        Args:
        - method (str): The method for anomaly detection. Default is "supervised".
        - trial_id: An identifier for the trial. If None, it is set to the current timestamp.
                    If "latest", it uses the latest trial ID in the specified method directory.
                    Otherwise, it uses the provided trial_id.
        - mode (str): The mode of operation, either "train" or "test". Default is "train".
        - class_names (tuple): Tuple containing class names, e.g., ("normal", "anomaly"). Default is ("normal", "anomaly").
        - input_shape (tuple): Input shape of the images in the format (height, width, channels). Default is (224, 224, 3).
        - batch_size (int): Batch size for training. Default is 32.
        - learning_rate (float): Learning rate for the model. Default is 1e-4.
        - epochs (int): Number of training epochs. Default is 100.
        - seed (int): Seed for reproducibility. Default is 2.
        """
        self.method = method.lower()
        if trial_id is None:
            self.trial_id = datetime.now().strftime("%Y%m%d%H%M%S")
        elif trial_id == "latest":
            self.trial_id = sorted(os.listdir(os.path.join(str(TRIALS_DIR), self.method)),
                                   key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        else:
            self.trial_id = str(trial_id)

        self.mode = mode.lower()
        self.trial_dir = os.path.join(TRIALS_DIR, self.method, str(self.trial_id))
        self.model_dir = os.path.join(str(self.trial_dir), "model")
        self.metrics_dir = os.path.join(str(self.trial_dir), "metrics")
        [os.makedirs(_dir, exist_ok=True) for _dir in [self.model_dir, self.metrics_dir]]

        self.class_names = class_names
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed


