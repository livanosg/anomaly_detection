import itertools
import os


ANOMALY_INDICES = list(itertools.chain(range(45, 538), range(799, 1266), range(1452, 2311),
                                       range(2495, 2931), range(2954, 3417), range(4092, 4817)))
CLASS_NAMES = ["normal", "anomaly"]

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRIALS_DIR = os.path.join(ROOT_DIR, "trials")

DATA_FILE = os.path.join(DATA_DIR, "data.csv")
RAW_DIR = os.path.join(DATA_DIR, "raw")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_NAME = "model.keras"
HISTORY_NAME = "history.csv"
THRESHOLD_NAME = "threshold.npy"