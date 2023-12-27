import itertools
import os
from datetime import datetime

URL_DATA_FILE = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
FILE_NAME = os.path.basename(URL_DATA_FILE)

TRIAL_ID = str(datetime.now().strftime("%Y%m%d%H%M%S"))
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
TRIALS_DIR = os.path.join(ROOT_DIR, "trials")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

CSV_FILE = os.path.join(DATA_DIR, "data.csv")
VIDEO_FILE = os.path.join(RAW_DIR, os.path.basename(URL_DATA_FILE))
CLASS_NAMES = ["normal", "anomaly"]
ANOMALY_INDICES = list(itertools.chain(range(45, 538), range(799, 1266), range(1452, 2311),
                                       range(2495, 2931), range(2954, 3417), range(4092, 4817)))

SEED = 1312
BATCH_SIZE = 64
IMG_HEIGHT = 180
IMG_WIDTH = 180
CHANNELS = 3
INPUT_SHAPE = [IMG_HEIGHT, IMG_WIDTH, CHANNELS]
EPOCHS = 100
LEARNING_RATE = 1e-3
