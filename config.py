import itertools
import os

VIDEO_URL = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
FILE_NAME = os.path.basename(VIDEO_URL)

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
VIDEO_FILE = os.path.join(RAW_DIR, os.path.basename(VIDEO_URL))

TRIALS_DIR = os.path.join(ROOT_DIR, "trials")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

CLASS_NAMES = ["normal", "anomaly"]
ANOMALIES = [range(45, 538), range(799, 1266), range(1452, 2311),
             range(2495, 2931), range(2954, 3417), range(4092, 4817)]
ANOMALIES_INDICES = list(itertools.chain(*ANOMALIES))

SEED = 2
MODE = "train"
TRIAL_ID = None
METHOD = "supervised"
EPOCHS = 300
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = [IMG_HEIGHT, IMG_WIDTH, CHANNELS]
