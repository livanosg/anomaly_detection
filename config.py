import itertools
import os
from datetime import datetime

URL_DATA_FILE = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
FILE_NAME = os.path.basename(URL_DATA_FILE)

SEED = 1312
TRIAL_ID = str(datetime.now().strftime("%Y%m%d%H%M%S"))
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RAW_DIR = os.path.join(DATA_DIR, "raw")


IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANOMALY_DIR = os.path.join(IMAGES_DIR, "anomaly")
NORMAL_DIR = os.path.join(IMAGES_DIR, "normal")
PROC_DIR = os.path.join(DATA_DIR, "processed_images")

[os.makedirs(dirs, exist_ok=True) for dirs in
 [DATA_DIR, MODELS_DIR, RAW_DIR, IMAGES_DIR, ANOMALY_DIR, NORMAL_DIR, PROC_DIR]]

CSV_FILE = os.path.join(DATA_DIR, "data.csv")
VIDEO_FILE = os.path.join(RAW_DIR, os.path.basename(URL_DATA_FILE))

ANOMALY_INDICES = list(itertools.chain(range(45, 538), range(799, 1266), range(1452, 2311),
                                       range(2495, 2931), range(2954, 3417), range(4092, 4817)))


BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
EPOCHS = 300
LEARNING_RATE = 1e-4
