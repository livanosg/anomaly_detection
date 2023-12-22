import os
from datetime import datetime

SEED = None
TRIAL_ID = str(datetime.now().strftime("%Y%m%d%H%M%S"))
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CSV_FILE = os.path.join(DATA_DIR, "data.csv")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR = os.path.join(DATA_DIR, "proc")
PROC_IMAGES_DIR = os.path.join(PROC_DATA_DIR, "processed_images")
IMAGES_DIR = os.path.join(PROC_DATA_DIR, "images")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
URL_DATA_FILE = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
FILE_NAME = os.path.basename(URL_DATA_FILE)
VIDEO_PATH = os.path.join(RAW_DATA_DIR, os.path.basename(URL_DATA_FILE))
TML_MODEL_FILE = os.path.join(MODELS_DIR, "isolation_forest", TRIAL_ID, "isolation_forest.model")
DL_MODEL_FILE = os.path.join(MODELS_DIR, "autoencoder", TRIAL_ID, "autoencoder_model.keras")
[os.makedirs(dirs) for dirs in [DATA_DIR, RAW_DATA_DIR, PROC_DATA_DIR, PROC_IMAGES_DIR, IMAGES_DIR] if
 not os.path.exists(dirs)]
ABNORMAL_INDICES = [range(45, 538),
                    range(799, 1266),
                    range(1452, 2311),
                    range(2495, 2931),
                    range(2954, 3417),
                    range(4092, 4817)]
