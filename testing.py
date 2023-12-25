import os

import keras
import numpy as np

from config import TEST_DIR
from local_utils import get_latest_trial_id, trial_dirs
from validation import validate_model

if __name__ == '__main__':
    trial_id = get_latest_trial_id()
    trial_dir, model_path, history_path, threshold_path = trial_dirs(trial_id)

    model = keras.models.load_model(model_path)
    threshold = np.load(os.path.join(str(trial_dir), "threshold.npy"))

    validate_model(model=model, data_dir=TEST_DIR, history_path=history_path, threshold_path=threshold_path, name="test")
