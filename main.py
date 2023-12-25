import os
from datetime import datetime

import numpy as np

from config import VALIDATION_DIR, TEST_DIR
from local_utils import trial_dirs
from training import training
from validation import validate_model, calculate_threshold

if __name__ == '__main__':
    trial_id = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    trial_dir, model_path, history_path, threshold_path = trial_dirs(trial_id)
    model = training(model_path=model_path, history_path=history_path)
    validate_model(model=model,
                   data_dir=VALIDATION_DIR,
                   threshold_path=threshold_path,
                   history_path=history_path, name="validation")
    validate_model(model=model,
                   data_dir=TEST_DIR,
                   history_path=history_path,
                   threshold_path=threshold_path, name="test",)

    x_input = data.map(lambda images, _: images, num_parallel_calls=tf.data.AUTOTUNE)
    y_output = model.predict(x_input)
    y_true = data.map(lambda _, labels: labels, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    y_true = np.asarray([label for label in y_true.as_numpy_iterator()])

    y_output = y_output.squeeze()
    y_true = y_true.squeeze()

    if os.path.isfile(threshold_path):
        threshold = np.load(threshold_path)
    else:
        threshold = calculate_threshold(y_true, y_output, threshold_path)
