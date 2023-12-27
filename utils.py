import os
import time
from datetime import datetime

import cv2
import keras.models
import numpy as np

from config import TRIALS_DIR, INPUT_SHAPE, IMAGES_DIR, TEST_DIR, RAW_DIR


def get_latest_trial_id():
    """
    Retrieves the latest trial ID from the trials' directory.

    Returns:
        str: The latest trial ID.
    """
    latest_trial_id = sorted(os.listdir(TRIALS_DIR), key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
    print("latest_trial_id", latest_trial_id)
    return latest_trial_id


def inspect_data(trial_id):
    """
    Allows interactive inspection of images, either with or without a trained model.

    Note:
        This function displays images interactively and accepts user input for inspection.
    """
    paused = False
    # anomaly_dir = os.path.join(TEST_DIR, "anomaly")
    # images_list = sorted(os.listdir(anomaly_dir))
    if trial_id:
        model_dir = os.path.join(TRIALS_DIR, str(trial_id), "model")
        model = keras.models.load_model(os.path.join(model_dir, "model.keras"))
        threshold = np.load(os.path.join(model_dir, "threshold.npy"))
    images_list = sorted(os.listdir(IMAGES_DIR))

    window_name = "Inspection tool"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    idx = 0
    while idx < len(images_list) - 1:
        image = cv2.imread(os.path.join(IMAGES_DIR, images_list[idx]))
        # todo color_code train and test set images
        window_title = " ".join([window_name, "(Paused)" if paused else len("(Paused)") * " "])

        if trial_id:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, dsize=INPUT_SHAPE[:-1])
            input_image = input_image / 225.
            y_pred_prob = model.predict(np.expand_dims(input_image, axis=0), verbose=0)
            y_pred_prob = y_pred_prob[:, 1]
            y_pred = np.greater_equal(y_pred_prob, threshold).astype(int)

            window_title = " ".join([window_title,
                                     f"Threshold: {round(float(threshold), 2)}",
                                     f"Probability: {round(float(y_pred_prob[0]), 2)}"])

            cv2.putText(image, "Anomaly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(image,
                          pt1=(int(image.shape[1] * 0.02),
                               int(image.shape[0] * 0.2) - int(image.shape[0] * 0.15 * y_pred_prob[0])),
                          pt2=(int(image.shape[1] * 0.05),
                               int(image.shape[0] * 0.2)),
                          color=(0, 0, int(128 * 0.7 + 127 * y_pred_prob[0])),
                          thickness=-1)
            cv2.putText(image, "Normal", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image,
                          pt1=(int(image.shape[1] * 0.09),
                               int(image.shape[0] * 0.2) - int(image.shape[0] * 0.15 * (1 - y_pred_prob[0]))),
                          pt2=(int(image.shape[1] * 0.12),
                               int(image.shape[0] * 0.2)),
                          color=(0, int(128 * 0.7 + 127 * (1 - y_pred_prob[0])), 0),
                          thickness=-1)
        else:
            cv2.putText(image, "Inspection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.setWindowTitle(window_name, window_title)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(10)
        if key == ord('p'):  # Pause
            paused = not paused
        elif key == ord('b'):  # Go back 10 frames
            idx = max(0, idx - 10)
        elif key == ord('v'):  # Go back 1 frame
            idx = max(0, idx - 1)
        elif key == ord('f'):  # Skip 10 frames
            idx += 10
        elif key == ord('d'):  # Skip 1 frame
            idx += 1
        elif key == ord('q'):  # Quit
            break
        if not paused:
            idx += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trial_id = get_latest_trial_id()
    # # trial_id = "20231227113941"
    inspect_data(trial_id=trial_id)
