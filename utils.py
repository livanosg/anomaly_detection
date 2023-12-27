import os
from datetime import datetime

import cv2
import keras.models
import numpy as np

from config import TRIALS_DIR, INPUT_SHAPE, IMAGES_DIR, TEST_DIR


def get_latest_trial_id():
    """
    Retrieves the latest trial ID from the trials' directory.

    Returns:
        str: The latest trial ID.
    """
    latest_trial_id = sorted(os.listdir(TRIALS_DIR), key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
    print("latest_trial_id", latest_trial_id)
    return latest_trial_id


def inspect_data(model, threshold=0.5):
    """
    Allows interactive inspection of images, either with or without a trained model.

    Args:
        model (tf.keras.Model): The trained model (or None for image inspection without a model).
        threshold (float): Threshold for anomaly detection.

    Note:
        This function displays images interactively and accepts user input for inspection.
    """
    paused = False
    anomaly_dir = os.path.join(TEST_DIR, "anomaly")
    images_list = sorted(os.listdir(anomaly_dir))
    # images_list = sorted(os.listdir(IMAGES_DIR))
    total_images = len(images_list)
    idx = 0
    while True:
        image = cv2.imread(os.path.join(IMAGES_DIR, images_list[idx]))
        # todo color_code train and test set images
        window_title = f"{'Paused' if paused else ''}"
        window_name = "Inspection tool"
        text = ""
        if not model:
            text = "Inspection".join(text)
            color = (255, 0, 0)
        else:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, dsize=INPUT_SHAPE[:-1])
            input_image = input_image / 225.
            y_pred_prob = model.predict(np.expand_dims(input_image, axis=0), verbose=0)
            y_pred_prob = y_pred_prob[:, 1]
            y_pred = np.greater_equal(y_pred_prob, threshold).astype(int)

            window_title += (f"Threshold: {np.round(threshold, 5)}, "
                             f"Probability: {np.round(y_pred_prob, 5)}, "
                             f"Prediction: {y_pred}")
            if y_pred == 1:
                text = "Anomaly"
                color = (0, 0, 255)
            else:
                text = "Normal"
                color = (0, 255, 0)
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(window_name, window_title)
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(5)
        if key == ord('p'):
            paused = not paused
        # # Move back by 10 frames on 'b' key press
        elif key == ord('b'):
            idx = max(0, idx - 10)
        # # Move back by 1 frame on 'v' key press
        elif key == ord('v'):
            idx = max(0, idx - 1)
        # # Move forward by 10 frames on 'f' key press
        elif key == ord('f'):
            idx += 10
            idx = min(idx + 10, total_images - 1)
        # # Move forward by 1 frame on 'd' key press
        elif key == ord('d'):
            idx = min(idx + 1, total_images - 1)
        # # Quit if 'q' key is pressed
        elif key == ord('q'):
            break
        if not paused:
            idx += 1
        if idx >= total_images:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trial_id = get_latest_trial_id()
    # trial_id = "20231227113941"
    trial_dir = os.path.join(TRIALS_DIR, str(trial_id))
    model_dir = os.path.join(str(trial_dir), "model")
    inspect_data(model=keras.models.load_model(os.path.join(model_dir, "model.keras")),
                 threshold=np.load(os.path.join(model_dir, "threshold.npy")))
