import os
from datetime import datetime

import cv2
import keras.models
import numpy as np

from config import TRIALS_DIR
from dataset import get_dataset_predictions


def get_latest_trial_id():
    latest_trial_id = sorted(os.listdir(TRIALS_DIR), key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
    print("latest_trial_id", latest_trial_id)
    return latest_trial_id


def inspect_data(model, mode, threshold=0.5):
    paused = False
    x_input, y_output, y_true = get_dataset_predictions(model=model, dataset=mode)
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    frame_index = 0
    for input_image in x_input:
        frame = np.squeeze(input_image, axis=0)
        print(frame)
        # while True:
        # frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # todo color_code train and test set images
        window_title = f"Frame Index: {int(frame_index)} - {'Paused' if paused else 'Playing'}\t"
        window_name = "Video Window"
        text = ""
        # if not ret:
        #     print("Error: Failed to read frame.")
        #     break
        if paused:
            window_title = " ".join([window_title, "(Paused)"])
            text = "".join([text, "(Paused)"])
            color = (255, 0, 0)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            if not model:
                text = "Inspection".join(text)
                color = (255, 0, 0)
            else:
                # input_image = np.expand_dims(cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)), axis=0).astype(float)
                y_probability = model.predict(input_image, verbose=0)
                y_prediction = np.where(y_probability > threshold, 1, 0)

                window_title += (f"Threshold: {np.round(threshold, 5)}, "
                                 f"Probability: {np.round(y_probability[0, 0], 5)}, "
                                 f"Prediction: {np.round(y_prediction[0, 0], 5)}")
                if y_prediction == 1:
                    text = "Anomaly"
                    color = (0, 0, 255)
                else:
                    text = "Normal"
                    color = (0, 255, 0)
            frame_index += 1

        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(window_name, window_title)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('p'):
            paused = not paused
        # Move back by 5 frames on 'b' key press
        elif key == ord('b'):
            frame_index = max(0, frame_index - 10)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Move back by 1 frame on 'v' key press
        elif key == ord('v'):
            frame_index = max(0, frame_index - 1)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Move forward by 10 frames on 'f' key press
        elif key == ord('f'):
            frame_index += 10
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Move forward by 1 frame on 'd' key press
        elif key == ord('d'):
            frame_index += 1
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Quit if 'q' key is pressed
        elif key == ord('q'):
            break
    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    inspect_data(model=model,mode="test", threshold=threshold)
