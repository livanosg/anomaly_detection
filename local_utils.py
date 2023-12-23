import cv2
import keras.models
import numpy as np

from config import VIDEO_FILE, IMG_WIDTH, IMG_HEIGHT


def inspect_video(video_path, model=None, threshold=0.5):
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    paused = True
    while True:
        window_title = f"Frame Index: {int(frame_index)} - {'Paused' if paused else 'Playing'}"
        cv2.namedWindow("Video Window", cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle("Video Window", window_title)
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not paused:
            # Check if the frame is successfully read
            if not ret:
                print("Error: Failed to read frame.")
                break
            if model is not None:
                y_pred = model.predict(np.expand_dims(cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)), axis=0), verbose=0)
                print(y_pred)
                if y_pred > threshold:
                    cv2.putText(frame, "Anomaly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Window", frame)
        else:
            cv2.putText(frame, "Paused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Video Window", frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused

        # Move back by 5 frames on 'b' key press
        elif key == ord('b'):
            frame_index = max(0, frame_index - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Move back by 1 frame on 'v' key press
        elif key == ord('v'):
            frame_index = max(0, frame_index - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Move forward by 10 frames on 'f' key press
        elif key == ord('f'):
            frame_index += 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Move forward by 1 frame on 'd' key press
        elif key == ord('d'):
            frame_index += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Quit if 'q' key is pressed
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = "model.keras"
    inspect_video(video_path=VIDEO_FILE, model=model_path, threshold=0.3)
