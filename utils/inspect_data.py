import cv2

from config import VIDEO_PATH


def inspect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    paused = True
    while True:
        window_title = f"Frame Index: {int(frame_index)} - {'Paused' if paused else 'Playing'}"
        cv2.namedWindow("Video Window", cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle("Video Window", window_title)

        if not paused:
            frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            # Check if the frame is successfully read
            if not ret:
                print("Error: Failed to read frame.")
                break
            # Update the window title with the frame index

            # Display the frame in the window
            cv2.imshow("Video Window", frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            cv2.imshow("Video Window", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused

        # Move back by 5 frames on 'b' key press
        elif key == ord('b'):
            frame_index = max(0, frame_index - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        elif key == ord('v'):
            frame_index = max(0, frame_index - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        elif key == ord('f'):
            frame_index += 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        elif key == ord('d'):
            frame_index += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break


if __name__ == '__main__':
    inspect_video(VIDEO_PATH)
