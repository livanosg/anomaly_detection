import os
import shutil
import sys
from multiprocessing import Process

import cv2
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import RAW_DIR, IMAGES_DIR, ANOMALY_INDICES, DATA_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR


def download_data(url, save_path):
    """
    Downloads a file from a remote URL and saves it to a local directory.

    Returns:
        None
    """
    file_name = os.path.basename(url)

    response = requests.get(url, stream=True)
    remote_file_size = int(response.headers.get("content-length", 0))

    if os.path.isfile(save_path):
        if remote_file_size == os.stat(save_path).st_size:
            print(f"File {file_name} is already downloaded!")
            return
        print(f"File {file_name} is not fully downloaded...")
    print(f"Downloading...")
    with open(save_path, "wb") as handle:
        for data in tqdm(response.iter_content(),
                         desc=f"{file_name}",
                         total=remote_file_size,
                         unit="bytes",
                         unit_scale=True,
                         unit_divisor=1024,
                         colour="#e92c6f",
                         smoothing=0.4,
                         file=sys.stdout):
            handle.write(data)
    print(f"File {file_name} downloaded!")


def extract_images(video_path):
    """
    Extract frames from a video and save them as images.
    """
    print("Extract video to images...")
    video = cv2.VideoCapture(video_path)
    processes = []
    image_base_name = os.path.splitext(os.path.basename(video_path))[0]
    while True:
        frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        formatted_index = str(frame_index).zfill(4)

        ret, frame = video.read()
        if not ret:
            break
        image_name = image_base_name + formatted_index + ".png"
        save_path = os.path.join(IMAGES_DIR, image_name)
        if not os.path.isfile(save_path):
            process = Process(target=cv2.imwrite, args=(save_path, frame))
            processes.append(process)
            process.start()
    for process in processes:
        process.join()
    print("Extract video to images done!")


def _get_label(image_path):
    """
    Determines the label (normal or anomaly) for an image based on its filename.

    Args:
        image_path (str): The path of the image file.

    Returns:
        int: 1 if the image is identified as an anomaly, 0 otherwise.
    """
    return 1 if int(os.path.basename(os.path.splitext(image_path)[0])[-4:]) in ANOMALY_INDICES else 0


def split_train_val_test(val_size=0.1, test_size=0.1):
    """
    Splits the dataset into training, validation, and test sets based on specified sizes.

    Args:
        val_size (float): Fraction of the dataset to use for validation.
        test_size (float): Fraction of the dataset to use for testing.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "data.csv"))
    images = df["images"].values
    labels = df["labels"].values
    x_train, x_temp, y_train, y_temp = train_test_split(images, labels,
                                                        test_size=val_size + test_size,
                                                        stratify=labels,
                                                        random_state=1)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                    test_size=test_size / (val_size + test_size),
                                                    stratify=y_temp,
                                                    random_state=1)

    for mode_dir, images in [[TRAIN_DIR, x_train], [VALIDATION_DIR, x_val], [TEST_DIR, x_test]]:
        if os.path.isdir(mode_dir):
            shutil.rmtree(mode_dir)
        for image in images:
            if int(os.path.basename(os.path.splitext(image)[0])[-4:]) in ANOMALY_INDICES:
                save_dir = os.path.join(mode_dir, "anomaly")
            else:
                save_dir = os.path.join(mode_dir, "normal")

            os.makedirs(save_dir, exist_ok=True)
            shutil.copyfile(os.path.join(IMAGES_DIR, image), os.path.join(save_dir, image))


def create_csv():
    """
    Creates a CSV file containing image filenames and corresponding labels.

    The labels are determined using the _get_label function and are based on the presence
    of images in the ANOMALY_INDICES set.

    The resulting CSV file is saved in the DATA_DIR directory with the name "data.csv".
    """
    data = pd.DataFrame(data={"images": os.listdir(IMAGES_DIR)})
    data["labels"] = data["images"].map(_get_label)
    data.sort_values("images", key=lambda x: x.map(lambda y: os.path.basename(os.path.splitext(y)[0])[-4:]),
                     inplace=True)
    data.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False)


if __name__ == '__main__':
    video_url = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
    video_file = os.path.basename(video_url)
    video_local_path = os.path.join(RAW_DIR, video_file)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    download_data(url=video_url, save_path=video_local_path)
    extract_images(video_path=video_local_path)
    create_csv()
    split_train_val_test(0.1, 0.1)
