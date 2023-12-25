import os
import sys
from multiprocessing import Process

import cv2
import pandas as pd
import requests
from tqdm import tqdm

from config import RAW_DIR, IMAGES_DIR, ANOMALY_INDICES, DATA_DIR


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
    return 1 if int(os.path.basename(os.path.splitext(image_path)[0])[-4:]) in ANOMALY_INDICES else 0


def create_csv():
    data = pd.DataFrame(data={"image": os.listdir(IMAGES_DIR)})
    data["labels"] = data["image"].map(_get_label)
    data.sort_values("image", key=lambda x: x.map(lambda y: os.path.basename(os.path.splitext(y)[0])[-4:]),
                     inplace=True)
    data.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False)


if __name__ == '__main__':
    video_url = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
    file_name = os.path.basename(video_url)
    file_local_path = os.path.join(RAW_DIR, file_name)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    download_data(url=video_url, save_path=file_local_path)
    extract_images(video_path=file_local_path)
