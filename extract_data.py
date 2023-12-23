import os
import sys
from multiprocessing import Process

import cv2
import requests
from tqdm import tqdm

from config import RAW_DIR, ANOMALY_INDICES, URL_DATA_FILE, VIDEO_FILE, ANOMALY_DIR, NORMAL_DIR


def download_data(url):
    """
    Downloads a file from a remote URL and saves it to a local directory.

    Returns:
        None
    """
    file_name = os.path.basename(url)
    save_path = os.path.join(RAW_DIR, file_name)

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


def extract_images_from_video(video_path):
    """
    Extract frames from a video and save them as images.
    """
    print("Extract frames to images...")
    cap = cv2.VideoCapture(video_path)
    procs = []
    while True:
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        file_name = f"{os.path.splitext(os.path.basename(video_path))[0]}{str(idx).zfill(4)}.png"
        if idx in ANOMALY_INDICES:
            save_path = os.path.join(ANOMALY_DIR, file_name)
        else:
            save_path = os.path.join(NORMAL_DIR, file_name)
        proc = Process(target=cv2.imwrite, args=(save_path, frame))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    print("Extract frames to images done!")


if __name__ == '__main__':
    if not os.path.isfile(VIDEO_FILE):
        download_data(URL_DATA_FILE)
    if not (os.listdir(ANOMALY_DIR) and os.listdir(NORMAL_DIR)):
        extract_images_from_video(VIDEO_FILE)

    # time_func(apply_transformations)
    # load_to_csv()
