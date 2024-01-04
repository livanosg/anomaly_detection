import itertools
import os
import shutil
import sys
from multiprocessing import Process

import cv2
import pandas as pd
import requests
from tqdm import tqdm

from project_manage import ROOT_DIR, CLASS_NAMES

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

anomalies = [range(45, 538), range(799, 1266), range(1452, 2311),
             range(2495, 2931), range(2954, 3417), range(4092, 4817)]
anomalies_idx = list(itertools.chain(*anomalies))


DATASETS = {"train": os.path.join(DATA_DIR, "train"),
            "validation": os.path.join(DATA_DIR, "validation"),
            "test": os.path.join(DATA_DIR, "test"),
            "all": os.path.join(DATA_DIR, "all")
            }


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
            return None
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


def extract_images(from_video, to_dir):
    """
    Extract frames from a video and save them as images.
    """
    print(f"Extract: {from_video}\n     to: {to_dir}")
    video = cv2.VideoCapture(from_video)
    processes = []
    image_base_name = os.path.splitext(os.path.basename(from_video))[0]
    shutil.rmtree(to_dir)
    [os.makedirs(os.path.join(to_dir, class_name), exist_ok=True) for class_name in CLASS_NAMES]

    while video.isOpened():
        frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        formatted_index = str(frame_index).zfill(4)
        ret, frame = video.read()
        if not ret:
            break
        image_name = image_base_name + formatted_index + ".png"
        save_path = os.path.join(to_dir, CLASS_NAMES[int(frame_index in anomalies_idx)], image_name)
        process = Process(target=cv2.imwrite, args=(save_path, frame))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    create_csv(dataset_dir=to_dir)
    print("Extract video to images done!")


def _get_index(image_path):
    return int(os.path.splitext(os.path.basename(image_path))[0][-4:])


def _get_label(image_path):
    return int(_get_index(image_path) in anomalies_idx)


def create_csv(dataset_dir):
    """
    Creates a CSV file containing image filenames and corresponding labels.

    The labels are determined using the _get_label function and are based on the presence
    of images in the ANOMALY_INDICES set.

    The resulting CSV file is saved in the DATA_DIR directory with the name "data.csv".
    """
    data = pd.concat(
        [pd.DataFrame(data={"images": os.listdir(os.path.join(dataset_dir, str(_class)))}) for _class in CLASS_NAMES])
    data["labels"] = data["images"].map(_get_label)
    data.sort_values("images", key=lambda x: x.map(_get_index), inplace=True)
    data.to_csv(os.path.join(dataset_dir, f"{os.path.basename(dataset_dir)}_data.csv"), index=False)


def split_train_val(val_size=0.1, test_size=0.1):
    """
    Splits the dataset into training, validation, and test sets based on specified sizes.

    Args:
        val_size (float): Fraction of the dataset to use for validation.
        test_size (float): Fraction of the dataset to use for test.
    """

    df = pd.read_csv(os.path.join(DATASETS["all"], f"{os.path.basename(DATASETS['all'])}_data.csv"))
    train_data = anomalies[:int(len(anomalies) * (1 - (val_size + test_size)))]
    val_data = anomalies[len(train_data): int(len(anomalies) * (1 - test_size))]
    train_idx = list(itertools.chain(*train_data))[-1]
    val_idx = list(itertools.chain(*val_data))[-1]
    train_df = df.iloc[df.index <= train_idx]
    val_df = df.iloc[(train_idx < df.index) & (df.index <= val_idx)]
    test_df = df.iloc[val_idx < df.index]
    for data_dir, df in [[DATASETS["train"], train_df], [DATASETS["validation"], val_df],  [DATASETS["test"], test_df]]:
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        for image_name in df["images"].values:
            _class = CLASS_NAMES[int(_get_index(image_name) in anomalies_idx)]
            from_dir = os.path.join(DATASETS["all"], _class, image_name)
            to_dir = os.path.join(data_dir, _class, image_name)
            os.makedirs(os.path.dirname(to_dir), exist_ok=True)
            shutil.copyfile(from_dir, to_dir)
        create_csv(data_dir)


if __name__ == '__main__':
    video_url = "https://repository.detectionnow.com/content/rgb/denim_elastane.mp4"
    video_file = os.path.basename(video_url)
    video_path = os.path.join(RAW_DIR, video_file)
    [os.makedirs(_dir, exist_ok=True) for _dir in [RAW_DIR, DATASETS["all"]]]
    download_data(url=video_url, save_path=video_path)
    extract_images(from_video=video_path, to_dir=DATASETS["all"])
    split_train_val(0.1, 0.1)
