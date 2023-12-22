import itertools
import os
import sys
from multiprocessing import Process

import cv2
import pandas as pd
import requests
from tqdm import tqdm

from config import URL_DATA_FILE, FILE_NAME, IMAGES_DIR, RAW_DATA_DIR, DATA_DIR, ABNORMAL_INDICES, VIDEO_PATH


def download_data(url):
    """
    Downloads a file from a remote URL and saves it to a local directory.

    Returns:
        None
    """
    response = requests.get(url, stream=True)

    # Get the size of the remote file
    remote_file_size = int(response.headers.get("content-length", 0))
    file_name = os.path.basename(url)
    video_path = os.path.join(RAW_DATA_DIR, file_name)
    
    # Get the size of the local file if it exists
    local_file_size = 0
    if os.path.isfile(video_path):
        local_file_size = os.stat(video_path).st_size

    # Check if the file is already downloaded
    if local_file_size == remote_file_size:
        print(f"File {file_name} is already downloaded!")
        return

    print(f"File {file_name} is not fully downloaded...")

    # Check if the file needs to be downloaded
    if not os.path.isfile(video_path) or local_file_size != remote_file_size:
        print(f"Downloading...")
        with open(video_path, "wb") as handle:
            for data in tqdm(
                    response.iter_content(),
                    desc=f"{file_name}",
                    unit="B",
                    unit_scale=True,
                    colour="green",
                    smoothing=0.2,
                    file=sys.stdout,
                    total=remote_file_size,
            ):
                handle.write(data)


def extract_images_from_video(video_path):
    """
    Extract frames from a video and save them as images.
    """
    
    print("Extract frames to images...")
    file_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    procs = []
    failed = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if ret:
            proc = Process(target=_save_original_frame, args=(file_name, frame, idx))
            procs.append(proc)
            proc.start()
        else:
            failed.append(idx)
    for proc in procs:
        proc.join()
    print("failed frames: ", failed)
    print("Extract frames to images done!")


def _save_original_frame(file_name, frame, idx):
    """
    Save the original frame as an image.

    Args:
        frame (numpy array): The original frame.
        idx (int): The index of the frame.
    """
    procs = []

    # Generate a save path for the image
    save_path = os.path.join(IMAGES_DIR, f"{os.path.splitext(file_name)[0]}{str(idx).zfill(4)}.png")

    if not os.path.isfile(save_path):
        cv2.imwrite(save_path, frame)


# def apply_transformations(force=False) -> None:
#     """
#     Save preprocessed images.
#     This function retrieves the paths of unprocessed images and uses multiprocessing
#     to save each image in parallel. It prints a message when the process is complete.
#     """
#     print("Saving preprocessed images...")
#
#     # Retrieve paths of unprocessed images
#     procs = []
#     # Use multiprocessing to save each image in parallel
#     for file_name in os.listdir(IMAGES_DIR):
#         proc = Process(target=_save_preprocessed_image, args=(file_name, force))
#         procs.append(proc)
#         proc.start()
#     for proc in procs:
#         proc.join()
#     print("Save preprocessed images is done!")


# def _save_preprocessed_image(file_name, force=False):
#     """
#     Save a preprocessed image to a new directory.
#
#     Args:
#         file_name (str): The path to the original image.
#
#     Returns:
#         None
#     """
#     init_path = os.path.join(IMAGES_DIR, file_name)
#     save_path = os.path.join(PROC_IMAGES_DIR, file_name)
#     if not os.path.isfile(save_path) or force:
#         processed_image = get_image(image_path=init_path, do_preprocess=True)
#         cv2.imwrite(save_path, processed_image)


def load_to_csv():
    """
    Create a CSV file with image names and their corresponding labels.
    Returns:
        None
    """

    # Get a list of image names from the IMAGES_DIR directory and sort them
    image_names = sorted(os.listdir(IMAGES_DIR))

    # Get the labels for each image
    labels = [get_label(img) for img in image_names]
    df = pd.DataFrame(data={"image": image_names, "label": labels})
    df.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False)


# def get_image(image_path: str, do_preprocess: bool = False):
#     """
#     Load an image from a given path and preprocess it if necessary.
#
#     Args:
#         image_path: The path to the image file.
#         do_preprocess: Whether to preprocess the image.
#
#     Returns:
#         The loaded and preprocessed image, or None if the image could not be loaded.
#     """
#     # Load the image from the specified path
#     image = cv2.imread(image_path)
#     # Preprocess the image if required
#     if do_preprocess:
#         image = preprocess_image(image)
#     return image


def get_label(image_path: str) -> int:
    """
    Get the label for an image based on its path.
    Arguments:
    image_path -- the path of the image
    Returns:
    label -- the label of the image (0 for normal, 1 for abnormal)
    """
    # Generate a list of abnormal indices as strings with leading zeros
    indices_to_str = list(map(lambda x: str(x).zfill(4), itertools.chain(*ABNORMAL_INDICES)))

    # Check if the last four characters of the image path are in the list of abnormal indices
    if os.path.splitext(image_path)[0][-4:] not in indices_to_str:
        label = 1  # Normal image
    else:
        label = -1  # Abnormal image

    return label


if __name__ == '__main__':
    download_data("https://www.tensorflow.org/tutorials/keras/classification")
    extract_images_from_video(VIDEO_PATH)
    # time_func(apply_transformations)
    # load_to_csv()
