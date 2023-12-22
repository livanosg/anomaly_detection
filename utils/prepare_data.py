import itertools
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import IMAGES_DIR, ABNORMAL_INDICES, CSV_FILE


def split_to_classes(image_dir):
    indices_to_str = list(map(lambda x: str(x).zfill(4), itertools.chain(*ABNORMAL_INDICES)))
    for image_file in os.listdir(image_dir):
        if os.path.splitext(image_file)[0][-4:] in indices_to_str:
            dir_name = "anomalies"
        else:
            dir_name = "normal"
        os.renames(os.path.join(image_dir, image_file), os.path.join(image_dir, dir_name, image_file))


def save_to_csv():
    """
    Create a CSV file with image names and their corresponding labels.
    Returns:
        None
    """
    anomalies_files = os.listdir(os.path.join(IMAGES_DIR, 'anomalies'))
    anomalies_files = list(map(lambda x: os.path.join('anomalies', x), anomalies_files))
    normal_files = os.listdir(os.path.join(IMAGES_DIR, 'normal'))
    normal_files = list(map(lambda x: os.path.join('normal', x), normal_files))
    
    anomalies_df = pd.DataFrame(data={"image": anomalies_files, "label": 0})
    normal_df = pd.DataFrame(data={"image": normal_files, "label": 1})
    all_df = pd.concat([anomalies_df, normal_df], axis=0, ignore_index=True)
    
    all_df.to_csv(CSV_FILE, index=False)


def get_train_test_data():
    df = pd.read_csv(CSV_FILE)
    X = df["image"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # download_data("https://www.tensorflow.org/tutorials/keras/classification")
    # extract_images_from_video(VIDEO_PATH)
    # split_to_classes(IMAGES_DIR)
    save_to_csv()
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # print(X_train)
