import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from config import IMAGES_DIR, DATA_DIR, ANOMALY_INDICES, TRAIN_DIR, VALIDATION_DIR, TEST_DIR


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


if __name__ == '__main__':
    split_train_val_test(0.20, 0.20)
