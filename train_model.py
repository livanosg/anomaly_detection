import concurrent
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from config import IMAGES_DIR
from utils.prepare_data import get_train_test_data

X_train, X_test, y_train, y_test = get_train_test_data()

# from github
def preproces_per_image(file_path):
    full_path = os.path.join(IMAGES_DIR, file_path)
    image = cv2.imread(full_path)
    image = cv2.blur(image, ksize=(9, 9))
    height, width, channels = image.shape
    ratio = 0.3
    image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    channel_slices = cv2.split(image)
    channels_hists = []
    for _slice in channel_slices:
        # Calculate the histogram for each channel
        channel_histogram = cv2.calcHist(images=_slice, channels=[0], mask=None, histSize=[255], ranges=(0, 256))
        channel_histogram = np.reshape(channel_histogram, (1, -1))
        channels_hists.append(channel_histogram)
        # Concatenate the channel histograms along the last axis
        histogram = np.concatenate(channels_hists, axis=-1)
        cv2.normalize(histogram, histogram)
    return histogram


def preprocess(images_list):
    histograms = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(preproces_per_image, file_path) for file_path in images_list]
    for future in concurrent.futures.as_completed(futures):
        histograms.append(future.result())
    histograms = np.concatenate(histograms, axis=0)
    return histograms


if __name__ == '__main__':
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    model = SVC(kernel="sigmoid")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
