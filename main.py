import keras.losses
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, layers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

from config import IMAGES_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, EPOCHS, INPUT_SHAPE, VIDEO_FILE, SEED, LEARNING_RATE
from local_utils import inspect_video


def train_val_ds(global_batch_size):
    return tf.keras.utils.image_dataset_from_directory(
        directory=IMAGES_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["normal", "anomaly"],
        color_mode="rgb",
        batch_size=global_batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False,
        validation_split=0.2,
        subset="both",
        crop_to_aspect_ratio=False
    )


def train_model():
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    global_batch_size = (BATCH_SIZE * strategy.num_replicas_in_sync)
    with strategy.scope():
        train_ds, val_ds = train_val_ds(global_batch_size)

        train_ds = train_ds.cache().shuffle(train_ds.cardinality()).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        model = Sequential([
            layers.Rescaling(1. / 255, input_shape=INPUT_SHAPE),
            layers.SeparableConv2D(16, 3, depth_multiplier=5, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.SeparableConv2D(32, 3, depth_multiplier=5, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.SeparableConv2D(64, 3, depth_multiplier=5, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, "sigmoid"),
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.BinaryAccuracy(),
                               keras.metrics.BinaryIoU()]
                      )

        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=EPOCHS,
                            callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                                       keras.callbacks.ReduceLROnPlateau(patience=10, cooldown=5),
                                       keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True)])
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv("history.csv")
        return model


def metrics(model, val_dataset, threshold_type="pr"):
    df = pd.read_csv("history.csv")
    acc = df['binary_accuracy']
    val_acc = df['val_binary_accuracy']

    loss = df['loss']
    val_loss = df['val_loss']
    epochs_range = range(len(df))

    plt.figure(figsize=(9, 9))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    input_data = val_dataset.map(lambda x, y: x)
    y_true = np.asarray([label for label in val_dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()]).astype(int)

    y_probability = model.predict(input_data)
    fpr, tpr, roc_thresholds = roc_curve(y_true=y_true, y_score=y_probability)
    roc_threshold_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(tpr - 1, 2)))
    roc_threshold = roc_thresholds[roc_threshold_idx]
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_probability)
    pr_threshold_idx = np.argmax(np.sqrt(np.power(precision, 2) + np.power(recall, 2)))
    pr_threshold = pr_thresholds[pr_threshold_idx]
    ap = average_precision_score(y_true=y_true, y_score=y_probability)

    if threshold_type == "roc":
        threshold = roc_threshold
    elif threshold_type == "pr":
        threshold = pr_threshold
    else:
        raise ValueError(f"Unknown threshold value: '{threshold_type}'. Please select between 'roc' and 'pr'")

    y_prediction = np.where(y_probability >= threshold, 1, 0)

    np.save("threshold.npy", threshold)

    # Plot the ROC curve
    plt.subplot(2, 2, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {np.round(roc_auc, 4)})')
    plt.plot([0, fpr[-1]], [0, tpr[-1]], color='navy', lw=2, linestyle='--')
    plt.plot(fpr[roc_threshold_idx], tpr[roc_threshold_idx], c='green', marker='o',
             label=f'ROC threshold = {np.round(roc_threshold, 4)}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Plot the ROC curve
    plt.subplot(2, 2, 4)
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (Average Precision = {np.round(ap, 4)})')
    plt.plot(recall[pr_threshold_idx], precision[pr_threshold_idx] , c='red', marker='o',
             label=f'PR threshold = {np.round(pr_threshold, 4)}')

    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Print the classification report
    print(classification_report(y_true, y_prediction, labels=(0, 1), target_names=["normal", "abnormall"]))
    return threshold


if __name__ == '__main__':
    # model = train_model()
    _, val_ds = train_val_ds(100)
    # threshold = metrics(model, val_ds)
    model = keras.models.load_model("model.keras")
    threshold = metrics(model, val_ds)
    inspect_video(VIDEO_FILE, model, threshold=threshold)
