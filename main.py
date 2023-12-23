import keras.losses
import numpy as np
import tensorflow as tf
from keras import Sequential, layers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

from config import IMAGES_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, EPOCHS, INPUT_SHAPE, VIDEO_FILE, SEED
from local_utils import inspect_video

if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

global_batch_size = (BATCH_SIZE * strategy.num_replicas_in_sync)
with strategy.scope():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=IMAGES_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["normal", "anomaly"],
        color_mode="rgb",
        batch_size=global_batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="both",
        crop_to_aspect_ratio=True,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
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

    loss = keras.losses.CategoricalFocalCrossentropy()

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save('model.keras')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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

val_ds = val_ds.unbatch()

x_input = val_ds.map(lambda x, y: tf.expand_dims(x, axis=0))
y_true = np.array([sample.numpy() for sample in val_ds.map(lambda x, y: y)])
probas_pred = model.predict(x_input)


fpr, tpr, roc_thresholds = roc_curve(y_true=y_true, y_score=probas_pred)
roc_threshold_idx = np.argmin(np.sqrt(np.power(fpr - 0, 2) + np.power(tpr - 1, 2)))
roc_threshold = roc_thresholds[roc_threshold_idx]

prec, rec, pr_thresholds = precision_recall_curve(y_true=y_true, probas_pred=probas_pred)
pr_threshold_idx = np.argmin(np.sqrt(np.power(1 - prec - 0, 2) + np.power(rec - 1, 2)))
pr_threshold = pr_thresholds[pr_threshold_idx]
# Calculate Euclidean distances to the top-left corner [0, 1]

# Calculate the AUC
roc_auc = auc(fpr, tpr)
ap = average_precision_score(y_true=y_true, y_score=probas_pred)

# Plot the ROC curve
plt.subplot(2, 2, 3)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {np.round(roc_auc, 4)})')
plt.plot([0, fpr[-1]], [0, tpr[-1]], color='navy', lw=2, linestyle='--')
plt.plot(fpr[roc_threshold_idx], tpr[roc_threshold_idx], c='green', marker='o', label=f'ROC threshold = {np.round(roc_threshold, 4)}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Plot the ROC curve
plt.subplot(2, 2, 4)
plt.plot(1 - prec, rec, color='darkorange', lw=2, label=f'PR curve (Average Precision = {np.round(ap, 4)})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.plot(1 - prec[pr_threshold_idx], rec[pr_threshold_idx], c='red', marker='o', label=f'PR threshold = {np.round(pr_threshold, 4)}')

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

threshold = roc_threshold
# threshold = pr_threshold

# Convert probabilities to binary predictions
y_pred = np.where(probas_pred > threshold, 1, 0)

# Print the classification report
print(classification_report(y_true, y_pred, target_names=class_names))

inspect_video(VIDEO_FILE, model, threshold=threshold)
