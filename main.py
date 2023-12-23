import keras.losses
import numpy as np
import tensorflow as tf
from keras import Sequential, layers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve

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
        label_mode="categorical",
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
    a = train_ds.as_numpy_iterator()
    b = val_ds.as_numpy_iterator()

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
        layers.Dense(2, "softmax"),
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

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

labels = np.argmax(np.asarray(list(val_ds.unbatch().map(lambda x, y: y).as_numpy_iterator())), axis=-1)
pred = model.predict(val_ds.map(lambda x, y: x))
y_pred = pred[..., 1]
fpr, tpr, roc_thresholds = roc_curve(labels, y_pred)
threshold_idx = np.argmin(np.sqrt(np.power(fpr - 0, 2) + np.power(tpr - 1, 2)))
threshold = roc_thresholds[threshold_idx]

prec, rec, pr_thresholds = precision_recall_curve(y_true=labels, probas_pred=y_pred)
pr_threshold_idx = np.argmin(np.sqrt(np.power(prec - 0, 2) + np.power(rec - 1, 2)))
pr_threshold = pr_thresholds[pr_threshold_idx]
# Calculate Euclidean distances to the top-left corner [0, 1]

# Calculate the AUC
roc_auc = auc(fpr, tpr)
pr_auc = auc(prec, rec)

# Plot the ROC curve
plt.figure(1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {np.round(roc_auc, 2)})\n'
                                                   f'threshold: {threshold}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.plot(fpr[threshold_idx], tpr[threshold_idx], c='red', marker='o', label=f'Threshold = {threshold}', s=100)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Plot the ROC curve
plt.figure(2)
plt.plot(prec, rec, color='darkorange', lw=2, label=f'PR curve (area = {np.round(pr_auc, 2)})\n'
                                                   f'threshold: {pr_threshold}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.plot(prec[pr_threshold_idx], rec[pr_threshold_idx], c='red', marker='o', label=f'Threshold = {pr_threshold}', s=100)

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')

# Convert probabilities to binary predictions
pred = [1 if p > threshold else 0 for p in y_pred]

# Convert labels and predictions to numpy arrays
labels = np.asarray(labels, dtype=np.int32)
pred = np.asarray(pred, dtype=np.int32)
plt.show()

# Print the classification report
print(classification_report(labels, pred, target_names=class_names))

inspect_video(VIDEO_FILE, model, threshold=threshold)
