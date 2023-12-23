import keras.losses
import numpy as np
import tensorflow as tf
from keras import Sequential, layers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

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

    print(b.next())
    exit()

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
predictions = model.predict(val_ds.map(lambda x, y: x))
y_prob = predictions[..., 1]
print("y_prob.shape", y_prob.shape)
fpr, tpr, _ = roc_curve(labels, y_prob)
# Calculate the AUC
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Convert probabilities to binary predictions
predictions = [1 if p > 0.5 else 0 for p in y_prob]

# Convert labels and predictions to numpy arrays
labels = np.asarray(labels, dtype=np.int32)
predictions = np.asarray(predictions, dtype=np.int32)
plt.show()

# Print the classification report
print(classification_report(labels, predictions, target_names=class_names))

inspect_video(VIDEO_FILE, model)
