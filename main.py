import keras.losses
import numpy as np
import tensorflow as tf
from keras import Sequential, layers
from matplotlib import pyplot as plt

from config import IMAGES_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, EPOCHS, INPUT_SHAPE, VIDEO_FILE
from local_utils import inspect_video
from metrics import get_auc

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=IMAGES_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "anomaly"],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    # shuffle=True,
    seed=1,
    validation_split=0.2,
    subset="both",
    crop_to_aspect_ratio=True
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

loss = keras.losses.BinaryFocalCrossentropy()

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
plt.show()

labels = np.asarray(list(val_ds.unbatch().map(lambda x, y: y).as_numpy_iterator()))
predictions = model.predict(val_ds.map(lambda x, y: x))
get_auc(labels=labels, predictions=predictions)
inspect_video(VIDEO_FILE, model)
