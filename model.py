import numpy as np
from keras import Sequential, layers

from hyper_parameters import INPUT_SHAPE

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


def classifier(model, x_input, threshold):
    y_output = model.predict(x_input)
    y_pred = np.greater_equal(y_output, threshold).astype(int)
    return y_pred
