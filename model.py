from keras import Sequential, Input
from keras.layers import SeparableConv2D, MaxPooling2D, Convolution2DTranspose, GlobalAveragePooling2D, Dense


def supervised_anomaly_detector(input_shape):
    return Sequential([
        Input(shape=input_shape, name="input"),
        SeparableConv2D(128, 3, depth_multiplier=4, padding='same', activation='relu', kernel_initializer="he_uniform"),  # 224
        MaxPooling2D(),  # 112
        SeparableConv2D(128, 3, depth_multiplier=4, padding='same', activation='relu', kernel_initializer="he_uniform"),  # 112
        MaxPooling2D(),  # 56
        SeparableConv2D(256, 3, depth_multiplier=2, padding='same', activation='relu', kernel_initializer="he_uniform"),  # 56
        MaxPooling2D(),  # 28
        SeparableConv2D(512, 3, depth_multiplier=2, padding='same', activation='relu', kernel_initializer="he_uniform"),  # 28
        SeparableConv2D(128, 3, depth_multiplier=2, padding='same', activation='relu', kernel_initializer="he_uniform"),  # 28
        SeparableConv2D(128, 3, depth_multiplier=2, padding='same', kernel_initializer="he_uniform"),  # 56
        GlobalAveragePooling2D(),
        Dense(2, activation='softmax', name="output")
    ])


def unsupervised_anomaly_detector(input_shape):
    return Sequential(
        [Input(shape=input_shape),
         SeparableConv2D(128, 7, depth_multiplier=5, padding="same", activation="relu"),  # (224, 224)
         MaxPooling2D(),  # (112, 112)
         SeparableConv2D(64, 5, depth_multiplier=4, padding="same", activation="relu"),  # (112, 112)
         SeparableConv2D(32, 3, depth_multiplier=3, padding="same", activation="relu"),  # (112, 112)
         MaxPooling2D(),  # (56, 56)
         SeparableConv2D(32, 3, depth_multiplier=2, padding="same", activation="relu"),  # (56, 56)
         MaxPooling2D(),  # (28, 28)
         SeparableConv2D(16, 3, depth_multiplier=1, padding="same", activation="relu"),  # (28, 28)
         Convolution2DTranspose(32, 3, (2, 2), padding="same", activation="relu"),  # (56, 56)
         Convolution2DTranspose(32, 3, (2, 2), padding="same", activation="relu"),  # (112, 112)
         SeparableConv2D(64, 5, depth_multiplier=2, padding="same", activation="relu"),  # (112, 112)
         Convolution2DTranspose(128, 7, (2, 2), padding="same", activation="relu"),  # (224, 224)
         SeparableConv2D(3, 3, depth_multiplier=3, padding="same", activation="sigmoid")  # (224, 224)
         ])
