from keras.models import Sequential
from keras.layers import SeparableConv2D, MaxPooling2D, Convolution2DTranspose, GlobalAveragePooling2D, Dense, \
    Conv2D, Dropout, BatchNormalization, Input


def classifier(input_shape):
    return Sequential([
        Input(input_shape),
        Conv2D(64, 3, padding='same', activation='relu'),  # 224
        Dropout(0.2),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),  # 224
        Dropout(0.2),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),  # 224
        Dropout(0.2),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(256, 3, padding='same'),  # 224
        GlobalAveragePooling2D(),
        Dense(2, activation='softmax')
    ])


def autoencoder(input_shape):
    return Sequential(
        [Input(shape=input_shape),
         Conv2D(128, 3, padding='same', activation='relu'),  # 224
         MaxPooling2D(),  # (112, 112)
         Conv2D(64, 3, padding='same', activation='relu'),  # 224
         Conv2D(32, 3, padding='same', activation='relu'),  # 224
         MaxPooling2D(),  # (56, 56)
         Conv2D(32, 3, padding='same', activation='relu'),  # 224
         MaxPooling2D(),  # (28, 28)
         Conv2D(16, 3, padding='same', activation='relu'),  # 224
         Convolution2DTranspose(16, 3, (2, 2), padding="same", activation="relu"),  # (56, 56)
         Convolution2DTranspose(32, 3, (2, 2), padding="same", activation="relu"),  # (112, 112)
         Conv2D(64, 3, padding='same', activation='relu'),  # 224
         Convolution2DTranspose(128, 7, (2, 2), padding="same", activation="relu"),  # (224, 224)
         SeparableConv2D(3, 3, depth_multiplier=3, padding="same", activation="sigmoid")  # (224, 224)
         ])
