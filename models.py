from keras.models import Sequential
from keras.layers import MaxPooling2D, Convolution2DTranspose, GlobalAveragePooling2D, Dense, \
    Conv2D, Dropout, BatchNormalization, Input


def classifier(input_shape):
    """
    Create a convolutional neural network classifier.

    Parameters:
    - input_shape (tuple): The shape of input data (height, width, channels).

    Returns:
    - Sequential: Keras Sequential model representing the classifier.
    """
    return Sequential([
        Input(input_shape),
        Conv2D(64, 3, padding="same", activation="relu"),  # 224
        Dropout(0.2),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),  # 112
        Dropout(0.2),
        MaxPooling2D(),
        Conv2D(128, 3, padding="same", activation="relu"),  # 56
        Dropout(0.2),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(256, 3, padding="same"),  # 28
        GlobalAveragePooling2D(),
        Dense(2, activation="softmax")
    ])


def autoencoder(input_shape):
    """
        Create a convolutional neural network autoencoder.

        Parameters:
        - input_shape (tuple): The shape of input data (height, width, channels).

        Returns:
        - Sequential: Keras Sequential model representing the autoencoder.
        """
    return Sequential(
        [Input(shape=input_shape),
         Conv2D(64, 3, activation="relu", padding="same"),  # 224
         MaxPooling2D(),
         Conv2D(64, 3, activation="relu", padding="same"),  # 112
         MaxPooling2D(),
         Conv2D(128, 3, activation="relu", padding="same"),  # 56
         MaxPooling2D(),
         Conv2D(512, 3, activation="relu", padding="same"),  # 56
         Convolution2DTranspose(128, 3, (2, 2), activation="relu", padding="same"),  # 56
         Convolution2DTranspose(64, 3, (2, 2), activation="relu", padding="same"),  # 112
         Convolution2DTranspose(64, 3, (2, 2), activation="relu", padding="same"),  # 224
         Conv2D(3, 3, activation="sigmoid", padding="same")  # 224
         ])
