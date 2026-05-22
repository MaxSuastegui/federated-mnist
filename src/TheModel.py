import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds a CNN model for MNIST classification.

    This model is intentionally different from the class example.
    It uses two convolutional blocks with BatchNormalization and Dropout
    to improve generalization during local federated training.
    """

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.30),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.40),

            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
