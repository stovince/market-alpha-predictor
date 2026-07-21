import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp_model(input_shape: int, dropout_rate: float = 0.2) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
