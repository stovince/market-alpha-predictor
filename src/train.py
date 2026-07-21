<<<<<<< HEAD

=======
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from prepare_data import prepare
from model import build_mlp_model


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def train(symbol: str, epochs: int = 50, batch_size: int = 32, model_dir: str = "models") -> None:
    set_seed()
    os.makedirs(model_dir, exist_ok=True)
    data = prepare(symbol)
    model = build_mlp_model(input_shape=len(data.feature_columns))
    checkpoint_path = os.path.join(model_dir, f"{symbol}_best_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(os.path.join(model_dir, f"{symbol}_final_model.h5"))
    np.save(os.path.join(model_dir, f"{symbol}_history.npy"), history.history)
    print(f"Training complete. Model and history saved to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TensorFlow financial model")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()
    train(args.symbol, epochs=args.epochs, batch_size=args.batch_size, model_dir=args.model_dir)
>>>>>>> feature/tf-research-pipeline
