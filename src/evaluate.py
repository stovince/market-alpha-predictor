import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from tensorflow.keras.models import load_model

from prepare_data import prepare


def evaluate(symbol: str, model_dir: str = "models", results_dir: str = "results") -> None:
    os.makedirs(results_dir, exist_ok=True)
    data = prepare(symbol)
    model_path = os.path.join(model_dir, f"{symbol}_best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, f"{symbol}_final_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for {symbol} in {model_dir}")
    model = load_model(model_path)
    y_pred_prob = model.predict(data.X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(data.X_train, data.y_train)
    y_baseline = baseline.predict(data.X_test)
    metrics = {
        "accuracy": accuracy_score(data.y_test, y_pred),
        "precision": precision_score(data.y_test, y_pred, zero_division=0),
        "recall": recall_score(data.y_test, y_pred, zero_division=0),
        "f1": f1_score(data.y_test, y_pred, zero_division=0),
        "baseline_accuracy": accuracy_score(data.y_test, y_baseline),
    }
    results = {
        "symbol": symbol,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(data.y_test, y_pred).tolist(),
        "test_size": len(data.y_test),
    }
    with open(os.path.join(results_dir, f"{symbol}_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    plt.figure(figsize=(10, 4))
    plt.plot(y_pred_prob, label="Predicted probability")
    plt.plot(data.y_test.values, label="Observed direction", alpha=0.75)
    plt.title(f"{symbol} Test Predictions vs Observed Direction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{symbol}_predictions.png"))
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(data.y_test.values, label="Observed direction")
    plt.plot(y_pred, alpha=0.75, label="Predicted direction")
    plt.title(f"{symbol} Test Classification Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{symbol}_classification.png"))
    plt.close()
    print(f"Evaluation complete. Results saved to {results_dir}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained TensorFlow financial model")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(args.symbol, model_dir=args.model_dir, results_dir=args.results_dir)
