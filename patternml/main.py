# patternml/main.py

import os
import logging

# -------------------------
# Suppress warnings (must be first)
# -------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("absl").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from patternml.factory.model_factory import ModelFactory
from patternml.template.training_pipeline import TrainingPipeline


def run_model(model_name, X_train, X_test, y_train, y_test, input_dim=None):
    print("\n" + "=" * 60)
    print(f"Running model: {model_name}")
    print("=" * 60)

    model = ModelFactory.create_model(model_name, input_dim=input_dim)
    pipeline = TrainingPipeline(model)

    metrics = pipeline.run(X_train, X_test, y_train, y_test)

    return metrics


def main():
    # -------------------------
    # Load dataset
    # -------------------------
    data = load_digits()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # Run experiments
    # -------------------------
    results = {}

    results["logistic_regression"] = run_model(
        "logistic_regression",
        X_train, X_test, y_train, y_test
    )

    results["knn"] = run_model(
        "knn",
        X_train, X_test, y_train, y_test
    )

    results["random_forest"] = run_model(
        "random_forest",
        X_train, X_test, y_train, y_test
    )

    results["ann"] = run_model(
        "ann",
        X_train, X_test, y_train, y_test,
        input_dim=X_train.shape[1]
    )

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    # -------------------------
    # Visualization (Accuracy only)
    # -------------------------
    model_names = list(results.keys())
    accuracies = [m["accuracy"] for m in results.values()]

    colors = ['skyblue', 'lightgreen', 'salmon', 'orange']
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, accuracies, color=colors)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha="center")

    plt.show()


if __name__ == "__main__":
    main()