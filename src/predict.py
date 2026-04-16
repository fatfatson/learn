"""
Predict handwritten digits using a trained model.

Supports:
  - Predict on MNIST test set with accuracy report
  - Visualize predictions on random test samples
  - Predict on a custom image file (28x28 grayscale)

Usage:
    python src/predict.py --eval                     # Evaluate on test set
    python src/predict.py --visualize                # Show predictions on samples
    python src/predict.py --image my_digit.png       # Predict a custom image
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from data_loader import load_mnist


def evaluate(model_path, data_dir):
    """Evaluate model on MNIST test set."""
    model = NeuralNetwork.load(model_path)
    data = load_mnist(data_dir=data_dir, normalize=True, one_hot=True, validation_size=0)

    test_acc = model.accuracy(data["X_test"], data["y_test"])
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Per-class accuracy
    predictions = model.predict(data["X_test"])
    y_true = data["y_test"]
    print("\nPer-class accuracy:")
    for digit in range(10):
        mask = y_true == digit
        if mask.sum() > 0:
            acc = np.mean(predictions[mask] == digit)
            print(f"  Digit {digit}: {acc:.4f} ({mask.sum()} samples)")

    return test_acc


def visualize_predictions(model_path, data_dir, num_samples=20, save_path="models/predictions.png"):
    """Visualize model predictions on random test samples."""
    model = NeuralNetwork.load(model_path)
    data = load_mnist(data_dir=data_dir, normalize=True, one_hot=True, validation_size=0)

    # Random sample indices
    n_test = data["X_test"].shape[0]
    indices = np.random.choice(n_test, num_samples, replace=False)

    X_samples = data["X_test"][indices]
    y_true = data["y_test"][indices]
    predictions = model.predict(X_samples)
    probabilities = model.predict_proba(X_samples)

    # Plot
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = X_samples[i].reshape(28, 28)
            ax.imshow(img, cmap="gray")
            true_label = y_true[i]
            pred_label = predictions[i]
            confidence = probabilities[i][pred_label]

            color = "green" if pred_label == true_label else "red"
            ax.set_title(
                f"True: {true_label} | Pred: {pred_label} ({confidence:.2f})",
                color=color, fontsize=9,
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Predictions visualization saved to {save_path}")


def visualize_confusion_matrix(model_path, data_dir, save_path="models/confusion_matrix.png"):
    """Plot confusion matrix."""
    model = NeuralNetwork.load(model_path)
    data = load_mnist(data_dir=data_dir, normalize=True, one_hot=True, validation_size=0)

    predictions = model.predict(data["X_test"])
    y_true = data["y_test"]

    # Build confusion matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, predictions):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=np.arange(10), yticklabels=np.arange(10),
           title="Confusion Matrix",
           xlabel="Predicted Label", ylabel="True Label")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def predict_image(model_path, image_path):
    """Predict a digit from a custom image file."""
    model = NeuralNetwork.load(model_path)

    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow is required for custom image prediction.")
        print("Install with: pip install Pillow")
        return

    # Load and preprocess image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28

    # Invert if needed (MNIST has white digits on black background)
    img_array = np.array(img, dtype=np.float32) / 255.0
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    # Flatten and predict
    X = img_array.reshape(1, 784)
    probs = model.predict_proba(X)[0]
    prediction = np.argmax(probs)

    print(f"\nPredicted digit: {prediction}")
    print("Class probabilities:")
    for digit in range(10):
        bar = "█" * int(probs[digit] * 50)
        print(f"  {digit}: {probs[digit]:.4f} {bar}")

    # Show image and probability chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img_array, cmap="gray")
    ax1.set_title(f"Input Image (Predicted: {prediction})")
    ax1.axis("off")

    ax2.bar(range(10), probs, color="steelblue")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Probabilities")
    ax2.set_xticks(range(10))

    plt.tight_layout()
    save_path = os.path.splitext(image_path)[0] + "_prediction.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST digit prediction")
    parser.add_argument("--model_path", type=str, default="models/mnist_net.npz", help="Model file path")
    parser.add_argument("--data_dir", type=str, default="data", help="MNIST data directory")
    parser.add_argument("--eval", action="store_true", help="Evaluate on test set")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--confusion", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--image", type=str, default=None, help="Path to custom image for prediction")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to visualize")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.model_path, args.data_dir)

    if args.visualize:
        visualize_predictions(
            args.model_path, args.data_dir,
            num_samples=args.num_samples,
        )

    if args.confusion:
        visualize_confusion_matrix(args.model_path, args.data_dir)

    if args.image:
        predict_image(args.model_path, args.image)

    # If no action specified, run all evaluations
    if not any([args.eval, args.visualize, args.confusion, args.image]):
        print("No action specified. Running full evaluation...\n")
        evaluate(args.model_path, args.data_dir)
        visualize_predictions(args.model_path, args.data_dir, num_samples=args.num_samples)
        visualize_confusion_matrix(args.model_path, args.data_dir)


if __name__ == "__main__":
    main()
