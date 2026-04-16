"""
Train a neural network on MNIST handwritten digits.

Usage:
    python src/train.py                        # Train with default settings
    python src/train.py --epochs 30 --lr 0.05  # Custom hyperparameters
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from data_loader import load_mnist, DataIterator


def train(args):
    """Main training loop."""
    # Load data
    data = load_mnist(
        data_dir=args.data_dir,
        normalize=True,
        one_hot=True,
        validation_size=args.val_size,
    )

    # Create model
    layer_sizes = [784] + [args.hidden_size] * args.hidden_layers + [10]
    print(f"\nNetwork architecture: {layer_sizes}")
    print(f"Learning rate: {args.lr}")
    print(f"Momentum: {args.momentum}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}\n")

    model = NeuralNetwork(layer_sizes, learning_rate=args.lr, momentum=args.momentum)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Learning rate decay
        if args.lr_decay and epoch > 0 and epoch % args.lr_decay_epoch == 0:
            model.learning_rate *= 0.5
            print(f"  Learning rate decayed to {model.learning_rate:.6f}")

        # Training
        train_iter = DataIterator(
            data["X_train"], data["y_train_oh"],
            batch_size=args.batch_size, shuffle=True
        )

        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_iter:
            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        val_pred = model.forward(data["X_val"])
        val_loss = model.compute_loss(val_pred, data["y_val_oh"])
        val_losses.append(val_loss)

        val_acc = model.accuracy(data["X_val"], data["y_val"])
        val_accuracies.append(val_acc)

        train_acc = model.accuracy(data["X_train"], data["y_train"])
        train_accuracies.append(train_acc)

        elapsed = time.time() - epoch_start

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.model_path)

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    # Test with best model
    print("\n--- Final Evaluation ---")
    best_model = NeuralNetwork.load(args.model_path)
    test_acc = best_model.accuracy(data["X_test"], data["y_test"])
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_accuracies, train_accuracies, args.plot_path)

    return best_model, test_acc


def plot_training_curves(train_losses, val_losses, val_accuracies, train_accuracies, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accuracies, "b-", label="Train Accuracy", linewidth=1.5)
    ax2.plot(epochs, val_accuracies, "r-", label="Val Accuracy", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nTraining curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model_path", type=str, default="models/mnist_net.npz", help="Model save path")
    parser.add_argument("--plot_path", type=str, default="models/training_curves.png", help="Plot save path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--val_size", type=int, default=5000, help="Validation set size")
    parser.add_argument("--lr_decay", action="store_true", help="Enable learning rate decay")
    parser.add_argument("--lr_decay_epoch", type=int, default=10, help="Epoch interval for LR decay")
    args = parser.parse_args()

    # Create output directories
    import os
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
