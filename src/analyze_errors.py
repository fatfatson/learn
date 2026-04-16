"""
Analyze misclassified MNIST digits.

Finds all incorrectly classified test images, saves individual images,
and generates analysis visualizations to understand WHY the model fails.

Usage:
    python src/analyze_errors.py
    python src/analyze_errors.py --top_n 5          # Show top 5 errors per confusion pair
    python src/analyze_errors.py --save_individual   # Save each misclassified image separately
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from neural_network import NeuralNetwork
from data_loader import load_mnist


def find_misclassified(model, X_test, y_test):
    """Find all misclassified samples with their details."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    misclassified = []
    for i in range(len(y_test)):
        if predictions[i] != y_test[i]:
            misclassified.append({
                "index": i,
                "true_label": y_test[i],
                "pred_label": predictions[i],
                "confidence": probabilities[i][predictions[i]],
                "true_prob": probabilities[i][y_test[i]],
                "probabilities": probabilities[i],
            })

    # Sort by confidence (most confidently wrong first)
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    return misclassified


def save_individual_images(misclassified, X_test, output_dir):
    """Save each misclassified image as a separate file."""
    img_dir = os.path.join(output_dir, "misclassified_images")
    os.makedirs(img_dir, exist_ok=True)

    for item in misclassified:
        img = X_test[item["index"]].reshape(28, 28)
        filename = f"idx{item['index']:04d}_true{item['true_label']}_pred{item['pred_label']}_conf{item['confidence']:.3f}.png"

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {item['true_label']} → Pred: {item['pred_label']}\nConf: {item['confidence']:.3f}", fontsize=8)
        ax.axis("off")
        plt.savefig(os.path.join(img_dir, filename), dpi=100, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(misclassified)} individual images to {img_dir}/")


def plot_error_summary(misclassified, X_test, output_dir):
    """Generate overall error summary statistics."""
    # Count errors by true label
    error_counts_by_true = np.zeros(10, dtype=int)
    for item in misclassified:
        error_counts_by_true[item["true_label"]] += 1

    # Top confused pairs
    confusion_pairs = {}
    for item in misclassified:
        pair = (item["true_label"], item["pred_label"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    top_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:15]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Error count per digit
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(10), error_counts_by_true, color="salmon", edgecolor="darkred", alpha=0.8)
    ax1.set_xlabel("True Digit", fontsize=11)
    ax1.set_ylabel("Number of Errors", fontsize=11)
    ax1.set_title("Misclassification Count by True Digit", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(10))
    for bar, count in zip(bars, error_counts_by_true):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(count), ha="center", va="bottom", fontsize=9)

    # 2. Top confused pairs
    ax2 = fig.add_subplot(gs[0, 1])
    pair_labels = [f"{t}→{p}" for (t, p), _ in top_pairs]
    pair_counts = [c for _, c in top_pairs]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(pair_counts)))
    ax2.barh(range(len(pair_labels)), pair_counts, color=colors)
    ax2.set_yticks(range(len(pair_labels)))
    ax2.set_yticklabels(pair_labels, fontsize=9)
    ax2.set_xlabel("Number of Errors", fontsize=11)
    ax2.set_title("Top Confused Pairs (True→Pred)", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    for i, count in enumerate(pair_counts):
        ax2.text(count + 0.5, i, str(count), va="center", fontsize=9)

    # 3. Confidence distribution of wrong predictions
    ax3 = fig.add_subplot(gs[1, 0])
    confidences = [item["confidence"] for item in misclassified]
    true_probs = [item["true_prob"] for item in misclassified]
    ax3.hist(confidences, bins=30, alpha=0.7, color="salmon", edgecolor="darkred", label="Pred label prob")
    ax3.hist(true_probs, bins=30, alpha=0.7, color="steelblue", edgecolor="navy", label="True label prob")
    ax3.set_xlabel("Probability", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Confidence Distribution of Misclassified Samples", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)

    # 4. Average probability bar for wrong vs correct
    ax4 = fig.add_subplot(gs[1, 1])
    avg_wrong_conf = np.mean(confidences)
    avg_true_prob = np.mean(true_probs)
    categories = ["Avg pred label\nprob (wrong)", "Avg true label\nprob"]
    values = [avg_wrong_conf, avg_true_prob]
    bar_colors = ["salmon", "steelblue"]
    bars = ax4.bar(categories, values, color=bar_colors, edgecolor="gray", width=0.5)
    ax4.set_ylabel("Probability", fontsize=11)
    ax4.set_title("Average Probability for Misclassified", fontsize=12, fontweight="bold")
    ax4.set_ylim(0, 1.0)
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.savefig(os.path.join(output_dir, "error_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Error summary saved to {os.path.join(output_dir, 'error_summary.png')}")


def plot_top_confusions(misclassified, X_test, output_dir, top_n=5):
    """Show examples for the top confused digit pairs."""
    # Group by confusion pair
    confusion_groups = {}
    for item in misclassified:
        pair = (item["true_label"], item["pred_label"])
        if pair not in confusion_groups:
            confusion_groups[pair] = []
        confusion_groups[pair].append(item)

    # Sort groups by count
    sorted_groups = sorted(confusion_groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Take top confusion pairs
    num_pairs = min(6, len(sorted_groups))
    fig, axes = plt.subplots(num_pairs, top_n + 1, figsize=(2.5 * (top_n + 1), 2.5 * num_pairs))

    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    for row_idx in range(num_pairs):
        pair, items = sorted_groups[row_idx]
        true_label, pred_label = pair

        # Sort by confidence (most confidently wrong first)
        items_sorted = sorted(items, key=lambda x: x["confidence"], reverse=True)

        for col_idx in range(top_n + 1):
            ax = axes[row_idx, col_idx]
            if col_idx < min(top_n, len(items_sorted)):
                item = items_sorted[col_idx]
                img = X_test[item["index"]].reshape(28, 28)
                ax.imshow(img, cmap="gray")
                conf_str = f"conf={item['confidence']:.2f}"
                true_prob_str = f"true_prob={item['true_prob']:.2f}"
                ax.set_title(f"#{item['index']}\n{conf_str}\n{true_prob_str}", fontsize=7, color="red")
            else:
                ax.text(0.5, 0.5, f"+{len(items) - top_n}\nmore", ha="center", va="center",
                        fontsize=14, color="gray", transform=ax.transAxes)
            ax.axis("off")

        # Row label
        axes[row_idx, 0].set_ylabel(
            f"True {true_label}\n↓\nPred {pred_label}\n({len(items)} errors)",
            fontsize=10, fontweight="bold", rotation=0, labelpad=70,
            color="darkred"
        )

    plt.suptitle("Top Misclassified Digit Pairs (most confidently wrong first)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_confusions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top confusions saved to {os.path.join(output_dir, 'top_confusions.png')}")


def plot_confidence_heatmap(misclassified, output_dir):
    """Show average confidence for each confusion pair."""
    # Build confidence matrix: avg predicted probability for each (true, pred) pair
    conf_sum = np.zeros((10, 10))
    conf_count = np.zeros((10, 10))

    for item in misclassified:
        t, p = item["true_label"], item["pred_label"]
        conf_sum[t][p] += item["confidence"]
        conf_count[t][p] += 1

    # Average confidence (only where errors exist)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_conf = np.where(conf_count > 0, conf_sum / conf_count, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Error count heatmap
    error_matrix = conf_count.copy()
    im1 = ax1.imshow(error_matrix, cmap="YlOrRd", vmin=0)
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_title("Error Count per (True, Pred) Pair", fontsize=13, fontweight="bold")
    for i in range(10):
        for j in range(10):
            if error_matrix[i][j] > 0:
                ax1.text(j, i, int(error_matrix[i][j]), ha="center", va="center",
                         fontsize=8, color="white" if error_matrix[i][j] > error_matrix.max() / 2 else "black")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Average confidence heatmap
    im2 = ax2.imshow(avg_conf, cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xlabel("Predicted Label", fontsize=12)
    ax2.set_ylabel("True Label", fontsize=12)
    ax2.set_title("Avg Confidence of Wrong Prediction", fontsize=13, fontweight="bold")
    for i in range(10):
        for j in range(10):
            if conf_count[i][j] > 0:
                ax2.text(j, i, f"{avg_conf[i][j]:.2f}\n(n={int(conf_count[i][j])})",
                         ha="center", va="center", fontsize=7,
                         color="white" if avg_conf[i][j] > 0.8 else "black")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confidence heatmap saved to {os.path.join(output_dir, 'confidence_heatmap.png')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze misclassified MNIST digits")
    parser.add_argument("--model_path", type=str, default="models/mnist_net.npz")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models/error_analysis")
    parser.add_argument("--top_n", type=int, default=5, help="Number of examples per confusion pair")
    parser.add_argument("--save_individual", action="store_true", help="Save each misclassified image separately")
    args = parser.parse_args()

    # Load model and data
    print("Loading model and data...")
    model = NeuralNetwork.load(args.model_path)
    data = load_mnist(data_dir=args.data_dir, normalize=True, one_hot=True, validation_size=0)

    # Find misclassified samples
    print("\nFinding misclassified samples...")
    misclassified = find_misclassified(model, data["X_test"], data["y_test"])

    total = len(data["y_test"])
    error_rate = len(misclassified) / total * 100
    print(f"\nTotal test samples: {total}")
    print(f"Misclassified: {len(misclassified)} ({error_rate:.2f}%)")
    print(f"Accuracy: {100 - error_rate:.2f}%")

    # Print top confused pairs
    confusion_pairs = {}
    for item in misclassified:
        pair = (item["true_label"], item["pred_label"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    top_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 confused pairs (True → Pred):")
    for (t, p), count in top_pairs:
        print(f"  {t} → {p}: {count} errors")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    print("\nGenerating analysis visualizations...")
    plot_error_summary(misclassified, data["X_test"], args.output_dir)
    plot_top_confusions(misclassified, data["X_test"], args.output_dir, top_n=args.top_n)
    plot_confidence_heatmap(misclassified, args.output_dir)

    if args.save_individual:
        save_individual_images(misclassified, data["X_test"], args.output_dir)

    # Print some "most confidently wrong" examples
    print("\n" + "=" * 60)
    print("Top 10 most confidently wrong predictions:")
    print("=" * 60)
    for item in misclassified[:10]:
        img = data["X_test"][item["index"]].reshape(28, 28)
        # Compute pixel density as a rough measure
        density = img.mean()
        print(f"  Sample #{item['index']:4d}: True={item['true_label']} → Pred={item['pred_label']}  "
              f"conf={item['confidence']:.3f}  true_prob={item['true_prob']:.3f}  "
              f"pixel_density={density:.3f}")

    print(f"\nAll analysis results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
