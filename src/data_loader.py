"""
MNIST data loader.

Downloads and loads the MNIST dataset from the internet,
splits into train/validation/test sets, and provides
utilities for preprocessing and batching.
"""

import os
import struct
import gzip
import urllib.request
import numpy as np


MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_file(url, save_dir):
    """Download a file from URL to save_dir."""
    filename = url.split("/")[-1]
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"  Already exists: {filename}")
        return filepath

    print(f"  Downloading: {filename}...")
    os.makedirs(save_dir, exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    print(f"  Done: {filename}")
    return filepath


def load_mnist(data_dir="data", normalize=True, one_hot=True, validation_size=5000):
    """
    Load MNIST dataset.

    Args:
        data_dir: Directory to store/load data
        normalize: Whether to normalize pixel values to [0, 1]
        one_hot: Whether to one-hot encode labels
        validation_size: Number of training samples to use for validation

    Returns:
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    print("Loading MNIST dataset...")

    # Download files if needed
    filepaths = {}
    for key, url in MNIST_URLS.items():
        filepaths[key] = download_file(url, data_dir)

    # Load images
    X_train = _load_images(filepaths["train_images"])
    X_test = _load_images(filepaths["test_images"])

    # Load labels
    y_train = _load_labels(filepaths["train_labels"])
    y_test = _load_labels(filepaths["test_labels"])

    # Normalize pixel values
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    # Split training set into train and validation
    if validation_size > 0:
        X_val = X_train[-validation_size:]
        y_val = y_train[-validation_size:]
        X_train = X_train[:-validation_size]
        y_train = y_train[:-validation_size]
    else:
        X_val = X_test
        y_val = y_test

    # One-hot encode labels
    if one_hot:
        y_train_oh = _one_hot(y_train, 10)
        y_val_oh = _one_hot(y_val, 10)
        y_test_oh = _one_hot(y_test, 10)
    else:
        y_train_oh = y_train
        y_val_oh = y_val
        y_test_oh = y_test

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_oh": y_train_oh,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_oh": y_val_oh,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_oh": y_test_oh,
    }


def _load_images(filepath):
    """Load MNIST images from gzipped idx file."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images


def _load_labels(filepath):
    """Load MNIST labels from gzipped idx file."""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def _one_hot(labels, num_classes):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


class DataIterator:
    """Mini-batch data iterator with shuffling."""

    def __init__(self, X, y, batch_size=64, shuffle=True):
        """
        Args:
            X: Input data, shape (n_samples, n_features)
            y: Labels, shape (n_samples, n_classes) or (n_samples,)
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle data each epoch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
