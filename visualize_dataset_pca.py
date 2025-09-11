#!/usr/bin/env python3
"""
MNIST PCA Visualization Script

This script loads the MNIST dataset, fits PCA on the full dataset (or all selected classes),
and visualizes chosen subsets of classes without retraining PCA.
Results are fully reproducible due to fixed random seeds.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


# Dataset configuration
DATASET_INFO = {
    "MNIST": {
        "class": torchvision.datasets.MNIST,
        "num_classes": 10,
        "image_size": (28, 28, 1),
        "classes": [str(i) for i in range(10)],
    }
}


def set_seed(seed: int = 42):
    """Ensure reproducibility across NumPy, Torch, and PCA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_mnist_pca(selected_classes=None, num_samples=60000, save=True, seed: int = 42):
    """
    Process MNIST dataset, fit PCA on the whole dataset (or all selected classes),
    and plot only the requested classes.

    Parameters
    ----------
    selected_classes : list or None
        List of class indices to visualize (e.g., [0, 1, 2]).
        If None, show all classes.
    num_samples : int
        Number of samples to use (default=60000, full MNIST).
    save : bool
        Whether to save the resulting plot as PNG.
    seed : int
        Random seed for reproducibility.
    """
    set_seed(seed)

    dataset_name = "MNIST"
    ds_info = DATASET_INFO[dataset_name]
    print(f"Processing {dataset_name} with seed={seed}...")

    # Transform to tensor
    transform = transforms.ToTensor()

    # Load training dataset
    dataset = ds_info["class"](root="./data", train=True, download=True, transform=transform)

    # Limit samples (seeded)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    loader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1).numpy()
    labels = labels.numpy()

    # Fit PCA on ALL available samples
    print("Fitting PCA on full dataset...")
    reducer = PCA(n_components=2, random_state=seed)
    embedding = reducer.fit_transform(images)

    # Select classes for visualization
    if selected_classes is not None:
        selected_classes = [
            int(c) for c in selected_classes if int(c) in range(ds_info["num_classes"])
        ]
        if not selected_classes:
            raise ValueError("No valid classes selected.")
        mask = np.isin(labels, selected_classes)
        embedding = embedding[mask]
        labels = labels[mask]

    # Get explained variances
    var1 = reducer.explained_variance_ratio_[0] * 100
    var2 = reducer.explained_variance_ratio_[1] * 100

    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    num_unique = len(unique_labels)
    colormap = matplotlib.colormaps["hsv"]

    for i, label_idx in enumerate(unique_labels):
        mask = labels == label_idx
        label_name = ds_info["classes"][label_idx]
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=colormap(i / num_unique),
            label=label_name,
            s=5,
            alpha=0.7,
        )

    plt.title(f"PCA 2D Projection of MNIST (Classes: {selected_classes if selected_classes else 'All'})")
    plt.xlabel(f"PC1 ({var1:.2f}% explained variance)")
    plt.ylabel(f"PC2 ({var2:.2f}% explained variance)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), ncol=1)
    plt.tight_layout()

    if save:
        fname = f"mnist_pca_2d_{selected_classes if selected_classes else 'all'}_{seed}_fittoall.png"
        plt.savefig(fname, dpi=300)
        print(f"Plot saved as {fname}")
    else:
        plt.show()

    plt.close()


def process_single_digit(digit, num_samples=60000, save=True, seed: int = 42):
    """
    Process a single digit class using PCA fitted on the whole dataset.
    """
    process_mnist_pca(selected_classes=[digit], num_samples=num_samples, save=save, seed=seed)


def process_variance_plots(num_components=10, num_samples=60000, save=True, seed: int = 42):
    """
    Compute and plot the explained variance ratio degradation over the first num_components
    principal components for each digit and the full MNIST in separate subplots.

    Parameters
    ----------
    num_components : int
        Number of principal components to consider (default=10).
    num_samples : int
        Number of samples to use (default=60000, full MNIST).
    save : bool
        Whether to save the resulting plot as PNG.
    seed : int
        Random seed for reproducibility.
    """
    set_seed(seed)

    dataset_name = "MNIST"
    ds_info = DATASET_INFO[dataset_name]
    print(f"Processing variance plots for {dataset_name} with seed={seed}...")

    # Transform to tensor
    transform = transforms.ToTensor()

    # Load training dataset
    dataset = ds_info["class"](root="./data", train=True, download=True, transform=transform)

    # Limit samples (seeded)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    loader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1).numpy()
    labels = labels.numpy()

    # Fit PCA on full dataset
    print("Fitting PCA on full dataset...")
    reducer_full = PCA(n_components=num_components, random_state=seed)
    reducer_full.fit(images)
    var_ratios_full = reducer_full.explained_variance_ratio_

    # Fit PCA on each digit separately
    var_ratios_digits = []
    for i in range(ds_info["num_classes"]):
        print(f"Fitting PCA on digit {i}...")
        mask = labels == i
        images_digit = images[mask]
        if len(images_digit) == 0:
            print(f"No samples for digit {i}, skipping.")
            continue
        reducer = PCA(n_components=num_components, random_state=seed)
        reducer.fit(images_digit)
        var_ratios_digits.append((i, reducer.explained_variance_ratio_))

    # Plot: Create subplots for each digit and one for full MNIST
    num_subplots = ds_info["num_classes"] + 1  # Digits 0-9 + Full MNIST
    rows = (num_subplots + 2) // 3  # Ceiling division for 3 plots per row
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easier indexing

    components = np.arange(1, num_components + 1)

    # Plot for full MNIST
    axes[0].plot(components, var_ratios_full, marker='o', linewidth=2, color='black')
    axes[0].set_title('Full MNIST')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].grid(True)

    # Plot for each digit
    for idx, (digit, ratios) in enumerate(var_ratios_digits, start=1):
        axes[idx].plot(components, ratios, marker='o', alpha=0.7)
        axes[idx].set_title(f'Digit {digit}')
        axes[idx].set_xlabel('Principal Component')
        axes[idx].set_ylabel('Explained Variance Ratio')
        axes[idx].grid(True)

    # Turn off unused subplots
    for i in range(len(var_ratios_digits) + 1, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Explained Variance Ratio Degradation over Principal Components (MNIST)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to avoid title overlap

    if save:
        fname = f"mnist_variance_degradation_subplots_{num_components}pcs_{seed}.png"
        plt.savefig(fname, dpi=300)
        print(f"Variance plot saved as {fname}")
    else:
        plt.show()

    plt.close()


def main():
    # Example 1: Fit on all MNIST, show all classes
    # process_mnist_pca(selected_classes=None, seed=12)

    # Example 2: Process each digit one by one
    # for i in range(10):
    #     process_single_digit(i, seed=12)

    # Add variance degradation plot
    process_variance_plots(num_components=10, seed=12)


if __name__ == "__main__":
    main()