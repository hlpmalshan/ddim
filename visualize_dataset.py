#!/usr/bin/env python3
"""
MNIST UMAP Visualization Script

This script loads the MNIST dataset, fits UMAP on the full dataset (or all selected classes),
and visualizes chosen subsets of classes without retraining UMAP.
Results are fully reproducible due to fixed random seeds.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import umap  # Requires umap-learn
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
    """Ensure reproducibility across NumPy, Torch, and UMAP."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_mnist_umap(selected_classes=None, num_samples=10000, save=True, seed: int = 42):
    """
    Process MNIST dataset, fit UMAP on the whole dataset (or all selected classes),
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

    # Fit UMAP on ALL available samples
    print("Fitting UMAP on full dataset...")
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
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

    plt.title(f"UMAP 2D Projection of MNIST (Classes: {selected_classes if selected_classes else 'All'})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), ncol=1)
    plt.tight_layout()

    if save:
        fname = f"mnist_umap_2d_{selected_classes if selected_classes else 'all'}_{seed}.png"
        plt.savefig(fname, dpi=300)
        print(f"Plot saved as {fname}")
    else:
        plt.show()

    plt.close()


def main():
    # Example 1: Fit on all MNIST, show all classes
    # process_mnist_umap(selected_classes=None, seed=12)

    # Example 2: Fit on all MNIST, show only digits [0, 1, 2]
    process_mnist_umap(selected_classes=[1], seed=12)


if __name__ == "__main__":
    main()
