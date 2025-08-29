import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import umap
from torch.utils.data import Subset, DataLoader

# Define dataset configurations
DATASETS = {
    'MNIST': {'class': torchvision.datasets.MNIST, 'num_classes': 10, 'image_size': (28, 28, 1), 'classes': [str(i) for i in range(10)]},
    'OxfordFlowers': {'class': torchvision.datasets.Flowers102, 'num_classes': 102, 'image_size': (None), 'classes': None},
    'OxfordPets': {'class': torchvision.datasets.OxfordIIITPet, 'num_classes': 37, 'image_size': (None), 'classes': None},
    'CIFAR10': {'class': torchvision.datasets.CIFAR10, 'num_classes': 10, 'image_size': (32, 32, 3), 'classes': None},
    'CIFAR100': {'class': torchvision.datasets.CIFAR100, 'num_classes': 100, 'image_size': (32, 32, 3), 'classes': None},
}

def process_dataset(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(DATASETS.keys())}")
    
    ds_info = DATASETS[dataset_name]
    print(f"Processing {dataset_name}...")
    
    # Transform to tensor
    transform = transforms.ToTensor()
    
    # Load the training dataset (downloads if not present)
    dataset = ds_info['class'](root='./data', train=True, download=True, transform=transform)
    
    # Get classes if available
    classes = ds_info['classes'] if ds_info['classes'] else (getattr(dataset, 'classes', None) or [str(i) for i in range(ds_info['num_classes'])])
    
    # Subsample for efficiency (e.g., 2000 samples; adjust as needed)
    num_samples = min(2000, len(dataset))
    np.random.seed(42)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    
    # DataLoader to get batches
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
    for images, labels in loader:
        # Flatten images to vectors
        if dataset_name == 'MNIST':
            images = images.view(images.size(0), -1).numpy()  # Grayscale
        else:
            images = images.view(images.size(0), -1).numpy()  # Color or variable size handled by ToTensor
        labels = labels.numpy()
        break
    
    # Apply UMAP for 2D reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(images)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    num_unique = len(unique_labels)
    colormap = plt.cm.get_cmap('hsv', num_unique)  # HSV for many colors
    
    for i, label_idx in enumerate(unique_labels):
        mask = labels == label_idx
        label_name = classes[label_idx] if label_idx < len(classes) else str(label_idx)
        plt.scatter(embedding[mask, 0], embedding[mask, 1], color=colormap(i), label=label_name, s=5, alpha=0.7)
    
    plt.title(f'UMAP 2D Projection of {dataset_name}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=2 if num_unique > 20 else 1)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_umap_2d.png')  # Save to file
    plt.show()
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize image datasets in 2D using UMAP.')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()),
                        help=f'Dataset to process. Choose from {list(DATASETS.keys())}')
    args = parser.parse_args()
    
    # Process the selected dataset
    process_dataset(args.dataset)
    print(f"Dataset {args.dataset} processed and plot saved as {args.dataset}_umap_2d.png")

if __name__ == '__main__':
    main()