import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images_from_batch(batch_file, output_dir, prefix):
    batch = unpickle(batch_file)
    data = batch[b'data']
    labels = batch[b'labels']
    filenames = batch[b'filenames']

    data = data.reshape(-1, 3, 32, 32)  # (num_images, channels, height, width)
    data = data.transpose(0, 2, 3, 1)   # (num_images, height, width, channels)

    for i, img_array in enumerate(data):
        label = labels[i]
        orig_name = filenames[i].decode('utf-8')
        # prevent collisions by prefixing batch name + index
        filename = f"{prefix}_{i}_{label}_{orig_name}"
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, filename))

# Create directories
os.makedirs("ddim_cifar10/datasets/cifar10/train", exist_ok=True)
os.makedirs("ddim_cifar10/datasets/cifar10/test", exist_ok=True)

# Save training batches
for i in range(1, 6):
    save_images_from_batch(f"ddim_cifar10/datasets/cifar10/cifar-10-batches-py/data_batch_{i}", "ddim_cifar10/datasets/cifar10/train", f"train{i}")

# Save test batch
save_images_from_batch("ddim_cifar10/datasets/cifar10_test/cifar-10-batches-py/test_batch", "ddim_cifar10/datasets/cifar10/test", "test")
