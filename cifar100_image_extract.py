import os
import pickle
import numpy as np
from PIL import Image

# Path to the extracted CIFAR-100 folder
cifar100_dir = "ddim_cifar100/datasets/cifar100/cifar-100-python"

# Load data function
def load_cifar100(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

# Output folders (avoid conflict with pickle files)
train_out = "ddim_cifar100/datasets/cifar100/train"
test_out = "ddim_cifar100/datasets/cifar100/test"
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

# Function to save images
def save_images(data_dict, split_dir):
    data = data_dict["data"]
    for i, img_array in enumerate(data):
        img = img_array.reshape(3, 32, 32).transpose(1, 2, 0)  # CHW → HWC
        img_path = os.path.join(split_dir, f"{i}.png")
        Image.fromarray(img).save(img_path)

# Process train and test
train_dict = load_cifar100(os.path.join(cifar100_dir, "train"))
test_dict = load_cifar100(os.path.join(cifar100_dir, "test"))

print("Saving training images...")
save_images(train_dict, train_out)

print("Saving testing images...")
save_images(test_dict, test_out)

print("Done! Images are saved into 'train/' and 'test/' folders.")
