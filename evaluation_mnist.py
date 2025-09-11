import os
import argparse
import glob
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch_fidelity
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from prdc import compute_prdc
import idx2numpy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

# -------- Get MNIST digit 1 images as numpy arrays --------
def get_mnist_digit_arrays(image_path, label_path, digit=1):
    images = idx2numpy.convert_from_file(image_path)  # Shape: [N, 28, 28]
    labels = idx2numpy.convert_from_file(label_path)  # Shape: [N]
    
    # Filter for digit 1
    digit_indices = np.where(labels == digit)[0]
    print(f"Found {len(digit_indices)} images of digit {digit}")
    
    return images[digit_indices]  # Shape: [N', 28, 28], uint8

# -------- Prepare resized images for FID/IS in memory --------
def prepare_resized_arrays(image_arrays, size=32, grayscale=True):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),          # shorter side resized to size
        transforms.CenterCrop(size),      # center crop size x size
        transforms.ToTensor(),           # [C, H, W], float in [0,1]
    ])
    resized_images = []
    for img_array in tqdm(image_arrays, desc="Processing images"):
        img_tensor = transform(img_array)  # [1, H, W]
        if not grayscale:
            img_tensor = img_tensor.repeat(3, 1, 1)  # Convert [1, H, W] to [3, H, W]
        resized_images.append(img_tensor)
    return torch.stack(resized_images)  # Shape: [N, C, H, W]

# -------- Save tensor batch to temporary directory for torch_fidelity --------
def save_tensors_to_temp_dir(tensor_batch, prefix="img"):
    tmp_dir = tempfile.mkdtemp()
    for i, img_tensor in enumerate(tqdm(tensor_batch, desc="Saving images to temp dir")):
        img = transforms.ToPILImage()(img_tensor)
        img.save(os.path.join(tmp_dir, f"{prefix}_{i}.png"))
    return tmp_dir

# -------- Dataset for PRDC --------
class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # Either file paths, numpy array, or list of numpy arrays
        self.transform = transform
        # Check if data is a numpy array or a list of numpy arrays
        self.is_array = isinstance(data, np.ndarray) or (isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_array:
            img = self.data[idx]  # numpy array
            img = Image.fromarray(img)  # Convert to PIL, mode inferred (L for uint8 grayscale)
        else:
            img = Image.open(self.data[idx]).convert("L")  # From file, ensure grayscale
        if self.transform:
            img = self.transform(img)
        return img, 0

# -------- Extract raw grayscale images as numpy arrays for PRDC --------
def extract_grayscale_features(dataloader):
    imgs_list = []
    for imgs, _ in tqdm(dataloader, desc="Collecting grayscale images for PRDC"):
        # imgs: float tensor [B,1,H,W] in [0,1]
        imgs_uint8 = (imgs * 255).to(torch.uint8)
        imgs_np = imgs_uint8.squeeze(1).cpu().numpy()  # B,H,W (grayscale)
        imgs_list.append(imgs_np)
    return np.concatenate(imgs_list, axis=0)

# -------- Main --------
def main(image_path, label_path, gen_dir, batch_size=32, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    print("[Loading MNIST digit 1 images]")
    real_images = get_mnist_digit_arrays(image_path, label_path, digit=1)  # Shape: [N, 28, 28]
    
    print("[Loading generated images from gen_dir]")
    gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))  # Only PNG files
    if not gen_paths:
        raise ValueError(f"No valid PNG images found in {gen_dir}")
    gen_images = []
    for f in tqdm(gen_paths, desc="Loading generated images"):
        try:
            img = Image.open(f).convert("L")  # Ensure grayscale
            gen_images.append(np.array(img))  # Convert to numpy array
        except Exception as e:
            print(f"Warning: Skipping problematic file {f}: {e}")

    print("[Preparing resized images for FID/IS]")
    # Prepare MNIST images (in memory)
    real_tensors = prepare_resized_arrays(real_images, grayscale=False)  # [N, 3, 32, 32]
    # Prepare generated images
    gen_tensors = prepare_resized_arrays(gen_images, grayscale=False)  # [N, 3, 32, 32]
    
    print("[Saving resized images to temporary directories for FID/IS]")
    real_tmp = save_tensors_to_temp_dir(real_tensors, prefix="mnist")
    gen_tmp = save_tensors_to_temp_dir(gen_tensors, prefix="gen")
    
    print("[Calculating FID and IS with torch-fidelity]")
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_tmp,
        input2=real_tmp,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        verbose=False
    )
    fid_score = metrics['frechet_inception_distance']
    is_mean = metrics['inception_score_mean']
    is_std = metrics['inception_score_std']

    shutil.rmtree(real_tmp)
    shutil.rmtree(gen_tmp)

    print("[Calculating PRDC metrics with raw grayscale images]")
    prdc_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),  # float tensor [1,H,W] in [0,1]
    ])
    ds_real = ImageDataset(real_images, prdc_transform)  # From numpy array
    ds_gen = ImageDataset(gen_images, prdc_transform)    # From list of arrays
    dl_real = DataLoader(ds_real, batch_size=batch_size, shuffle=False)
    dl_gen = DataLoader(ds_gen, batch_size=batch_size, shuffle=False)

    real_feats = extract_grayscale_features(dl_real)
    gen_feats = extract_grayscale_features(dl_gen)

    real_feats = real_feats.reshape(real_feats.shape[0], -1)
    gen_feats = gen_feats.reshape(gen_feats.shape[0], -1)
    prdc_metrics = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=5)

    print("\n==== Final Metrics ====")
    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Precision: {prdc_metrics['precision']:.4f}")
    print(f"Recall: {prdc_metrics['recall']:.4f}")
    print(f"Density: {prdc_metrics['density']:.4f}")
    print(f"Coverage: {prdc_metrics['coverage']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID, IS, PRDC metrics for MNIST digit 1 vs generated grayscale images")
    parser.add_argument("--image_path", type=str, help="Path to MNIST train-images-idx3-ubyte file")
    parser.add_argument("--label_path", type=str, help="Path to MNIST train-labels-idx1-ubyte file")
    parser.add_argument("--gen_dir", type=str, help="Directory with generated grayscale PNG images")
    args = parser.parse_args()
    main(args.image_path, args.label_path, args.gen_dir)