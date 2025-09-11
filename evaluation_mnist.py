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

# -------- Clean images: remove non-images, corrupted files --------
def clean_image_dir(img_dir):
    valid_exts = [".jpg", ".jpeg", ".png"]
    files = glob.glob(os.path.join(img_dir, "*"))
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in valid_exts:
            print(f"Removing non-image file: {f}")
            os.remove(f)
            continue
        try:
            img = Image.open(f)
            img.verify()  # check corruption
            if img.mode != "L":
                print(f"Converting {f} to grayscale")
                img = img.convert("L")  # enforce grayscale
                img.save(f)
        except Exception:
            print(f"Removing corrupted file: {f}")
            os.remove(f)

# -------- Convert MNIST IDX to grayscale images in a temporary directory --------
def convert_mnist_to_grayscale_images(image_path, label_path, digit=1):
    tmp_dir = tempfile.mkdtemp()
    images = idx2numpy.convert_from_file(image_path)  # Shape: [N, 28, 28]
    labels = idx2numpy.convert_from_file(label_path)  # Shape: [N]
    
    # Filter for digit 1
    digit_indices = np.where(labels == digit)[0]
    print(f"Found {len(digit_indices)} images of digit {digit}")
    
    for idx in tqdm(digit_indices, desc=f"Saving MNIST digit {digit} images"):
        img_array = images[idx]  # Shape: [28, 28], uint8
        img = Image.fromarray(img_array, mode="L")  # Grayscale mode
        img.save(os.path.join(tmp_dir, f"mnist_digit1_{idx}.png"))
    
    return tmp_dir

# -------- Prepare resized copy with torchvision transforms --------
def prepare_resized_copy(src_dir, size=32, grayscale=True):
    tmp_dir = tempfile.mkdtemp()
    transform = transforms.Compose([
        transforms.Resize(size),          # shorter side resized to size
        transforms.CenterCrop(size),      # center crop size x size
    ])
    for f in tqdm(glob.glob(os.path.join(src_dir, "*")), desc=f"Processing images in {src_dir}"):
        try:
            img = Image.open(f).convert("L")  # ensure grayscale
            img = transform(img)
            # Convert to pseudo-RGB for torch_fidelity (Inception V3 expects 3 channels)
            if not grayscale:
                img = img.convert("RGB")
            img.save(os.path.join(tmp_dir, os.path.basename(f)))
        except Exception:
            print(f"Skipping invalid file: {f}")
    return tmp_dir

# -------- Dataset for PRDC --------
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root, "*")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")  # ensure grayscale
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
def main(image_path, label_path, gen_dir, batch_size=32, device='cuda:1' if torch.cuda.is_available() else 'cpu'):
    print("[Converting MNIST digit 1 to grayscale images]")
    real_dir = convert_mnist_to_grayscale_images(image_path, label_path, digit=1)
    
    print("[Cleaning generated image directory]")
    clean_image_dir(gen_dir)

    print("[Preparing resized copies for FID/IS]")
    # Convert to pseudo-RGB for torch_fidelity
    real_tmp = prepare_resized_copy(real_dir, grayscale=False)
    gen_tmp = prepare_resized_copy(gen_dir, grayscale=False)
    
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
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),  # float tensor [1,H,W] in [0,1]
    ])
    ds_real = ImageDataset(real_dir, prdc_transform)
    ds_gen = ImageDataset(gen_dir, prdc_transform)
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

    shutil.rmtree(real_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID, IS, PRDC metrics for MNIST digit 1 vs generated grayscale images")
    parser.add_argument("--image_path", type=str, help="Path to MNIST train-images-idx3-ubyte file")
    parser.add_argument("--label_path", type=str, help="Path to MNIST train-labels-idx1-ubyte file")
    parser.add_argument("--gen_dir", type=str, help="Directory with generated grayscale images")
    args = parser.parse_args()
    main(args.image_path, args.label_path, args.gen_dir)