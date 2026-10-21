import os
import argparse
import glob
import tempfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch_fidelity
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from prdc import compute_prdc

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

# -------- Validate inputs without modifying source images --------
def clean_image_dir(img_dir):
    if not os.path.isdir(img_dir):
        raise ValueError(
            f"Image directory does not exist: {img_dir}. "
            "For CIFAR-10, run python cifar10_batch_to_image.py to export the batch files first."
        )
    files = []
    for path in sorted(glob.glob(os.path.join(img_dir, "*"))):
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            files.append(path)
        except (OSError, ValueError):
            print(f"Skipping corrupted image: {path}")
    if not files:
        raise ValueError(f"No valid PNG/JPEG images found directly in {img_dir}")
    return files


# -------- Prepare resized copies of the selected images --------
def prepare_resized_copy(files, dst_dir, size=32):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])
    for index, path in enumerate(tqdm(files, desc="Preparing images")):
        with Image.open(path) as img:
            transform(img.convert("RGB")).save(os.path.join(dst_dir, f"{index}.png"))

# -------- Dataset for PRDC --------
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = list(root)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

# -------- Extract raw RGB images as numpy arrays for PRDC --------
def extract_rgb_features(dataloader):
    imgs_list = []
    for imgs, _ in tqdm(dataloader, desc="Collecting RGB images for PRDC"):
        # PILToTensor supplies uint8 pixels [B,C,H,W] in [0,255].
        imgs_uint8 = imgs.to(torch.uint8)
        imgs_np = imgs_uint8.permute(0, 2, 3, 1).cpu().numpy()  # B,H,W,C uint8
        imgs_list.append(imgs_np)
    return np.concatenate(imgs_list, axis=0)

# -------- Main --------
def main(real_dir, gen_dir, batch_size=32, device='cuda:1' if torch.cuda.is_available() else 'cpu', image_size=32):
    print("[Validating image directories]")
    real_files = clean_image_dir(real_dir)
    gen_files = clean_image_dir(gen_dir)
    count = min(len(real_files), len(gen_files))
    if count <= 5:
        raise ValueError(f"PRDC with nearest_k=5 requires at least 6 images per set; got {count}")
    real_files = real_files[:count]
    gen_files = gen_files[:count]
    print(f"Images per set: {count}")

    with tempfile.TemporaryDirectory() as real_tmp, tempfile.TemporaryDirectory() as gen_tmp:
        print("[Preparing resized copies for FID/IS]")
        prepare_resized_copy(real_files, real_tmp, size=image_size)
        prepare_resized_copy(gen_files, gen_tmp, size=image_size)
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
    
    print("\n==== Final Metrics ====")
    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")


    print("[Calculating PRDC metrics with raw RGB images]")
    prdc_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),  # uint8 tensor in [0,255]
    ])
    ds_real = ImageDataset(real_files, prdc_transform)
    ds_gen = ImageDataset(gen_files, prdc_transform)
    dl_real = DataLoader(ds_real, batch_size=batch_size, shuffle=False)
    dl_gen = DataLoader(ds_gen, batch_size=batch_size, shuffle=False)

    real_feats = extract_rgb_features(dl_real)
    gen_feats = extract_rgb_features(dl_gen)

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
    parser = argparse.ArgumentParser(description="Calculate FID, IS, PRDC metrics on image directories")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images")
    parser.add_argument("--image_size", type=int, default=32, help="Image resolution (eg. 32 or 64)")
    args = parser.parse_args()
    main(args.real_dir, args.gen_dir, image_size=args.image_size)
