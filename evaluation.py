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
from torchmetrics.image.fid import FrechetInceptionDistance
from prdc import compute_prdc

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

# -------- Clean images: remove non-images, corrupted files, convert to RGB --------
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
            img = img.convert("RGB")  # enforce RGB
            img.verify()  # check corruption
        except Exception:
            print(f"Removing corrupted file: {f}")
            os.remove(f)

# -------- Prepare resized copy with torchvision transforms (resize shorter side + center crop) --------
def prepare_resized_copy(src_dir, size=64):
    tmp_dir = tempfile.mkdtemp()
    transform = transforms.Compose([
        transforms.Resize(size),          # shorter side resized to 299
        transforms.CenterCrop(size),      # center crop 299x299
    ])
    for f in tqdm(glob.glob(os.path.join(src_dir, "*")), desc=f"Processing images in {src_dir}"):
        try:
            img = Image.open(f).convert("RGB")
            img = transform(img)
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
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

# -------- Extract raw RGB images as numpy arrays for PRDC --------
def extract_rgb_features(dataloader):
    imgs_list = []
    for imgs, _ in tqdm(dataloader, desc="Collecting RGB images for PRDC"):
        # imgs: float tensor [B,C,H,W] in [0,1]
        imgs_uint8 = (imgs * 255).to(torch.uint8)
        imgs_np = imgs_uint8.permute(0, 2, 3, 1).cpu().numpy()  # B,H,W,C uint8
        imgs_list.append(imgs_np)
    return np.concatenate(imgs_list, axis=0)

# -------- Main --------
def main(real_dir, gen_dir, batch_size=32, device='cuda:1' if torch.cuda.is_available() else 'cpu', image_size=64):
    print("[Cleaning image directories]")
    clean_image_dir(real_dir)
    clean_image_dir(gen_dir)

    # Ensure equal number of images in both dirs
    real_files = sorted(glob.glob(os.path.join(real_dir, "*")))
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*")))
    if len(gen_files) > len(real_files):
        gen_files = gen_files[:len(real_files)]
        # optionally remove the extras from disk so following code sees equal sets
        for f in sorted(glob.glob(os.path.join(gen_dir, "*")))[len(real_files):]:
            os.remove(f)
    
    if len(real_files) > len(gen_files):
        real_files = real_files[:len(gen_files)]
    
    print("[Preparing resized copies for FID/IS]")
    real_tmp = prepare_resized_copy(real_dir, size=image_size)
    gen_tmp = prepare_resized_copy(gen_dir, size=image_size)
    
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

    print("[Calculating PRDC metrics with raw RGB images]")
    prdc_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),  # float tensor in [0,1]
    ])
    ds_real = ImageDataset(real_dir, prdc_transform)
    ds_gen = ImageDataset(gen_dir, prdc_transform)
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
    parser.add_argument("--real_dir", type=str, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, help="Directory with generated images")
    parser.add_argument("--image_size", type=int, default=64, help="Image resolution (eg. 32 or 64)")
    args = parser.parse_args()
    main(args.real_dir, args.gen_dir, args.image_size)