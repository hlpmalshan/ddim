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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")


# ---- CelebA-style crop used in your training code ----
class Crop:
    # Training code passed (x1, x2, y1, y2); PIL crop uses (left=x1, upper=y1, right=x2, lower=y2).
    def __init__(self, x1, x2, y1, y2):
        self.box = (x1, y1, x2, y2)
    def __call__(self, img: Image.Image):
        return img.crop(self.box)


# ---- Utils ----
def list_images(root):
    return sorted(p for p in glob.glob(os.path.join(root, "*"))
                  if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"])

def select_balanced(real_dir, gen_dir, num_images=None):
    real_files = list_images(real_dir)
    gen_files  = list_images(gen_dir)
    if not real_files or not gen_files:
        raise RuntimeError("No valid images in real_dir or gen_dir.")
    cap = min(len(real_files), len(gen_files))
    if num_images is not None:
        cap = min(cap, max(0, int(num_images)))
        if cap < num_images:
            print(f"[Info] Requested {num_images}, using {cap}.")
    return real_files[:cap], gen_files[:cap], cap


def save_real_subset_to_tmp(real_files, train_pil_tf, fid_size):
    """Apply training Crop+Resize (and optional flip) to REAL images, then upsize to fid_size for torch-fidelity."""
    out = tempfile.mkdtemp()
    post = transforms.Compose([transforms.Resize(fid_size), transforms.CenterCrop(fid_size)])
    for i, f in enumerate(tqdm(real_files, desc="Saving REAL (train transform → FID size)", leave=False)):
        try:
            img = Image.open(f).convert("RGB")
            img = train_pil_tf(img)   # Crop+Resize(+Flip if enabled)
            img = post(img)           # → fid_size
            img.save(os.path.join(out, f"real_{i:06d}.png"))
        except Exception:
            pass
    return out


def save_gen_subset_to_tmp(gen_files, fid_size):
    """No special crop for GEN; just resize/center-crop to fid_size for torch-fidelity."""
    out = tempfile.mkdtemp()
    post = transforms.Compose([transforms.Resize(fid_size), transforms.CenterCrop(fid_size)])
    for f in tqdm(gen_files, desc="Saving GEN (→ FID size)", leave=False):
        try:
            img = Image.open(f).convert("RGB")
            img = post(img)
            img.save(os.path.join(out, os.path.basename(f)))
        except Exception:
            pass
    return out


class FilesTensorDataset(Dataset):
    """Generic image-files → tensor dataset with a given transform."""
    def __init__(self, files, tf):
        self.files = files
        self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.tf(img), 0


def extract_prdc_features(dl):
    """Return float32 features in [-1, 1], flattened per image."""
    feats = []
    for x, _ in tqdm(dl, desc="Collecting PRDC tensors", leave=False):
        # x already in [-1,1]; move to CPU float32
        feats.append(x.cpu().to(torch.float32).view(x.size(0), -1))
    return torch.cat(feats, dim=0).numpy()


def main(args):
    # ---- Training crop parameters (from your snippet) ----
    cx, cy = 89, 121
    x1, x2 = cy - 64, cy + 64
    y1, y2 = cx - 64, cx + 64

    # Training transform for REAL (exactly like training: Crop→Resize, optional flip)
    common = [Crop(x1, x2, y1, y2), transforms.Resize(args.image_size)]
    real_train_pil_tf = transforms.Compose(common)  # for FID/IS temp copies
    # For PRDC: ToTensor->[0,1] then scale to [-1,1]
    real_train_tensor_tf = transforms.Compose(
        common + [transforms.ToTensor()]
    )

    # GEN transform for PRDC: ensure SAME size as real; then scale to [-1,1]
    gen_tensor_tf = transforms.Compose([
        # transforms.Resize(args.image_size),
        # transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])

    # ---- Balance subsets ----
    real_files, gen_files, cap = select_balanced(args.real_dir, args.gen_dir, args.num_images)
    print(f"[Subset] Using {cap} samples per set.")

    # ---- FID/IS on SAME subset ----
    real_tmp = save_real_subset_to_tmp(real_files, real_train_pil_tf, args.fid_size)
    gen_tmp  = save_gen_subset_to_tmp(gen_files, args.fid_size)

    metrics = torch_fidelity.calculate_metrics(
        input1=gen_tmp,
        input2=real_tmp,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        verbose=False
    )
    fid_score = metrics["frechet_inception_distance"]
    is_mean   = metrics["inception_score_mean"]
    is_std    = metrics["inception_score_std"]

    shutil.rmtree(real_tmp, ignore_errors=True)
    shutil.rmtree(gen_tmp,  ignore_errors=True)

    # ---- PRDC on SAME subset; tensors in [-1,1] ----
    dl_real = DataLoader(FilesTensorDataset(real_files, real_train_tensor_tf),
                         batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    dl_gen  = DataLoader(FilesTensorDataset(gen_files,  gen_tensor_tf),
                         batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    real_feats = extract_prdc_features(dl_real)  # shape: [cap, C*H*W], float32 in [-1,1]
    gen_feats  = extract_prdc_features(dl_gen)

    # Safety: nearest_k must be < number of samples
    k = min(args.nearest_k, max(1, cap - 1))
    prdc = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=k)

    # ---- Report ----
    print("\n==== Final Metrics (REAL dir with training transform; PRDC on [-1,1]) ====")
    print(f"Samples used per set: {cap}")
    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall:    {prdc['recall']:.4f}")
    print(f"Density:   {prdc['density']:.4f}")
    print(f"Coverage:  {prdc['coverage']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate FID/IS/PRDC using REAL images from a directory with CelebA training transform.")
    ap.add_argument("--real_dir", type=str, required=True, help="Directory of REAL images (files).")
    ap.add_argument("--gen_dir",  type=str, required=True, help="Directory of generated images.")
    ap.add_argument("--num_images", type=int, default=50000, help="Max images per set (balanced).")
    ap.add_argument("--image_size", type=int, default=64, help="Training Resize size after Crop for REAL; PRDC target size.")
    ap.add_argument("--fid_size", type=int, default=64, help="Resize+CenterCrop size for FID/IS temp copies.")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for PRDC.")
    ap.add_argument("--nearest_k", type=int, default=5, help="k for PRDC.")
    ap.add_argument("--use_flip", action="store_true", help="Apply RandomHorizontalFlip to REAL like training.")
    args = ap.parse_args()
    main(args)
