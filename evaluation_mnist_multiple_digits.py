import os
import argparse
import glob
import shutil
import tempfile
from typing import List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import idx2numpy
import torch_fidelity
from prdc import compute_prdc

# ------------------------- Utilities -------------------------

def load_mnist_digits(image_path: str, label_path: str, digits: List[int]|None) -> np.ndarray:
    """Return MNIST images of given digits (or all if None) as uint8 arrays [N,28,28]."""
    imgs = idx2numpy.convert_from_file(image_path)    # [N,28,28], uint8
    labels = idx2numpy.convert_from_file(label_path)  # [N], uint8
    if digits is not None:
        sel = np.isin(labels, digits)
        return imgs[sel]
    else:
        return imgs

def pad_to_32(img28: np.ndarray) -> np.ndarray:
    """Pad a 28×28 uint8 grayscale image to 32×32 with 2-px zero border."""
    return np.pad(img28, pad_width=((2,2),(2,2)), mode="constant", constant_values=0)

def load_generated_32x32_pngs(gen_dir: str) -> List[np.ndarray]:
    """Load generated grayscale PNGs (assumed 32×32). Converts to uint8 arrays [H,W]."""
    paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
    if not paths:
        raise ValueError(f"No PNG files in {gen_dir}")
    out = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("L")
            if im.size != (32, 32):
                im = im.resize((32, 32), Image.BILINEAR)
            out.append(np.array(im, dtype=np.uint8))
    return out

def to_3ch_tensor(img32: np.ndarray) -> torch.Tensor:
    """[32,32] uint8 -> [3,32,32] float tensor in [0,1] by channel repeat."""
    t = transforms.functional.to_tensor(img32)  # [1,32,32], float32 in [0,1]
    return t.repeat(3, 1, 1)

def save_tensor_batch_to_dir(img_batch: np.ndarray, prefix: str) -> str:
    """Save [N,32,32] uint8 grayscale to a temp dir as 3ch PNGs. Returns dir path."""
    tmp = tempfile.mkdtemp()
    to_pil = transforms.ToPILImage()
    for i, img_np in enumerate(img_batch):
        img_t = to_3ch_tensor(img_np)
        to_pil(img_t).save(os.path.join(tmp, f"{prefix}_{i:06d}.png"))
    return tmp

# ------------------------- PRDC Dataset -------------------------

class GrayscaleArrayDataset(Dataset):
    """Wrap a list/array of uint8 [H,W] images, yielding [1,32,32] float tensors in [0,1]."""
    def __init__(self, imgs_32x32: np.ndarray):
        self.imgs = imgs_32x32
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),   # [1,32,32], float32 in [0,1]
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return self.tf(img), 0

def collect_flat_features(dl: DataLoader) -> np.ndarray:
    """Collect grayscale features by flattening [1,32,32] -> [1024]."""
    feats = []
    for x, _ in dl:
        x_uint8 = (x * 255).to(torch.uint8)              # [B,1,32,32]
        x_np = x_uint8.squeeze(1).cpu().numpy()          # [B,32,32]
        feats.append(x_np.reshape(x_np.shape[0], -1))    # [B,1024]
    return np.concatenate(feats, axis=0)

# ------------------------- Main -------------------------

def main(
    image_path: str,
    label_path: str,
    gen_dir: str,
    digits: int = "1",
    batch_size: int = 64,
    nearest_k: int = 5,
):
    if digits.lower() == "all":
        digits_list = None
        digits_str = "all digits"
    else:
        digits_list = [int(d.strip()) for d in digits.split(",") if d.strip()]
        if not digits_list:
            raise ValueError("Invalid digits specified")
        digits_str = ", ".join(map(str, digits_list))
    
    # Real: select digit and pad to 32×32
    real_28 = load_mnist_digits(image_path, label_path, digits_list)  # [N,28,28]
    real_32 = np.stack([pad_to_32(im) for im in real_28], axis=0)  # [N,32,32]

    # Fake: load generated (resize to 32×32 if needed)
    gen_32 = load_generated_32x32_pngs(gen_dir)  # list of [32,32]
    gen_32 = np.stack(gen_32, axis=0)            # [M,32,32]

    # --- FID/IS preprocessing: to 3-channel tensors, saved to temp dirs ---
    # real_3ch = torch.stack([to_3ch_tensor(im) for im in real_32], dim=0)  # [N,3,32,32]
    # gen_3ch  = torch.stack([to_3ch_tensor(im) for im in gen_32],  dim=0)  # [M,3,32,32]

    real_dir = save_tensor_batch_to_dir(real_32, prefix="real")
    gen_dir_ = save_tensor_batch_to_dir(gen_32,  prefix="gen")

    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir_,
            input2=real_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            isc=True,
            verbose=False,
        )
        fid = float(metrics["frechet_inception_distance"])
        is_mean = float(metrics["inception_score_mean"])
        is_std  = float(metrics["inception_score_std"])
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(gen_dir_, ignore_errors=True)

    # --- PRDC on raw grayscale (flattened 32×32) ---
    ds_real = GrayscaleArrayDataset(real_32)
    ds_gen  = GrayscaleArrayDataset(gen_32)
    dl_real = DataLoader(ds_real, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_gen  = DataLoader(ds_gen,  batch_size=batch_size, shuffle=False, num_workers=0)

    num_feat = 10000
    real_feat = collect_flat_features(dl_real)[:num_feat]  # [N,1024]
    gen_feat  = collect_flat_features(dl_gen)[:num_feat]   # [M,1024]
    prdc = compute_prdc(real_features=real_feat, fake_features=gen_feat, nearest_k=nearest_k)

    # --- Output ---
    print("\n==== Metrics (Digits = {}) ====".format(digits_str))
    print(f"FID: {fid:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall: {prdc['recall']:.4f}")
    print(f"Density: {prdc['density']:.4f}")
    print(f"Coverage: {prdc['coverage']:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("FID/IS/PRDC for a specific MNIST digits (or all) vs generated 32×32 PNGs")
    p.add_argument("--image_path", required=True, help="MNIST train-images-idx3-ubyte")
    p.add_argument("--label_path", required=True, help="MNIST train-labels-idx1-ubyte")
    p.add_argument("--gen_dir",    required=True, help="Directory of generated PNGs (32×32 or resizable)")
    p.add_argument("--digits",      type=str, default="1", help="Comma-separated MNIST digits (0-9) or 'all' (e.g., '0,1')")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--nearest_k",  type=int, default=5)
    args = p.parse_args()
    main(args.image_path, args.label_path, args.gen_dir, args.digit, args.batch_size, args.nearest_k)
