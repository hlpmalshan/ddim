#!/usr/bin/env python3
# celebahq_eval.py
import os, glob, shutil, tempfile, argparse, random
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch_fidelity
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from prdc import compute_prdc

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(d):
    return sorted(
        p for p in glob.glob(os.path.join(d, "*"))
        if os.path.splitext(p)[1].lower() in IMG_EXTS
    )

def ensure_size_tmp(files, size):
    """Write images to a temp dir at exactly size×size (resize+centercrop)."""
    tmp = tempfile.mkdtemp(prefix=f"imgs_{size}_")
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])
    written = 0
    for f in tqdm(files, desc=f"Prep {size}×{size}", leave=False):
        try:
            img = Image.open(f).convert("RGB")
            if img.size != (size, size):
                img = tf(img)
            img.save(os.path.join(tmp, os.path.basename(f)))
            written += 1
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
    if written == 0:
        raise RuntimeError("No images written (check inputs).")
    return tmp

class ImageTensorDS(Dataset):
    def __init__(self, files, size):
        self.files = files
        self.tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img), 0

@torch.no_grad()
def tensor_stack(dl):
    xs = []
    for x, _ in tqdm(dl, desc="PRDC tensors", leave=False):
        xs.append(x.to(torch.float32).view(x.size(0), -1).cpu())
    return torch.cat(xs, 0).numpy()

def main(args):
    # --- Gather ---
    real_files = list_images(args.real_dir)
    gen_files  = list_images(args.gen_dir)
    if not real_files or not gen_files:
        raise RuntimeError("Empty real/gen directories or no valid images.")

    # --- Subset with reproducible shuffle ---
    n = min(len(real_files), len(gen_files), args.max_images or 10**9)
    rng = random.Random(args.seed)
    rng.shuffle(real_files); rng.shuffle(gen_files)
    real_files, gen_files = real_files[:n], gen_files[:n]
    print(f"[Subset] Using {n} images per set at target {args.resolution}×{args.resolution}.")

    # --- FID (torch-fidelity) ---
    real_tmp = ensure_size_tmp(real_files, args.resolution)
    gen_tmp  = ensure_size_tmp(gen_files,  args.resolution)
    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_tmp, input2=real_tmp,
            cuda=torch.cuda.is_available() and (not args.cpu),
            fid=True, verbose=False, samples_find_deep=False
        )
        fid_score = metrics["frechet_inception_distance"]
        print("\n==== CelebA-HQ EVAL ====")
        print(f"Resolution: {args.resolution}×{args.resolution}")
        print(f"Images per set: {n}")
        print(f"FID: {fid_score:.4f}")
    finally:
        shutil.rmtree(real_tmp, ignore_errors=True)
        shutil.rmtree(gen_tmp,  ignore_errors=True)

    # --- PRDC (pixel-space vectors, same resize/crop pipeline) ---
    dl_kw = dict(batch_size=args.batch_size, shuffle=False,
                 num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0))
    dl_real = DataLoader(ImageTensorDS(real_files, args.resolution), **dl_kw)
    dl_gen  = DataLoader(ImageTensorDS(gen_files,  args.resolution), **dl_kw)

    real_feats = tensor_stack(dl_real)
    gen_feats  = tensor_stack(dl_gen)

    k = min(args.nearest_k, max(1, n - 1))
    prdc = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=k)
    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall:    {prdc['recall']:.4f}")
    print(f"Density:   {prdc['density']:.4f}")
    print(f"Coverage:  {prdc['coverage']:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("CelebA-HQ eval @ specified resolution (default 256): FID + PRDC")
    p.add_argument("--real_dir", required=True, type=str, help="Path to CelebA-HQ real images (256×256).")
    p.add_argument("--gen_dir",  required=True, type=str, help="Path to generated images.")
    p.add_argument("--resolution", type=int, default=256, help="Eval resolution (default 256).")
    p.add_argument("--max_images", type=int, default=27000, help="Max images per set.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nearest_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cpu", action="store_true", help="Force CPU for FID.")
    args = p.parse_args()
    main(args)
