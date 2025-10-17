#!/usr/bin/env python3
import os, glob, shutil, tempfile, argparse
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
    return sorted(p for p in glob.glob(os.path.join(d, "*"))
                  if os.path.splitext(p)[1].lower() in IMG_EXTS)

def prepare_tmp64(files, out_size=64):
    """Ensure 64×64 on disk for torch-fidelity (both REAL and GEN)."""
    tmp = tempfile.mkdtemp(prefix="imgs_64_")
    post = transforms.Compose([transforms.Resize(out_size), transforms.CenterCrop(out_size)])
    written = 0
    for f in tqdm(files, desc="Ensure 64×64", leave=False):
        try:
            img = Image.open(f).convert("RGB")
            if img.size != (out_size, out_size):
                img = post(img)
            img.save(os.path.join(tmp, os.path.basename(f)))
            written += 1
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
    if written == 0:
        raise RuntimeError("No images written to tmp (check inputs).")
    return tmp

class ImageTensorDS(Dataset):
    def __init__(self, files, tf):
        self.files = files; self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img), 0

def extract_prdc(dl):
    xs = []
    for x, _ in tqdm(dl, desc="PRDC tensors", leave=False):
        xs.append(x.to(torch.float32).view(x.size(0), -1).cpu())
    return torch.cat(xs, 0).numpy()

def main(args):
    real_files = list_images(args.real_dir)
    gen_files  = list_images(args.gen_dir)
    if not real_files or not gen_files:
        raise RuntimeError("Empty real/gen directories or no valid images found.")

    n = min(len(real_files), len(gen_files), args.max_images or 10**9)
    real_files, gen_files = real_files[:n], gen_files[:n]
    print(f"[Subset] Using {n} images per set (preprocessed real, 64×64).")

    # ---- FID (both ensured 64×64) ----
    real_tmp = prepare_tmp64(real_files, out_size=64)
    gen_tmp  = prepare_tmp64(gen_files,  out_size=64)

    metrics = torch_fidelity.calculate_metrics(
        input1=gen_tmp, input2=real_tmp, cuda=torch.cuda.is_available(),
        fid=True, verbose=False, samples_find_deep=False
    )
    
    fid_score = metrics["frechet_inception_distance"]
    print("\n==== CelebA (preprocessed REAL 64×64; GEN 64×64) ====")
    print(f"Images per set: {n}")
    print(f"FID: {fid_score:.4f}")
    
    # ---- PRDC (same tensor pipeline) ----
    prdc_tf = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()])
    dl_real = DataLoader(ImageTensorDS(real_files, prdc_tf),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    dl_gen  = DataLoader(ImageTensorDS(gen_files,  prdc_tf),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    real_feats = extract_prdc(dl_real)
    gen_feats  = extract_prdc(dl_gen)

    k = min(args.nearest_k, max(1, n - 1))
    prdc = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=k)

    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall:    {prdc['recall']:.4f}")
    print(f"Density:   {prdc['density']:.4f}")
    print(f"Coverage:  {prdc['coverage']:.4f}")

    shutil.rmtree(real_tmp, ignore_errors=True)
    shutil.rmtree(gen_tmp,  ignore_errors=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser("CelebA eval (preprocessed REAL 64×64; GEN 64×64): FID + PRDC")
    p.add_argument("--real_dir", required=True, type=str)
    p.add_argument("--gen_dir",  required=True, type=str)
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nearest_k", type=int, default=5)
    args = p.parse_args()
    main(args)
