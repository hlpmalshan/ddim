#!/usr/bin/env python3
import os, re, glob, shutil, tempfile, argparse, warnings
from typing import List, Optional, Set, Tuple
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch_fidelity
from prdc import compute_prdc

warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

VALID_EXTS = {".jpg", ".jpeg", ".png"}

# ---------- filename parsing ----------
def class_id_from_stem(stem: str) -> Optional[int]:
    # third numeric token
    nums = re.findall(r"\d+", stem)
    if len(nums) < 3:
        return None
    try:
        return int(nums[2])
    except ValueError:
        return None

def filter_by_classes(paths: List[str], wanted: Optional[Set[int]]) -> List[str]:
    print(wanted)
    if wanted is None:
        return paths
    out = []
    for p in paths:
        cid = class_id_from_stem(os.path.splitext(os.path.basename(p))[0])
        if cid is not None and cid in wanted:
            out.append(p)
    return out

def parse_classes(arg: Optional[str]) -> Optional[Set[int]]:
    if not arg:
        return None
    toks = [t for t in re.split(r"[,\s]+", arg.strip()) if t]
    return set(int(t) for t in toks)

# ---------- IO helpers ----------
def list_images_flat(root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "*")))
    return [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS]

def clean_image_dir_inplace(img_dir: str) -> None:
    files = glob.glob(os.path.join(img_dir, "*"))
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in VALID_EXTS:
            try:
                os.remove(f)
            except Exception:
                pass
            continue
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img.verify()
        except Exception:
            try:
                os.remove(f)
            except Exception:
                pass

def copy_resized(paths: List[str], size: int) -> str:
    """Copy a given list of files into a temp dir with resize+center-crop."""
    tmp_dir = tempfile.mkdtemp()
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])
    for f in tqdm(paths, desc="Preparing resized copies"):
        try:
            img = Image.open(f).convert("RGB")
            img = tfm(img)
            img.save(os.path.join(tmp_dir, os.path.basename(f)))
        except Exception:
            # skip silently
            pass
    return tmp_dir

# ---------- Dataset for PRDC from explicit file list ----------
class ImageListDataset(Dataset):
    def __init__(self, paths: List[str], image_size: int):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # float [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, 0

def extract_rgb_features(dataloader: DataLoader) -> np.ndarray:
    chunks = []
    for imgs, _ in tqdm(dataloader, desc="Collecting RGB images for PRDC"):
        imgs_uint8 = (imgs * 255.0).clamp(0, 255).to(torch.uint8)  # B,C,H,W
        arr = imgs_uint8.permute(0, 2, 3, 1).cpu().numpy()         # B,H,W,C
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)

# ---------- main pipeline ----------
def main(real_dir: str, gen_dir: str, classes: Optional[str], image_size: int, batch_size: int,
         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    print("[Sanity] Cleaning input directories")
    clean_image_dir_inplace(real_dir)
    clean_image_dir_inplace(gen_dir)

    wanted = parse_classes(classes)
    real_all = list_images_flat(real_dir)
    gen_all  = list_images_flat(gen_dir)
    real_sel = filter_by_classes(real_all, wanted)
    gen_sel  = gen_all

    if len(real_sel) == 0 or len(gen_sel) == 0:
        raise SystemExit("No images after filtering. Check --classes and filename pattern (3rd number).")

    # Balance counts globally
    n = min(len(real_sel), len(gen_sel))
    real_sel = real_sel[:n]
    gen_sel  = gen_sel[:n]
    print(f"[Info] Using {n} images per set after filtering")

    # FID/IS with resized temp copies
    print("[FID/IS] Preparing resized copies")
    real_tmp = copy_resized(real_sel, size=image_size)
    gen_tmp  = copy_resized(gen_sel,  size=image_size)

    print("[FID/IS] Computing metrics via torch-fidelity")
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_tmp,
        input2=real_tmp,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        verbose=False
    )
    fid_score = metrics['frechet_inception_distance']
    is_mean   = metrics['inception_score_mean']
    is_std    = metrics['inception_score_std']
    
    print("\n==== Final Metrics (filtered) ====")
    if wanted is None:
        print("Classes: all")
    else:
        print(f"Classes: {sorted(list(wanted))}")
    print(f"Count per set: {n}")
    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

    # Cleanup resized temp dirs
    shutil.rmtree(real_tmp)
    shutil.rmtree(gen_tmp)

    # PRDC on raw resized tensors from explicit lists
    print("[PRDC] Building dataloaders")
    ds_real = ImageListDataset(real_sel, image_size)
    ds_gen  = ImageListDataset(gen_sel,  image_size)
    dl_real = DataLoader(ds_real, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_gen  = DataLoader(ds_gen,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    real_feats = extract_rgb_features(dl_real).reshape(len(ds_real), -1)
    gen_feats  = extract_rgb_features(dl_gen).reshape(len(ds_gen),  -1)

    prdc_metrics = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=10)

    
    print(f"Precision: {prdc_metrics['precision']:.4f}")
    print(f"Recall: {prdc_metrics['recall']:.4f}")
    print(f"Density: {prdc_metrics['density']:.4f}")
    print(f"Coverage: {prdc_metrics['coverage']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FID/IS/PRDC on filtered CIFAR-10 classes parsed from filenames")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--gen_dir",  type=str, required=True, help="Directory with generated images")
    parser.add_argument("--classes",  type=str, default=None,
                        help='Filter to class IDs by third number in filename. Example: "7 3" or "7,3". Omit for all.')
    parser.add_argument("--image_size", type=int, default=32, help="Resize/crop resolution")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for PRDC extraction")
    args = parser.parse_args()
    main(args.real_dir, args.gen_dir, args.classes, args.image_size, args.batch_size)
