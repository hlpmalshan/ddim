# import os
# import argparse
# import glob
# import shutil
# import tempfile
# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# import torch
# import torch_fidelity
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from prdc import compute_prdc
# import idx2numpy

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

# # -------- Get MNIST digit 1 images as numpy arrays --------
# def get_mnist_digit_arrays(image_path, label_path, digit=2):
#     images = idx2numpy.convert_from_file(image_path)  # Shape: [N, 28, 28]
#     labels = idx2numpy.convert_from_file(label_path)  # Shape: [N]
    
#     # Filter for digit 1
#     digit_indices = np.where(labels == digit)[0]
#     print(f"Found {len(digit_indices)} images of digit {digit}")
    
#     return images[digit_indices]  # Shape: [N', 28, 28], uint8

# # -------- Prepare resized images for FID/IS in memory --------
# def prepare_resized_arrays(image_arrays, size=32, grayscale=False):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(size),          # shorter side resized to size
#         transforms.CenterCrop(size),      # center crop size x size
#         transforms.ToTensor(),           # [C, H, W], float in [0,1]
#     ])
#     resized_images = []
#     for img_array in tqdm(image_arrays, desc="Processing images"):
#         img_tensor = transform(img_array)  # [1, H, W]
#         if not grayscale:
#             img_tensor = img_tensor.repeat(3, 1, 1)  # Convert [1, H, W] to [3, H, W]
#         resized_images.append(img_tensor)
#     return torch.stack(resized_images)  # Shape: [N, C, H, W]

# # -------- Save tensor batch to temporary directory for torch_fidelity --------
# def save_tensors_to_temp_dir(tensor_batch, prefix="img"):
#     tmp_dir = tempfile.mkdtemp()
#     for i, img_tensor in enumerate(tqdm(tensor_batch, desc="Saving images to temp dir")):
#         img = transforms.ToPILImage()(img_tensor)
#         img.save(os.path.join(tmp_dir, f"{prefix}_{i}.png"))
#     return tmp_dir

# # -------- Dataset for PRDC --------
# class ImageDataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data  # Either file paths, numpy array, or list of numpy arrays
#         self.transform = transform
#         # Check if data is a numpy array or a list of numpy arrays
#         self.is_array = isinstance(data, np.ndarray) or (isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if self.is_array:
#             img = self.data[idx]  # numpy array
#             img = Image.fromarray(img)  # Convert to PIL, mode inferred (L for uint8 grayscale)
#         else:
#             img = Image.open(self.data[idx]).convert("L")  # From file, ensure grayscale
#         if self.transform:
#             img = self.transform(img)
#         return img, 0

# # -------- Extract raw grayscale images as numpy arrays for PRDC --------
# def extract_grayscale_features(dataloader):
#     imgs_list = []
#     for imgs, _ in tqdm(dataloader, desc="Collecting grayscale images for PRDC"):
#         # imgs: float tensor [B,1,H,W] in [0,1]
#         imgs_uint8 = (imgs * 255).to(torch.uint8)
#         imgs_np = imgs_uint8.squeeze(1).cpu().numpy()  # B,H,W (grayscale)
#         imgs_list.append(imgs_np)
#     return np.concatenate(imgs_list, axis=0)

# # -------- Main --------
# def main(image_path, label_path, gen_dir, batch_size=32, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
#     print("[Loading MNIST digit 1 images]")
#     real_images = get_mnist_digit_arrays(image_path, label_path, digit=1)  # Shape: [N, 28, 28]
    
#     print("[Loading generated images from gen_dir]")
#     gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))  # Only PNG files
#     if not gen_paths:
#         raise ValueError(f"No valid PNG images found in {gen_dir}")
#     gen_images = []
#     for f in tqdm(gen_paths, desc="Loading generated images"):
#         try:
#             img = Image.open(f).convert("L")  # Ensure grayscale
#             gen_images.append(np.array(img))  # Convert to numpy array
#         except Exception as e:
#             print(f"Warning: Skipping problematic file {f}: {e}")

#     print("[Preparing resized images for FID/IS]")
#     # Prepare MNIST images (in memory)
#     real_tensors = prepare_resized_arrays(real_images, grayscale=True)  # [N, 3, 32, 32]
#     # Prepare generated images
#     gen_tensors = prepare_resized_arrays(gen_images, grayscale=True)  # [N, 3, 32, 32]
    
#     print("[Saving resized images to temporary directories for FID/IS]")
#     real_tmp = save_tensors_to_temp_dir(real_tensors, prefix="mnist")
#     gen_tmp = save_tensors_to_temp_dir(gen_tensors, prefix="gen")
    
#     print("[Calculating FID and IS with torch-fidelity]")
#     metrics = torch_fidelity.calculate_metrics(
#         input1=gen_tmp,
#         input2=real_tmp,
#         cuda=torch.cuda.is_available(),
#         fid=True,
#         isc=True,
#         verbose=False
#     )
#     fid_score = metrics['frechet_inception_distance']
#     is_mean = metrics['inception_score_mean']
#     is_std = metrics['inception_score_std']

#     shutil.rmtree(real_tmp)
#     shutil.rmtree(gen_tmp)

#     print("[Calculating PRDC metrics with raw grayscale images]")
#     prdc_transform = transforms.Compose([
#         transforms.Resize(32),
#         transforms.CenterCrop(32),
#         transforms.ToTensor(),  # float tensor [1,H,W] in [0,1]
#     ])
#     ds_real = ImageDataset(real_images, prdc_transform)  # From numpy array
#     ds_gen = ImageDataset(gen_images, prdc_transform)    # From list of arrays
#     dl_real = DataLoader(ds_real, batch_size=batch_size, shuffle=False)
#     dl_gen = DataLoader(ds_gen, batch_size=batch_size, shuffle=False)

#     real_feats = extract_grayscale_features(dl_real)
#     gen_feats = extract_grayscale_features(dl_gen)

#     real_feats = real_feats.reshape(real_feats.shape[0], -1)
#     gen_feats = gen_feats.reshape(gen_feats.shape[0], -1)
#     prdc_metrics = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=5)

#     print("\n==== Final Metrics ====")
#     print(f"FID: {fid_score:.4f}")
#     print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
#     print(f"Precision: {prdc_metrics['precision']:.4f}")
#     print(f"Recall: {prdc_metrics['recall']:.4f}")
#     print(f"Density: {prdc_metrics['density']:.4f}")
#     print(f"Coverage: {prdc_metrics['coverage']:.4f}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Calculate FID, IS, PRDC metrics for MNIST digit 1 vs generated grayscale images")
#     parser.add_argument("--image_path", type=str, help="Path to MNIST train-images-idx3-ubyte file")
#     parser.add_argument("--label_path", type=str, help="Path to MNIST train-labels-idx1-ubyte file")
#     parser.add_argument("--gen_dir", type=str, help="Directory with generated grayscale PNG images")
#     args = parser.parse_args()
#     main(args.image_path, args.label_path, args.gen_dir)

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

def load_mnist_digit(image_path: str, label_path: str, digit: int) -> np.ndarray:
    """Return MNIST images of a given digit as uint8 arrays [N,28,28]."""
    imgs = idx2numpy.convert_from_file(image_path)    # [N,28,28], uint8
    labels = idx2numpy.convert_from_file(label_path)  # [N], uint8
    sel = np.where(labels == digit)[0]
    return imgs[sel]

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

def save_tensor_batch_to_dir(tensor_batch: torch.Tensor, prefix: str) -> str:
    """Save [N,3,32,32] to a temp dir as PNGs. Returns dir path."""
    tmp = tempfile.mkdtemp()
    to_pil = transforms.ToPILImage()
    for i, img_t in enumerate(tensor_batch):
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
    digit: int = 1,
    batch_size: int = 64,
    nearest_k: int = 5,
):
    # Real: select digit and pad to 32×32
    real_28 = load_mnist_digit(image_path, label_path, digit)  # [N,28,28]
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

    real_feat = collect_flat_features(dl_real)  # [N,1024]
    gen_feat  = collect_flat_features(dl_gen)   # [M,1024]
    prdc = compute_prdc(real_features=real_feat, fake_features=gen_feat, nearest_k=nearest_k)

    # --- Output ---
    print("\n==== Metrics (Digit = {}) ====".format(digit))
    print(f"FID: {fid:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall: {prdc['recall']:.4f}")
    print(f"Density: {prdc['density']:.4f}")
    print(f"Coverage: {prdc['coverage']:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("FID/IS/PRDC for a specific MNIST digit vs generated 32×32 PNGs")
    p.add_argument("--image_path", required=True, help="MNIST train-images-idx3-ubyte")
    p.add_argument("--label_path", required=True, help="MNIST train-labels-idx1-ubyte")
    p.add_argument("--gen_dir",    required=True, help="Directory of generated PNGs (32×32 or resizable)")
    p.add_argument("--digit",      type=int, default=1, help="MNIST digit to evaluate (0–9)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--nearest_k",  type=int, default=5)
    args = p.parse_args()
    main(args.image_path, args.label_path, args.gen_dir, args.digit, args.batch_size, args.nearest_k)
