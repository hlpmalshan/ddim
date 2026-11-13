"""Compute FD and PRDC in the T4096 embedding: ImageNet-pretrained VGG16 fc2 features.

T4096 is the embedding of Naeem et al. (2020), "Reliable Fidelity and Diversity
Metrics for Generative Models": torchvision VGG16 (IMAGENET1K_V1) with its last
classifier layer removed, giving the 4096-d output of the second fully connected
layer (after ReLU; dropout is inactive in eval mode). Images are resized to
224x224 and normalized with ImageNet statistics.

FD_VGG16 is the Frechet distance (the FID formula) on these VGG16 features. It is
not FID: FID is defined on InceptionV3 features, so FD_VGG16 has a different scale
and is not comparable to published FID values. PRDC lines use the same printed
format as evaluation_feature_space.py. Source image files are never modified.
PRDC memory is quadratic in sample count; use --max_samples for a smaller,
reproducible evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from prdc import compute_prdc
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, root, resize):
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"Image directory does not exist: {root}")
        self.paths = []
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                self.paths.append(path)
            except (OSError, ValueError):
                print(f"Skipping invalid image: {path}")
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


def build_t4096(device):
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    model = torchvision.models.vgg16(weights=weights)
    # Drop the final 4096->1000 classifier layer: output is fc2 (4096-d).
    model.classifier = model.classifier[:-1]
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def extract_features(dataset, model, device, batch_size, num_workers, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    features = []
    for images in tqdm(loader, desc=desc):
        features.append(model(images.to(device)).float().cpu().numpy())
    return np.concatenate(features, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between N(mu1, sigma1) and N(mu2, sigma2), as in pytorch-fid."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # Near-singular product (e.g. sparse ReLU features): regularize the diagonal.
        print(f"sqrtm not finite; adding {eps} to the covariance diagonals")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {max_imag} in sqrtm result")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def feature_statistics(features):
    features = features.astype(np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


def real_features_cached(real, model, args):
    """Reuse saved real-image features when they match this exact real set."""
    meta = {'real_dir': str(Path(args.real_dir).resolve()), 'count': len(real.paths),
            'resize': args.resize, 'embedding': 'vgg16_imagenet1k_v1_fc2_relu'}
    if args.real_cache:
        cache = Path(args.real_cache)
        meta_path = cache.with_suffix('.json')
        if cache.is_file() and meta_path.is_file() and json.loads(meta_path.read_text()) == meta:
            print(f"Loading cached real features: {cache}")
            return np.load(cache)
    features = extract_features(real, model, args.device, args.batch_size,
                                args.num_workers, 'Extracting VGG16 features (real)')
    if args.real_cache:
        np.save(cache, features)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved real features to {cache}")
    return features


def main(args):
    real = ImageDataset(args.real_dir, args.resize)
    generated = ImageDataset(args.gen_dir, args.resize)
    # Evaluate equally sized sets in stable filename order, without deleting files.
    count = min(len(real), len(generated))
    if args.max_samples is not None:
        count = min(count, args.max_samples)
    if count <= args.nearest_k:
        raise ValueError(f"Each set needs more than nearest_k={args.nearest_k} images; got {count}")
    real.paths = real.paths[:count]
    generated.paths = generated.paths[:count]

    model = build_t4096(args.device)
    print(f"Feature space: T4096 = ImageNet VGG16 fc2 features (4096 dimensions), "
          f"input {args.resize}x{args.resize}, nearest_k={args.nearest_k}")
    print(f"Images per set: {count}")
    real_features = real_features_cached(real, model, args)
    fake_features = extract_features(generated, model, args.device, args.batch_size,
                                     args.num_workers, 'Extracting VGG16 features (generated)')
    print('[Computing Frechet distance]')
    mu_r, sigma_r = feature_statistics(real_features)
    mu_g, sigma_g = feature_statistics(fake_features)
    fd = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    metrics = compute_prdc(real_features=real_features, fake_features=fake_features,
                           nearest_k=args.nearest_k)
    print('\n==== VGG16 (T4096) Feature Space: FD and PRDC ====')
    print(f'FD_VGG16: {fd:.4f}')
    for name in ('precision', 'recall', 'density', 'coverage'):
        print(f'{name.capitalize()}: {metrics[name]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--real_dir', required=True)
    parser.add_argument('--gen_dir', required=True)
    parser.add_argument('--resize', type=int, default=224, help='VGG16 input size (default 224)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nearest_k', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--real_cache', default=None,
                        help='Optional .npy path to save/reuse real-image features across models')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    for name in ('resize', 'batch_size', 'nearest_k'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.num_workers < 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error('--num_workers must be nonnegative and --max_samples must be positive')
    main(args)
