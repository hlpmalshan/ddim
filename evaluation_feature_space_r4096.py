"""Compute FD and PRDC in the R4096 embedding: randomly initialized VGG16 fc2 features.

R4096 is the random embedding of Naeem et al. (2020), "Reliable Fidelity and
Diversity Metrics for Generative Models": the same VGG16 architecture as T4096,
but with randomly initialized, never-trained weights. Features are the 4096-d
output of the second fully connected layer (after ReLU; dropout is inactive in
eval mode). Images are resized to 224x224 and normalized with ImageNet statistics,
exactly as for T4096 in evaluation_feature_space_vgg16.py.

The random weights come from torchvision's default VGG16 initialization under a
fixed --seed, so every run with the same seed uses the identical network; always
use the same seed for all models being compared. The printed weight checksum lets
you confirm that runs used the same network.

PRDC depends only on nearest-neighbour distances, so the arbitrary scale of random
features does not affect it. FD_R4096 (the FID formula on these features) depends
on that scale: compare it only between models evaluated with the same seed.
Dataset, feature extraction and Frechet distance are imported from
evaluation_feature_space_vgg16.py. Source image files are never modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from prdc import compute_prdc

from evaluation_feature_space_vgg16 import (ImageDataset, extract_features,
                                            feature_statistics, frechet_distance)


def build_r4096(device, seed):
    # Build on CPU under a fixed seed so the random weights are reproducible.
    torch.manual_seed(seed)
    model = torchvision.models.vgg16(weights=None)
    # Drop the final 4096->1000 classifier layer: output is fc2 (4096-d).
    model.classifier = model.classifier[:-1]
    checksum = float(sum(p.double().abs().sum() for p in model.parameters()))
    return model.to(device).eval().requires_grad_(False), checksum


def real_features_cached(real, model, args, checksum):
    """Reuse saved real-image features when they match this real set and network."""
    meta = {'real_dir': str(Path(args.real_dir).resolve()), 'count': len(real.paths),
            'resize': args.resize, 'embedding': 'vgg16_random_init_fc2_relu',
            'seed': args.seed, 'weight_checksum': round(checksum, 4)}
    if args.real_cache:
        cache = Path(args.real_cache)
        meta_path = cache.with_suffix('.json')
        if cache.is_file() and meta_path.is_file() and json.loads(meta_path.read_text()) == meta:
            print(f"Loading cached real features: {cache}")
            return np.load(cache)
    features = extract_features(real, model, args.device, args.batch_size,
                                args.num_workers, 'Extracting R4096 features (real)')
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

    model, checksum = build_r4096(args.device, args.seed)
    print(f"Feature space: R4096 = randomly initialized VGG16 fc2 features (4096 dimensions), "
          f"seed {args.seed}, input {args.resize}x{args.resize}, nearest_k={args.nearest_k}")
    print(f"Random-weight checksum: {checksum:.4f}")
    print(f"Images per set: {count}")
    real_features = real_features_cached(real, model, args, checksum)
    fake_features = extract_features(generated, model, args.device, args.batch_size,
                                     args.num_workers, 'Extracting R4096 features (generated)')
    print('[Computing Frechet distance]')
    mu_r, sigma_r = feature_statistics(real_features)
    mu_g, sigma_g = feature_statistics(fake_features)
    fd = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    metrics = compute_prdc(real_features=real_features, fake_features=fake_features,
                           nearest_k=args.nearest_k)
    print('\n==== Random VGG16 (R4096) Feature Space: FD and PRDC ====')
    print(f'FD_R4096: {fd:.4f}')
    for name in ('precision', 'recall', 'density', 'coverage'):
        print(f'{name.capitalize()}: {metrics[name]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--real_dir', required=True)
    parser.add_argument('--gen_dir', required=True)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the random VGG16 weights; use the same value for all models')
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
