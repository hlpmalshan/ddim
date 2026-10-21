"""Compute FID and PRDC using torch-fidelity InceptionV3 features.

Uses RGB uint8 inputs; the extractor handles Inception resizing and normalization.
Source image files are never modified. Standard PRDC uses quadratic memory in
sample count; use --max_samples for a smaller, reproducible evaluation.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch_fidelity
from PIL import Image
from prdc import compute_prdc
from torch.utils.data import DataLoader, Dataset
from torch_fidelity import FeatureExtractorInceptionV3
from torchvision import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, root, image_size):
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
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


@torch.inference_mode()
def extract_features(dataset, extractor, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    features = []
    for images in tqdm(loader, desc='Extracting InceptionV3 features'):
        features.append(extractor(images.to(device))[0].cpu().numpy())
    return np.concatenate(features, axis=0)


def main(args):
    real = ImageDataset(args.real_dir, args.image_size)
    generated = ImageDataset(args.gen_dir, args.image_size)
    # Evaluate equally sized sets in stable filename order, without deleting files.
    count = min(len(real), len(generated))
    if args.max_samples is not None:
        count = min(count, args.max_samples)
    if count <= args.nearest_k:
        raise ValueError(f"Each set needs more than nearest_k={args.nearest_k} images; got {count}")
    real.paths = real.paths[:count]
    generated.paths = generated.paths[:count]
    device = torch.device(args.device)
    print('[Calculating FID with torch-fidelity]')
    fid_metrics = torch_fidelity.calculate_metrics(
        input1=generated,
        input2=real,
        cuda=device.type == 'cuda',
        cuda_device=(device.index if device.index is not None else 0),
        batch_size=args.batch_size,
        fid=True,
        isc=False,
        verbose=False,
    )
    fid_score = fid_metrics['frechet_inception_distance']
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).to(device).eval()
    print(f"PRDC space: InceptionV3 pool features (2048 dimensions), nearest_k={args.nearest_k}")
    print(f"Images per set: {count}")
    real_features = extract_features(real, extractor, device, args.batch_size, args.num_workers)
    fake_features = extract_features(generated, extractor, device, args.batch_size, args.num_workers)
    metrics = compute_prdc(real_features=real_features, fake_features=fake_features,
                           nearest_k=args.nearest_k)
    print('\n==== FID and InceptionV3 Feature-Space PRDC ====')
    print(f'FID: {fid_score:.4f}')
    for name in ('precision', 'recall', 'density', 'coverage'):
        print(f'{name.capitalize()}: {metrics[name]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real_dir', required=True)
    parser.add_argument('--gen_dir', required=True)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nearest_k', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    for name in ('image_size', 'batch_size', 'nearest_k'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.num_workers < 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error('--num_workers must be nonnegative and --max_samples must be positive')
    main(args)
