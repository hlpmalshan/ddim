import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import Flowers102
from torchvision.datasets import OxfordIIITPet
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.celebahq import CelebAHQ
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size), 
                transforms.ToTensor()
            ]
        )


    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

        # >>> ADD: class filtering like MNIST selected_digits
        selected_classes = getattr(config.data, "selected_classes", None)  # e.g., [0,1,7]
        remap_to_sequential = getattr(config.data, "remap_to_sequential", False)  # optional

        if selected_classes:
            selected_classes = list(map(int, selected_classes))
            # optional label remap: old -> {0..K-1} in sorted order
            if remap_to_sequential:
                remap = {c: i for i, c in enumerate(sorted(selected_classes))}
                def map_fn(t):
                    return remap[int(t)] if int(t) in remap else -1
                # set before subsetting so __getitem__ returns remapped labels
                dataset.target_transform = map_fn
                test_dataset.target_transform = map_fn

            tr_idx = [i for i, t in enumerate(dataset.targets) if t in selected_classes]
            te_idx = [i for i, t in enumerate(test_dataset.targets) if t in selected_classes]
            dataset = Subset(dataset, tr_idx)
            test_dataset = Subset(test_dataset, te_idx)

            # keep num_classes in sync if present in config
            if hasattr(config.data, "num_classes"):
                config.data.num_classes = len(selected_classes)


    elif config.data.dataset == "CIFAR100":
        dataset = CIFAR100(
            os.path.join(args.exp, "datasets", "cifar100"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR100(
            os.path.join(args.exp, "datasets", "cifar100_test"),
            train=False,
            download=True,
            transform=test_transform,
        )
    
    elif config.data.dataset == "OXFORD_FLOWERS":
        dataset = Flowers102(
            root=os.path.join(args.exp, "datasets", "oxford_flowers"),
            split="train",
            download=True,
            transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )
        
        test_dataset = Flowers102(
            root=os.path.join(args.exp, "datasets", "oxford_flowers"),
            split="test",
            download=True,
            transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
        )
    
    elif config.data.dataset == "OXFORD_IIIT_PET":
        dataset = OxfordIIITPet(
            root=os.path.join(args.exp, "datasets", "oxford_pets"),
            split="trainval",
            download=True,
            transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )
        
        test_dataset = OxfordIIITPet(
            root=os.path.join(args.exp, "datasets", "oxford_pets"),
            split="test",
            download=True,
            transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
        )
    
    # Inside get_dataset function, add this before the final else: dataset, test_dataset = None, None
    elif config.data.dataset == "MNIST":
        if config.data.random_flip:
            train_transform = transforms.Compose([
                transforms.Pad(2),  # Pad 28x28 to 32x32
                transforms.Resize(config.data.image_size),  # Ensure 32x32
                transforms.RandomHorizontalFlip(p=0.5),  # Unlikely for MNIST, but respect config
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Pad(2),
                transforms.Resize(config.data.image_size),
                transforms.ToTensor(),
            ])
        else:
            train_transform = test_transform = transforms.Compose([
                transforms.Pad(2),  # Pad 28x28 to 32x32
                transforms.Resize(config.data.image_size),  # Ensure 32x32
                transforms.ToTensor(),
            ])
        
        dataset = MNIST(
            root=os.path.join(args.exp, "datasets", "mnist"),
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = MNIST(
            root=os.path.join(args.exp, "datasets", "mnist_test"),
            train=False,
            download=True,
            transform=test_transform,
        )
        
        # Filter digits if specified
        selected_digits = getattr(config.data, 'selected_digits', None)
        if selected_digits:
            labels = [label for _, label in dataset]
            indices = [i for i, l in enumerate(labels) if l in selected_digits]
            dataset = Subset(dataset, indices)
            test_labels = [label for _, label in test_dataset]
            test_indices = [i for i, l in enumerate(test_labels) if l in selected_digits]
            test_dataset = Subset(test_dataset, test_indices)
    
    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    elif config.data.dataset == "CELEBA_HQ":
        # Prefer an explicit path in config; else default under exp dir
        dataset_path = getattr(config.data, 'dataset_path', None)
        if dataset_path is None:
            dataset_path = os.path.join(args.exp, "datasets", "celebahq256_imgs")
        if config.data.random_flip:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform = test_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size), 
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            )

        dataset = CelebAHQ(
            root=dataset_path,
            split="train", 
            transform=train_transform,
        )

        # Try to use a dedicated test split, fall back handled by dataset class
        test_split = getattr(config.data, 'valid_split', 'valid')
        test_dataset = CelebAHQ(
            root=dataset_path,
            split=test_split,
            transform=test_transform,
        )
    
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
