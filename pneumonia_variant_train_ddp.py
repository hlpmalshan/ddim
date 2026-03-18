#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fname))
    paths.sort()
    return paths


class BinaryFolderDataset(Dataset):
    def __init__(self, normal_dir: str, pneumonia_dir: str, transform=None):
        if not os.path.isdir(normal_dir):
            raise FileNotFoundError(f"Missing normal_dir: {normal_dir}")
        if not os.path.isdir(pneumonia_dir):
            raise FileNotFoundError(f"Missing pneumonia_dir: {pneumonia_dir}")

        normal_paths = list_images(normal_dir)
        pneumonia_paths = list_images(pneumonia_dir)
        if len(normal_paths) == 0:
            raise RuntimeError(f"No images found in normal_dir: {normal_dir}")
        if len(pneumonia_paths) == 0:
            raise RuntimeError(f"No images found in pneumonia_dir: {pneumonia_dir}")

        self.samples: List[Tuple[str, int]] = []
        self.samples.extend([(p, 0) for p in normal_paths])
        self.samples.extend([(p, 1) for p in pneumonia_paths])
        self.transform = transform
        self.num_normal = len(normal_paths)
        self.num_pneumonia = len(pneumonia_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ShardedSequentialSampler(Sampler):
    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.indices = list(range(len(dataset)))[rank::world_size]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def init_distributed() -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return True, rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def build_model(model_name: str, pretrained: bool, head_type: str) -> nn.Module:
    if model_name == "resnet50":
        if pretrained:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
            except Exception:
                model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        if head_type == "mlp3":
            model.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )
        else:
            model.fc = nn.Linear(in_features, 2)
        return model

    if model_name == "vgg16":
        if pretrained:
            try:
                weights = models.VGG16_Weights.DEFAULT
                model = models.vgg16(weights=weights)
            except Exception:
                model = models.vgg16(pretrained=True)
        else:
            model = models.vgg16(pretrained=False)
        in_features = model.classifier[-1].in_features
        if head_type == "mlp3":
            model.classifier[-1] = nn.Identity()
            model.classifier = nn.Sequential(
                *list(model.classifier[:-1]),
                # nn.Linear(in_features, 128),
                # nn.ReLU(inplace=True),
                # nn.Dropout(0.2),
                nn.Linear(in_features, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )
        else:
            model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def set_trainable(model: nn.Module, model_name: str, train_mode: str):
    for param in model.parameters():
        param.requires_grad = False

    if train_mode == "all":
        for param in model.parameters():
            param.requires_grad = True
        return

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "vgg16":
        for param in model.classifier.parameters():
            param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


def print_counts(name: str, normal_count: int, pneumonia_count: int):
    total = normal_count + pneumonia_count
    print(f"{name} counts -> NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}, total: {total}")


def build_class_weights(num_normal: int, num_pneumonia: int, device: torch.device) -> torch.Tensor:
    total = float(num_normal + num_pneumonia)
    w_normal = total / max(1.0, 2.0 * float(num_normal))
    w_pneumonia = total / max(1.0, 2.0 * float(num_pneumonia))
    return torch.tensor([w_normal, w_pneumonia], dtype=torch.float32, device=device)


def build_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    metrics = torch.tensor([total_loss, total_correct, total_samples], device=device, dtype=torch.float64)
    metrics = reduce_sum(metrics)
    total_loss, total_correct, total_samples = metrics.tolist()
    avg_loss = total_loss / max(1.0, total_samples)
    acc = total_correct / max(1.0, total_samples)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_cm: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cm = torch.zeros((2, 2), device=device, dtype=torch.long) if compute_cm else None

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        if cm is not None:
            for t, p in zip(targets.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    metrics = torch.tensor([total_loss, total_correct, total_samples], device=device, dtype=torch.float64)
    metrics = reduce_sum(metrics)
    total_loss, total_correct, total_samples = metrics.tolist()

    if cm is not None:
        cm = reduce_sum(cm)
        cm = cm.cpu()

    avg_loss = total_loss / max(1.0, total_samples)
    acc = total_correct / max(1.0, total_samples)
    return avg_loss, acc, cm


@torch.no_grad()
def evaluate_with_probs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cm = torch.zeros((2, 2), device=device, dtype=torch.long)
    local_labels = []
    local_probs = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        for t, p in zip(targets.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        local_labels.append(targets.detach().cpu().numpy())
        local_probs.append(probs.detach().cpu().numpy())

    metrics = torch.tensor([total_loss, total_correct, total_samples], device=device, dtype=torch.float64)
    metrics = reduce_sum(metrics)
    total_loss, total_correct, total_samples = metrics.tolist()

    cm = reduce_sum(cm).cpu()

    labels_np = np.concatenate(local_labels, axis=0) if local_labels else np.zeros((0,), dtype=np.int64)
    probs_np = np.concatenate(local_probs, axis=0) if local_probs else np.zeros((0,), dtype=np.float32)

    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        gathered_labels = [None for _ in range(world_size)]
        gathered_probs = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_labels, labels_np)
        dist.all_gather_object(gathered_probs, probs_np)
        if rank == 0:
            labels_np = np.concatenate([x for x in gathered_labels if x is not None and len(x) > 0], axis=0)
            probs_np = np.concatenate([x for x in gathered_probs if x is not None and len(x) > 0], axis=0)
        else:
            labels_np = None
            probs_np = None

    avg_loss = total_loss / max(1.0, total_samples)
    acc = total_correct / max(1.0, total_samples)
    return avg_loss, acc, cm, labels_np, probs_np


def compute_metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    f1 = (2 * precision * sensitivity) / max(1e-12, precision + sensitivity)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_train_dir = os.path.join(script_dir, "Pneumonia_dataset", "chest_xray_preprocessed", "train")
    default_val_dir = os.path.join(script_dir, "Pneumonia_dataset", "chest_xray_preprocessed", "val")
    default_test_dir = os.path.join(script_dir, "Pneumonia_dataset", "chest_xray_preprocessed", "test")
    default_output_dir = os.path.join(script_dir, "Pneumonia_dataset", "outputs_variant_ddp")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=default_train_dir)
    parser.add_argument("--val_dir", default=default_val_dir)
    parser.add_argument("--test_dir", default=default_test_dir)
    parser.add_argument(
        "--normal_variant",
        default="NORMAL",
        choices=["NORMAL", "NORMAL_DDPM_100", "NORMAL_DDPM_25", "NORMAL_DDPM_50", "NORMAL_DDPM_75", "NORMAL_ISO_25", "NORMAL_ISO_50", "NORMAL_ISO_75", "NORMAL_ISO_100"],
        help="Which NORMAL variant folder to use for training",
    )
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "vgg16"])
    parser.add_argument("--loss_type", default="ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument(
        "--train_mode",
        choices=["head", "mlp3", "all"],
        default="head",
        help="head: train only classifier head; mlp3: 3-layer MLP head; all: train entire model",
    )
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    distributed, rank, world_size, device = init_distributed()
    set_seed(args.seed + rank)

    if rank == 0:
        print("Distributed:", distributed, "| World size:", world_size)
        print("Device:", device)
        print("Model:", args.model)
        print("Normal variant (train):", args.normal_variant)
        print("Train dir:", args.train_dir)
        print("Val dir  :", args.val_dir)
        print("Test dir :", args.test_dir)
        print("Output dir:", args.output_dir)
        print("Train mode:", args.train_mode)
        print("Loss type:", args.loss_type)

    train_normal_dir = os.path.join(args.train_dir, args.normal_variant)
    train_pneumonia_dir = os.path.join(args.train_dir, "PNEUMONIA")
    val_normal_dir = os.path.join(args.val_dir, "NORMAL")
    val_pneumonia_dir = os.path.join(args.val_dir, "PNEUMONIA")
    test_normal_dir = os.path.join(args.test_dir, "NORMAL")
    test_pneumonia_dir = os.path.join(args.test_dir, "PNEUMONIA")

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = BinaryFolderDataset(train_normal_dir, train_pneumonia_dir, transform=train_tf)
    val_ds = BinaryFolderDataset(val_normal_dir, val_pneumonia_dir, transform=eval_tf)
    test_ds = BinaryFolderDataset(test_normal_dir, test_pneumonia_dir, transform=eval_tf)

    if rank == 0:
        print_counts("Train", train_ds.num_normal, train_ds.num_pneumonia)
        print_counts("Val  ", val_ds.num_normal, val_ds.num_pneumonia)
        print_counts("Test ", test_ds.num_normal, test_ds.num_pneumonia)

    class_weights = build_class_weights(train_ds.num_normal, train_ds.num_pneumonia, device)
    if rank == 0:
        print(f"Class weights (NORMAL, PNEUMONIA): {class_weights.tolist()}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = ShardedSequentialSampler(val_ds, rank, world_size) if distributed else None
    test_sampler = ShardedSequentialSampler(test_ds, rank, world_size) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args.model, pretrained=True, head_type=("mlp3" if args.train_mode == "mlp3" else "head"))
    set_trainable(model, args.model, args.train_mode)
    model.to(device)
    if rank == 0:
        print("Trainable params:", count_trainable_params(model))
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")
    last_path = os.path.join(args.output_dir, "last_model.pt")

    for epoch in range(1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp=args.amp and device.type == "cuda"
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, compute_cm=False)
        elapsed = time.time() - start

        if rank == 0:
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)

    test_loss, test_acc, cm, y_true, y_prob = evaluate_with_probs(model, test_loader, criterion, device)
    if rank == 0:
        metrics = compute_metrics_from_probs(y_true, y_prob, threshold=args.decision_threshold)
        print("\nTest metrics:")
        print(f"  variant : {args.normal_variant}")
        print(f"  loss: {test_loss:.4f}")
        print(f"  acc : {test_acc:.4f}")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall   : {metrics['recall']:.4f}")
        print(f"  f1       : {metrics['f1']:.4f}")
        print(f"  auroc    : {metrics['auroc']:.4f}")
        print(f"  auprc    : {metrics['auprc']:.4f}")
        print(f"  bal_acc  : {metrics['balanced_accuracy']:.4f}")
        print(f"  sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  specificity: {metrics['specificity']:.4f}")
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(np.array(metrics["confusion_matrix"], dtype=np.int64))

        torch.save({"epoch": args.epochs, "model": model.state_dict()}, last_path)
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        output_metrics = {
            "normal_variant": args.normal_variant,
            "model": args.model,
            "train_mode": args.train_mode,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "decision_threshold": args.decision_threshold,
            "seed": args.seed,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "sensitivity": float(metrics["sensitivity"]),
            "specificity": float(metrics["specificity"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tn": int(metrics["tn"]),
            "confusion_matrix": metrics["confusion_matrix"],
        }
        with open(metrics_path, "w") as f:
            json.dump(output_metrics, f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    main()
