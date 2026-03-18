#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# Preserve old behavior unless overridden by the caller.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def has_images(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(IMG_EXTS):
                return True
    return False


def count_labels(dataset, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    if isinstance(dataset, Subset):
        targets = dataset.dataset.targets
        for idx in dataset.indices:
            counts[targets[idx]] += 1
    elif hasattr(dataset, "targets"):
        for t in dataset.targets:
            counts[t] += 1
    else:
        for _, t in dataset:
            counts[int(t)] += 1
    return counts


def build_transforms(img_size):
    train_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(model_name: str, pretrained: bool) -> nn.Module:
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
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def freeze_backbone(model: nn.Module, model_name: str):
    for param in model.parameters():
        param.requires_grad = False
    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "vgg16":
        for param in model.classifier.parameters():
            param.requires_grad = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(rows)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, compute_cm=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cm = torch.zeros((2, 2), dtype=torch.long) if compute_cm else None

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

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc, cm


def metrics_from_cm(cm: torch.Tensor):
    tn = cm[0, 0].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()
    tp = cm[1, 1].item()
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pneumonia_dir = os.path.join(script_dir, "Pneumonia_dataset")
    default_data_root = os.path.join(default_pneumonia_dir, "chest_xray")
    default_output_dir = os.path.join(default_pneumonia_dir, "outputs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=default_data_root)
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model", choices=["resnet50", "vgg16"], default="resnet50")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch:", torch.__version__)
    print("Device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    data_root = os.path.abspath(args.data_root)
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        nested = os.path.join(data_root, "chest_xray")
        if os.path.isdir(os.path.join(nested, "train")) and os.path.isdir(os.path.join(nested, "test")):
            data_root = nested
            train_dir = os.path.join(data_root, "train")
            test_dir = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing: {test_dir}")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Train:", train_dir)
    print("Test :", test_dir)
    print("Output:", out_dir)

    img_size = tuple(args.img_size)
    train_tf, eval_tf = build_transforms(img_size)

    val_dir = os.path.join(data_root, "val")
    use_val_dir = has_images(val_dir)

    train_base = datasets.ImageFolder(train_dir, transform=train_tf)
    class_names = train_base.classes

    if use_val_dir:
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        train_ds = train_base
        print("Validation: using separate val/ folder")
    else:
        eval_base = datasets.ImageFolder(train_dir, transform=eval_tf)
        indices = list(range(len(train_base)))
        random.Random(args.seed).shuffle(indices)
        val_size = int(0.2 * len(indices))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        train_ds = Subset(train_base, train_idx)
        val_ds = Subset(eval_base, val_idx)
        print("Validation: 20% split from train/")

    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    if hasattr(val_ds, "classes") and val_ds.classes != class_names:
        print("Warning: val class order differs from train classes")
    if hasattr(test_ds, "classes") and test_ds.classes != class_names:
        print("Warning: test class order differs from train classes")

    train_counts = count_labels(train_ds, len(class_names))
    val_counts = count_labels(val_ds, len(class_names))
    test_counts = count_labels(test_ds, len(class_names))

    print("Classes:", class_names)
    print("Train counts:", dict(zip(class_names, train_counts)))
    print("Val counts  :", dict(zip(class_names, val_counts)))
    print("Test counts :", dict(zip(class_names, test_counts)))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, pretrained=(not args.no_pretrained))
    if args.freeze_backbone:
        freeze_backbone(model, args.model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_acc = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")
    final_path = os.path.join(out_dir, "xray_pneumonia_final.pt")
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, compute_cm=False)
        elapsed = time.time() - start

        history_rows.append([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)

    test_loss, test_acc, cm = evaluate(model, test_loader, criterion, device, compute_cm=True)
    metrics = metrics_from_cm(cm)

    print("\nTEST metrics:")
    print(f"  loss: {test_loss:.4f}")
    print(f"  acc : {test_acc:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm.numpy())

    torch.save({"epoch": args.epochs, "model": model.state_dict()}, final_path)

    save_history(os.path.join(out_dir, "history.csv"), history_rows)
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm.numpy(), fmt="%d", delimiter=",")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
            f,
            indent=2,
        )
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "data_root": data_root,
                "output_dir": out_dir,
                "img_size": img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "model": args.model,
                "pretrained": not args.no_pretrained,
                "freeze_backbone": args.freeze_backbone,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {best_path}")
    print(f"Saved: {final_path}")
    print(f"Outputs directory: {out_dir}")


if __name__ == "__main__":
    main()
