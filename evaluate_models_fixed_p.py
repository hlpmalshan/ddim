#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

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
        normal_paths = list_images(normal_dir)
        pneumonia_paths = list_images(pneumonia_dir)
        if len(normal_paths) == 0:
            raise RuntimeError(f"No images found in normal_dir: {normal_dir}")
        if len(pneumonia_paths) == 0:
            raise RuntimeError(f"No images found in pneumonia_dir: {pneumonia_dir}")

        self.samples = [(p, 0) for p in normal_paths] + [(p, 1) for p in pneumonia_paths]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_model(model_name: str, train_mode: str) -> nn.Module:
    head_type = "mlp3" if train_mode == "mlp3" else "head"
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
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
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        if head_type == "mlp3":
            model.classifier[-1] = nn.Identity()
            model.classifier = nn.Sequential(
                *list(model.classifier[:-1]),
                nn.Linear(in_features, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )
        else:
            model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def load_state_dict(model: nn.Module, ckpt_path: str, device: torch.device):
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def infer_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_acc_f1(y_true: np.ndarray, y_prob: np.ndarray, p: float) -> Tuple[float, float]:
    y_pred = (y_prob >= p).astype(np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    total = tn + fp + fn + tp
    acc = (tp + tn) / max(1, total)


    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    return acc, f1


def discover_runs(root: str) -> List[Tuple[str, str, str]]:
    runs = []
    for condition in sorted(os.listdir(root)):
        cond_dir = os.path.join(root, condition)
        if not os.path.isdir(cond_dir):
            continue
        for seed_dir in sorted(os.listdir(cond_dir)):
            run_dir = os.path.join(cond_dir, seed_dir)
            metrics_path = os.path.join(run_dir, "metrics.json")
            ckpt_path = os.path.join(run_dir, "best_model.pt")
            if os.path.isfile(metrics_path) and os.path.isfile(ckpt_path):
                runs.append((condition, seed_dir, run_dir))
    return runs


def summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size > 0 else float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    test_normal_dir = os.path.join(args.test_dir, "NORMAL")
    test_pneumonia_dir = os.path.join(args.test_dir, "PNEUMONIA")

    tf = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_ds = BinaryFolderDataset(test_normal_dir, test_pneumonia_dir, transform=tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    runs = discover_runs(args.experiments_root)
    if len(runs) == 0:
        raise RuntimeError(f"No runs found under: {args.experiments_root}")

    rows = []
    for condition, seed_dir, run_dir in runs:
        with open(os.path.join(run_dir, "metrics.json"), "r") as f:
            meta = json.load(f)
        model_name = meta.get("model", args.model_fallback)
        train_mode = meta.get("train_mode", args.train_mode_fallback)

        model = build_model(model_name, train_mode).to(device)
        load_state_dict(model, os.path.join(run_dir, "best_model.pt"), device)
        y_true, y_prob = infer_probs(model, test_loader, device)
        acc, f1 = compute_acc_f1(y_true, y_prob, args.p)

        row = {
            "condition": condition,
            "seed": seed_dir,
            "p": float(args.p),
            "accuracy": acc,
            "f1": f1,
        }
        rows.append(row)
        print(f"{condition:30s} {seed_dir:10s} p={args.p:.3f} acc={acc:.4f} f1={f1:.4f}")

    by_condition: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        by_condition.setdefault(r["condition"], {"accuracy": [], "f1": []})
        by_condition[r["condition"]]["accuracy"].append(r["accuracy"])
        by_condition[r["condition"]]["f1"].append(r["f1"])

    summary = []
    print("\nCondition Summary (mean +- std)")
    for condition in sorted(by_condition.keys()):
        acc_mean, acc_std = summarize(by_condition[condition]["accuracy"])
        f1_mean, f1_std = summarize(by_condition[condition]["f1"])
        item = {
            "condition": condition,
            "n_runs": len(by_condition[condition]["accuracy"]),
            "p": float(args.p),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
        }
        summary.append(item)
        print(
            f"{condition:30s} n={item['n_runs']:2d} "
            f"acc={acc_mean:.4f}+-{acc_std:.4f} "
            f"f1={f1_mean:.4f}+-{f1_std:.4f}"
        )

    overall_acc_mean, overall_acc_std = summarize([r["accuracy"] for r in rows])
    overall_f1_mean, overall_f1_std = summarize([r["f1"] for r in rows])
    overall = {
        "p": float(args.p),
        "n_models": len(rows),
        "accuracy_mean": overall_acc_mean,
        "accuracy_std": overall_acc_std,
        "f1_mean": overall_f1_mean,
        "f1_std": overall_f1_std,
    }
    print("\nOverall (all models)")
    print(
        f"n={overall['n_models']} "
        f"acc={overall_acc_mean:.4f}+-{overall_acc_std:.4f} "
        f"f1={overall_f1_mean:.4f}+-{overall_f1_std:.4f}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    details_path = os.path.join(args.output_dir, "fixed_p_details.csv")
    summary_path = os.path.join(args.output_dir, "fixed_p_summary.json")

    with open(details_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "seed", "p", "accuracy", "f1"])
        w.writeheader()
        w.writerows(rows)

    with open(summary_path, "w") as f:
        json.dump({"per_condition": summary, "overall": overall}, f, indent=2)

    print(f"\nSaved details: {details_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_root", required=True, help="Root containing condition/seed_x directories")
    parser.add_argument("--test_dir", required=True, help="Path containing test/NORMAL and test/PNEUMONIA")
    parser.add_argument("--output_dir", default=".", help="Where to save fixed-p outputs")
    parser.add_argument("--p", type=float, default=0.53)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_fallback", default="resnet50")
    parser.add_argument("--train_mode_fallback", default="head")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
