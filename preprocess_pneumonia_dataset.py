#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_dirs(path: str) -> List[str]:
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
    ])


def list_images(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMG_EXTS):
                files.append(os.path.join(dirpath, fname))
    files.sort()
    return files


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def pad_resize(img: Image.Image, size: int, pad_color: Tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    if img.mode == "L":
        background = Image.new("L", (size, size), color=pad_color[0])
    else:
        background = Image.new("RGB", (size, size), color=pad_color)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    background.paste(img, (left, top))
    return background


def stretch_resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BICUBIC)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def convert_mode(img: Image.Image, output_mode: str) -> Image.Image:
    if output_mode == "rgb":
        return img.convert("RGB")
    if output_mode == "grayscale":
        return img.convert("L")
    raise ValueError(f"Unsupported output_mode: {output_mode}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_root = os.path.join(script_dir, "Pneumonia_dataset", "chest_xray")
    default_output_root = os.path.join(script_dir, "Pneumonia_dataset", "chest_xray_preprocessed")
    default_report_dir = os.path.join(default_output_root, "_report")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default=default_input_root)
    parser.add_argument("--output_root", default=default_output_root)
    parser.add_argument("--report_dir", default=default_report_dir)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument(
        "--strategy",
        choices=["center_crop", "pad", "stretch"],
        default="center_crop",
    )
    parser.add_argument("--output_mode", choices=["rgb", "grayscale"], default="rgb")
    parser.add_argument("--pad_color", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--sample_count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    report_dir = os.path.abspath(args.report_dir)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Missing input_root: {input_root}")

    ensure_dir(report_dir)

    splits = list_dirs(input_root)
    if not splits:
        raise RuntimeError(f"No splits found in {input_root}")

    print("Input root :", input_root)
    print("Output root:", output_root)
    print("Report dir :", report_dir)
    print("Splits     :", ", ".join(splits))
    print("Target size:", args.target_size)
    print("Strategy   :", args.strategy)
    print("Output mode:", args.output_mode)

    # Track per-split/class stats
    stats = defaultdict(lambda: defaultdict(list))
    channel_stats = defaultdict(lambda: {"L": 0, "RGB": 0, "Other": 0})
    failures = []
    total_processed = 0

    samples = []
    if args.sample_count > 0:
        all_images = []
        for split in splits:
            split_dir = os.path.join(input_root, split)
            for cls in list_dirs(split_dir):
                all_images.extend(list_images(os.path.join(split_dir, cls)))
        random.shuffle(all_images)
        samples = all_images[: args.sample_count]

    csv_path = os.path.join(report_dir, "image_stats.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "class", "rel_path", "width", "height", "mode"])

        for split in splits:
            split_dir = os.path.join(input_root, split)
            class_dirs = list_dirs(split_dir)
            if not class_dirs:
                continue

            for cls in class_dirs:
                cls_dir = os.path.join(split_dir, cls)
                img_paths = list_images(cls_dir)
                if not img_paths:
                    continue

                for img_path in img_paths:
                    rel_path = os.path.relpath(img_path, input_root)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            mode = img.mode

                            writer.writerow([split, cls, rel_path, width, height, mode])
                            stats[(split, cls)]["width"].append(width)
                            stats[(split, cls)]["height"].append(height)
                            stats[(split, cls)]["ratio"].append(width / max(1.0, height))

                            if mode == "L":
                                channel_stats[(split, cls)]["L"] += 1
                            elif mode == "RGB":
                                channel_stats[(split, cls)]["RGB"] += 1
                            else:
                                channel_stats[(split, cls)]["Other"] += 1

                            if args.analyze_only:
                                continue

                            out_path = os.path.join(output_root, rel_path)
                            ensure_dir(os.path.dirname(out_path))

                            processed = convert_mode(img, args.output_mode)

                            if args.strategy == "center_crop":
                                processed = center_crop_resize(processed, args.target_size)
                            elif args.strategy == "pad":
                                processed = pad_resize(processed, args.target_size, tuple(args.pad_color))
                            else:
                                processed = stretch_resize(processed, args.target_size)

                            processed.save(out_path)
                            total_processed += 1

                            if img_path in samples:
                                sample_dir = os.path.join(report_dir, "samples")
                                ensure_dir(sample_dir)
                                sample_out = os.path.join(sample_dir, rel_path.replace(os.sep, "__"))
                                processed.save(sample_out)
                    except Exception as exc:
                        failures.append((rel_path, str(exc)))

    # Summaries
    summary = {"splits": {}, "overall": {}}
    overall_widths = []
    overall_heights = []
    overall_ratios = []
    overall_channels = {"L": 0, "RGB": 0, "Other": 0}

    for split in splits:
        split_dir = os.path.join(input_root, split)
        split_summary = {}
        for cls in list_dirs(split_dir):
            key = (split, cls)
            widths = stats[key]["width"]
            heights = stats[key]["height"]
            ratios = stats[key]["ratio"]
            ch = channel_stats[key]
            split_summary[cls] = {
                "count": len(widths),
                "width": compute_stats(widths),
                "height": compute_stats(heights),
                "aspect_ratio": compute_stats(ratios),
                "channels": ch,
            }
            overall_widths.extend(widths)
            overall_heights.extend(heights)
            overall_ratios.extend(ratios)
            overall_channels["L"] += ch["L"]
            overall_channels["RGB"] += ch["RGB"]
            overall_channels["Other"] += ch["Other"]
        summary["splits"][split] = split_summary

    summary["overall"] = {
        "count": len(overall_widths),
        "width": compute_stats(overall_widths),
        "height": compute_stats(overall_heights),
        "aspect_ratio": compute_stats(overall_ratios),
        "channels": overall_channels,
    }

    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(report_dir, "config.json"), "w") as f:
        json.dump(
            {
                "input_root": input_root,
                "output_root": output_root,
                "report_dir": report_dir,
                "target_size": args.target_size,
                "strategy": args.strategy,
                "output_mode": args.output_mode,
                "pad_color": args.pad_color,
                "sample_count": args.sample_count,
                "analyze_only": args.analyze_only,
                "seed": args.seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    if failures:
        fail_path = os.path.join(report_dir, "failures.txt")
        with open(fail_path, "w") as f:
            for rel_path, err in failures:
                f.write(f"{rel_path}\t{err}\n")
        print(f"Failures: {len(failures)} (see {fail_path})")

    if args.analyze_only:
        print("Analysis complete. No images were written (analyze_only enabled).")
    else:
        print(f"Processed images: {total_processed}")
        print("Preprocessing complete.")


if __name__ == "__main__":
    main()
