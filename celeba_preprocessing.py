#!/usr/bin/env python3
import os, glob, argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as F

# ---- EXACT training-style CelebA 128x128 crop via F.crop ----
# Training used: F.crop(img, x1, y1, x2-x1, y2-y1) with:
# cx=89, cy=121; x1=cy-64, x2=cy+64, y1=cx-64, y2=cx+64
class CropCelebA128_Exact:
    def __init__(self, cx=89, cy=121):
        self.x1 = cy - 64  # treated as 'top' by F.crop
        self.x2 = cy + 64
        self.y1 = cx - 64  # treated as 'left' by F.crop
        self.y2 = cx + 64
        self.h = self.x2 - self.x1
        self.w = self.y2 - self.y1
        if self.h != 128 or self.w != 128:
            raise ValueError("Expected 128x128 crop window.")

    def __call__(self, img: Image.Image) -> Image.Image:
        # NOTE: This matches training exactly; no boundary clamping.
        return F.crop(img, self.x1, self.y1, self.h, self.w)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*"))
                  if os.path.splitext(p)[1].lower() in IMG_EXTS)

def transform_and_save_real(src_dir, dst_dir, out_size=64, overwrite=False):
    os.makedirs(dst_dir, exist_ok=True)
    crop128 = CropCelebA128_Exact()
    post = transforms.Compose([transforms.Resize(out_size), transforms.CenterCrop(out_size)])

    srcs = list_images(src_dir)
    if not srcs:
        raise RuntimeError(f"No images found in {src_dir}")

    written = 0
    for f in tqdm(srcs, desc="REAL: exact F.crop(128) → resize64 → save", leave=False):
        try:
            img = Image.open(f).convert("RGB")
            img = crop128(img)  # exact training crop
            img = post(img)

            stem = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(dst_dir, f"{stem}_64.png")
            if (not overwrite) and os.path.exists(out_path):
                continue
            img.save(out_path)  # PNG
            written += 1
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")

    print(f"[Done] Wrote {written} images to {dst_dir}")

def main():
    ap = argparse.ArgumentParser("Transform CelebA REAL images: exact training F.crop 128×128, then resize 64×64.")
    ap.add_argument("--src_dir", required=True, help="Folder with original CelebA images")
    ap.add_argument("--dst_dir", required=True, help="Output folder for processed images")
    ap.add_argument("--size", type=int, default=64, help="Final size (default 64)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    transform_and_save_real(args.src_dir, args.dst_dir, out_size=args.size, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
