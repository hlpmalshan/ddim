import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _gather_images(root: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        return []
    files = [os.path.join(root, f) for f in entries if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(exts)]
    files.sort()
    return files


class CelebAHQ(Dataset):
    """Simple image-folder dataset for CelebA-HQ.

    Expected structure (but flexible):
        root/
          train/  # images
          test/   # images (optional)
          val/    # images (optional)

    If split subfolder does not exist, falls back to using `root` directly.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__()
        split_root = os.path.join(root, split)
        if os.path.isdir(split_root):
            self.root = split_root
        else:
            # Fallback to flat folder of images
            self.root = root
        self.transform = transform
        self.files = _gather_images(self.root)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found for CelebA-HQ at: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # No labels for CelebA-HQ; return dummy target 0 for compatibility
        target = 0
        return img, target

