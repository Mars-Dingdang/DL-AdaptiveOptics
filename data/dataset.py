"""Dataset and DataLoader definitions for turbulence restoration training.

This dataset reads clear remote sensing images and applies on-the-fly atmospheric
degradation to form paired samples: (degraded_img, clear_img).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.degradation import TurbulenceParams, add_atmospheric_turbulence


IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class DatasetParams:
    """Configuration for dataset preprocessing and augmentation."""

    image_size: int = 256
    random_crop: bool = True
    horizontal_flip_prob: float = 0.5


def _scan_image_files(root_dir: Path, extensions: Sequence[str]) -> list[Path]:
    """Recursively scan image files from root directory."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    extension_set = {ext.lower() for ext in extensions}
    files = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in extension_set]
    files.sort()

    if not files:
        raise RuntimeError(f"No image files found under {root_dir} with extensions: {sorted(extension_set)}")

    return files


def _read_rgb_image(path: Path) -> np.ndarray:
    """Read image from disk and convert to RGB float [0, 1]."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_f = image_rgb.astype(np.float32) / 255.0
    return np.clip(image_f, 0.0, 1.0)


def _resize_if_needed(image: np.ndarray, target_size: int) -> np.ndarray:
    """Upscale image if its shorter side is smaller than target crop size."""
    h, w = image.shape[:2]
    short_side = min(h, w)
    if short_side >= target_size:
        return image

    scale = float(target_size) / float(short_side)
    new_w = max(target_size, int(round(w * scale)))
    new_h = max(target_size, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized


def _crop_image(
    image: np.ndarray,
    crop_size: int,
    random_crop: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Crop image to crop_size x crop_size."""
    h, w = image.shape[:2]

    if h == crop_size and w == crop_size:
        return image

    if h < crop_size or w < crop_size:
        image = _resize_if_needed(image, target_size=crop_size)
        h, w = image.shape[:2]

    if random_crop:
        top = int(rng.integers(0, h - crop_size + 1))
        left = int(rng.integers(0, w - crop_size + 1))
    else:
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)

    return image[top : top + crop_size, left : left + crop_size, :]


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HxWxC float image in [0,1] to CHW torch float tensor."""
    if image.ndim != 3:
        raise ValueError("Expected HxWxC image")
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    return tensor


class TurbulencePairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset generating (degraded, clear) image pairs online."""

    def __init__(
        self,
        root_dir: str | Path,
        dataset_params: DatasetParams | None = None,
        turbulence_params: TurbulenceParams | None = None,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
        seed: int | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root_dir: Directory containing clear images.
            dataset_params: Preprocessing and augmentation config.
            turbulence_params: Physical degradation config.
            extensions: Allowed image suffixes.
            seed: Optional RNG seed.
        """
        self.root_dir = Path(root_dir)
        self.dataset_params = dataset_params or DatasetParams()
        self.turbulence_params = turbulence_params or TurbulenceParams()
        self.extensions = tuple(extensions)
        self.files = _scan_image_files(root_dir=self.root_dir, extensions=self.extensions)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (degraded, clear) pair for training."""
        path = self.files[index]
        clear_img = _read_rgb_image(path)

        clear_img = _resize_if_needed(clear_img, target_size=self.dataset_params.image_size)
        clear_img = _crop_image(
            image=clear_img,
            crop_size=self.dataset_params.image_size,
            random_crop=self.dataset_params.random_crop,
            rng=self.rng,
        )

        if float(self.dataset_params.horizontal_flip_prob) > 0.0:
            if float(self.rng.uniform(0.0, 1.0)) < float(self.dataset_params.horizontal_flip_prob):
                clear_img = clear_img[:, ::-1, :]

        degraded_img = add_atmospheric_turbulence(
            image=clear_img,
            params=self.turbulence_params,
            rng=self.rng,
            return_meta=False,
        )

        degraded_tensor = _to_tensor(degraded_img)
        clear_tensor = _to_tensor(clear_img)
        return degraded_tensor, clear_tensor


def build_dataloader(
    root_dir: str | Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    dataset_params: DatasetParams | None = None,
    turbulence_params: TurbulenceParams | None = None,
    seed: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build DataLoader for turbulence restoration training."""
    dataset = TurbulencePairDataset(
        root_dir=root_dir,
        dataset_params=dataset_params,
        turbulence_params=turbulence_params,
        seed=seed,
    )
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    return loader
