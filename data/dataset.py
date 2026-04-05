"""Dataset and DataLoader definitions for turbulence restoration training.

This dataset reads clear remote sensing images and applies on-the-fly atmospheric
degradation to form paired samples: (degraded_img, clear_img).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

import cv2
import lmdb
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


@dataclass
class SequenceDatasetParams:
    """Configuration for sequence dataset preprocessing and augmentation."""

    image_size: int = 256
    num_frames: int = 7
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


def _decode_rgb_image_from_png_bytes(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode PNG bytes from LMDB record.")
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


def _to_tensor_sequence(images: np.ndarray) -> torch.Tensor:
    """Convert TxHxWxC float images in [0,1] to TxCxHxW tensor."""
    if images.ndim != 4:
        raise ValueError("Expected TxHxWxC sequence")
    tensor = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float()
    return tensor


def _scan_sequence_sample_dirs(root_dir: Path) -> list[Path]:
    """Scan sample directories containing clean.png and frame_*.png files."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Sequence dataset root does not exist: {root_dir}")

    sample_dirs: list[Path] = []
    for path in sorted(root_dir.iterdir()):
        if not path.is_dir():
            continue
        clean_path = path / "clean.png"
        if not clean_path.exists():
            continue
        frame_files = sorted(path.glob("frame_*.png"))
        if not frame_files:
            continue
        sample_dirs.append(path)

    if not sample_dirs:
        raise RuntimeError(f"No valid sequence samples found under {root_dir}")
    return sample_dirs


def _crop_image_with_coords(image: np.ndarray, top: int, left: int, crop_size: int) -> np.ndarray:
    """Crop image with predefined coordinates."""
    return image[top : top + crop_size, left : left + crop_size, :]


def _sample_crop_coords(
    image: np.ndarray,
    crop_size: int,
    random_crop: bool,
    rng: np.random.Generator,
) -> tuple[int, int, np.ndarray]:
    """Sample crop coordinates and return resized source image."""
    src = _resize_if_needed(image, target_size=crop_size)
    h, w = src.shape[:2]
    if random_crop:
        top = int(rng.integers(0, h - crop_size + 1))
        left = int(rng.integers(0, w - crop_size + 1))
    else:
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
    return top, left, src


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


class TurbulenceSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset reading pre-generated sequence samples from sample directories."""

    def __init__(
        self,
        root_dir: str | Path,
        dataset_params: SequenceDatasetParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_params = dataset_params or SequenceDatasetParams()
        self.sample_dirs = _scan_sequence_sample_dirs(self.root_dir)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[index]
        clean_path = sample_dir / "clean.png"

        frame_paths = sorted(sample_dir.glob("frame_*.png"))
        if len(frame_paths) < int(self.dataset_params.num_frames):
            raise RuntimeError(
                f"Sample {sample_dir} has {len(frame_paths)} frames but requires {self.dataset_params.num_frames}."
            )

        frame_paths = frame_paths[: int(self.dataset_params.num_frames)]
        clear_img = _read_rgb_image(clean_path)

        top, left, clear_img = _sample_crop_coords(
            image=clear_img,
            crop_size=int(self.dataset_params.image_size),
            random_crop=bool(self.dataset_params.random_crop),
            rng=self.rng,
        )
        clear_img = _crop_image_with_coords(
            image=clear_img,
            top=top,
            left=left,
            crop_size=int(self.dataset_params.image_size),
        )

        seq_frames: list[np.ndarray] = []
        for frame_path in frame_paths:
            frame = _read_rgb_image(frame_path)
            frame = _resize_if_needed(frame, target_size=int(self.dataset_params.image_size))
            frame = _crop_image_with_coords(
                image=frame,
                top=top,
                left=left,
                crop_size=int(self.dataset_params.image_size),
            )
            seq_frames.append(frame)

        if float(self.dataset_params.horizontal_flip_prob) > 0.0:
            if float(self.rng.uniform(0.0, 1.0)) < float(self.dataset_params.horizontal_flip_prob):
                clear_img = clear_img[:, ::-1, :]
                seq_frames = [frame[:, ::-1, :] for frame in seq_frames]

        seq_arr = np.stack(seq_frames, axis=0).astype(np.float32)
        clear_tensor = _to_tensor(clear_img)
        seq_tensor = _to_tensor_sequence(seq_arr)

        meta_path = sample_dir / "meta.json"
        if meta_path.exists():
            try:
                _ = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return seq_tensor, clear_tensor


class TurbulenceSequenceLmdbDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset reading pre-generated sequence samples from LMDB."""

    def __init__(
        self,
        lmdb_root: str | Path,
        dataset_params: SequenceDatasetParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.lmdb_root = Path(lmdb_root)
        self.dataset_params = dataset_params or SequenceDatasetParams()
        self.rng = np.random.default_rng(seed)
        self._env: lmdb.Environment | None = None
        self._length: int | None = None

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            if not self.lmdb_root.exists():
                raise FileNotFoundError(f"LMDB root does not exist: {self.lmdb_root}")
            self._env = lmdb.open(
                str(self.lmdb_root),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
            )
        return self._env

    def _read_lmdb_bytes(self, key: str) -> bytes:
        env = self._get_env()
        with env.begin(write=False) as txn:
            raw = txn.get(key.encode("utf-8"))
        if raw is None:
            raise KeyError(f"Missing LMDB key: {key}")
        return bytes(raw)

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        raw = self._read_lmdb_bytes("__len__")
        self._length = int(raw.decode("utf-8"))
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_key = f"sample-{index:07d}"
        clear_raw = self._read_lmdb_bytes(f"{sample_key}-clean")
        clear_img = _decode_rgb_image_from_png_bytes(clear_raw)

        top, left, clear_img = _sample_crop_coords(
            image=clear_img,
            crop_size=int(self.dataset_params.image_size),
            random_crop=bool(self.dataset_params.random_crop),
            rng=self.rng,
        )
        clear_img = _crop_image_with_coords(
            image=clear_img,
            top=top,
            left=left,
            crop_size=int(self.dataset_params.image_size),
        )

        seq_frames: list[np.ndarray] = []
        for frame_idx in range(int(self.dataset_params.num_frames)):
            frame_raw = self._read_lmdb_bytes(f"{sample_key}-frame-{frame_idx:03d}")
            frame = _decode_rgb_image_from_png_bytes(frame_raw)
            frame = _resize_if_needed(frame, target_size=int(self.dataset_params.image_size))
            frame = _crop_image_with_coords(
                image=frame,
                top=top,
                left=left,
                crop_size=int(self.dataset_params.image_size),
            )
            seq_frames.append(frame)

        if float(self.dataset_params.horizontal_flip_prob) > 0.0:
            if float(self.rng.uniform(0.0, 1.0)) < float(self.dataset_params.horizontal_flip_prob):
                clear_img = clear_img[:, ::-1, :]
                seq_frames = [frame[:, ::-1, :] for frame in seq_frames]

        seq_arr = np.stack(seq_frames, axis=0).astype(np.float32)
        clear_tensor = _to_tensor(clear_img)
        seq_tensor = _to_tensor_sequence(seq_arr)
        return seq_tensor, clear_tensor

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_env"] = None
        return state


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
