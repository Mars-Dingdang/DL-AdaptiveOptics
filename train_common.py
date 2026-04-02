"""Shared training utilities for restoration models."""

from __future__ import annotations

from pathlib import Path
import argparse
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from data.dataset import DatasetParams, TurbulencePairDataset
from utils.degradation import TurbulenceParams


def parse_train_args(description: str) -> argparse.Namespace:
    """Parse common CLI args for training entrypoints."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config into dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a dictionary")
    return cfg


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    """Resolve runtime device from config string."""
    value = device_cfg.lower().strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value in {"cuda", "cpu"}:
        if value == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device(value)
    raise ValueError(f"Unsupported device config: {device_cfg}")


def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Convert [0,1] tensor to [-1,1]."""
    return x * 2.0 - 1.0


def to_0_1(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] tensor to [0,1]."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def build_turbulence_params(cfg: dict[str, Any]) -> TurbulenceParams:
    """Build turbulence parameters from config."""
    dcfg = cfg["degradation"]
    return TurbulenceParams(
        zernike_order=int(dcfg["zernike_order"]),
        phase_strength=float(dcfg["phase_strength"]),
        psf_kernel_size=int(dcfg["psf_kernel_size"]),
        gaussian_sigma_range=(float(dcfg["gaussian_sigma_range"][0]), float(dcfg["gaussian_sigma_range"][1])),
        motion_blur_prob=float(dcfg["motion_blur_prob"]),
        motion_blur_kernel_range=(int(dcfg["motion_blur_kernel_range"][0]), int(dcfg["motion_blur_kernel_range"][1])),
        poisson_scale_range=(float(dcfg["poisson_scale_range"][0]), float(dcfg["poisson_scale_range"][1])),
        gaussian_noise_std_range=(
            float(dcfg["gaussian_noise_std_range"][0]),
            float(dcfg["gaussian_noise_std_range"][1]),
        ),
        jpeg_quality_range=(int(dcfg["jpeg_quality_range"][0]), int(dcfg["jpeg_quality_range"][1])),
    )


def _resolve_path(value: str | Path) -> Path:
    """Normalize a path string for robust equality checks."""
    return Path(value).expanduser().resolve()


def validate_data_protocol(cfg: dict[str, Any]) -> None:
    """Validate and warn about train/val/test split configuration."""
    data_cfg = cfg.get("data", {})
    train_root_str = str(data_cfg.get("train_root", "")).strip()
    val_root_str = str(data_cfg.get("val_root", "")).strip()
    test_root_str = str(data_cfg.get("test_root", "")).strip()

    if not train_root_str:
        raise RuntimeError("data.train_root must be set.")

    train_root = _resolve_path(train_root_str)
    val_root = _resolve_path(val_root_str) if val_root_str else None
    test_root = _resolve_path(test_root_str) if test_root_str else None

    if val_root is not None and val_root == train_root:
        print("[WARN] data.val_root points to the same location as data.train_root.")
        print("[WARN] This can cause validation leakage during model selection.")

    if test_root is not None and test_root == train_root:
        print("[WARN] data.test_root points to the same location as data.train_root.")
        print("[WARN] Final test metrics will be biased if test data is seen in training.")

    if test_root is not None and val_root is not None and test_root == val_root:
        print("[WARN] data.test_root points to the same location as data.val_root.")
        print("[WARN] Keep validation for tuning and reserve test for final one-time reporting.")

    if test_root is None:
        print("[WARN] data.test_root is empty. Configure a held-out test set for final reporting.")


def build_datasets(cfg: dict[str, Any], seed: int) -> tuple[Dataset[Any], Dataset[Any]]:
    """Create train/val datasets from config.

    If val_root is empty, split train_root by val_ratio.
    """
    data_cfg = cfg["data"]
    turbulence_params = build_turbulence_params(cfg)

    train_root = Path(data_cfg["train_root"])
    val_root_str = str(data_cfg.get("val_root", "")).strip()

    train_ds_params = DatasetParams(
        image_size=int(data_cfg["image_size"]),
        random_crop=bool(data_cfg["random_crop"]),
        horizontal_flip_prob=float(data_cfg["horizontal_flip_prob"]),
    )
    val_ds_params = DatasetParams(
        image_size=int(data_cfg["image_size"]),
        random_crop=False,
        horizontal_flip_prob=0.0,
    )

    train_full = TurbulencePairDataset(
        root_dir=train_root,
        dataset_params=train_ds_params,
        turbulence_params=turbulence_params,
        seed=seed,
    )

    if val_root_str:
        val_root = Path(val_root_str)
        val_ds = TurbulencePairDataset(
            root_dir=val_root,
            dataset_params=val_ds_params,
            turbulence_params=turbulence_params,
            seed=seed + 1,
        )
        return train_full, val_ds

    n_total = len(train_full)
    if n_total < 2:
        raise RuntimeError("Need at least 2 images for train/val split.")

    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    val_ratio = min(max(val_ratio, 0.01), 0.5)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = max(1, n_total - n_val)
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    indices = np.arange(n_total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val : n_val + n_train].tolist()

    val_full = TurbulencePairDataset(
        root_dir=train_root,
        dataset_params=val_ds_params,
        turbulence_params=turbulence_params,
        seed=seed + 1,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    return train_ds, val_ds


def build_dataloaders(cfg: dict[str, Any], seed: int) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train and validation dataloaders."""
    data_cfg = cfg["data"]
    train_ds, val_ds = build_datasets(cfg=cfg, seed=seed)
    num_workers = int(data_cfg["num_workers"])
    pin_memory = bool(data_cfg["pin_memory"])

    loader_extra_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_extra_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_extra_kwargs["prefetch_factor"] = max(1, int(data_cfg.get("prefetch_factor", 2)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **loader_extra_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("val_batch_size", data_cfg["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **loader_extra_kwargs,
    )
    return train_loader, val_loader


def compute_mean_stats(sum_stats: dict[str, float], count: int | dict[str, int]) -> dict[str, float]:
    """Compute mean stats from accumulated sums."""
    if isinstance(count, int):
        if count <= 0:
            return {k: 0.0 for k in sum_stats}
        return {k: v / float(count) for k, v in sum_stats.items()}

    out: dict[str, float] = {}
    for k, v in sum_stats.items():
        denom = int(count.get(k, 0))
        out[k] = (v / float(denom)) if denom > 0 else 0.0
    return out


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Save checkpoint dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
