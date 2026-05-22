"""Shared training utilities for restoration models."""

from __future__ import annotations

from pathlib import Path
import argparse
import random
from typing import Any

import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
import yaml

from data.dataset import (
    DatasetParams,
    SequenceDatasetParams,
    TurbulencePairDataset,
    TurbulenceSequenceDataset,
    TurbulenceSequenceLmdbDataset,
)
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


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seed for reproducibility.

    Under DDP, callers should pass the per-process rank so that each rank's
    DataLoader workers and CPU augmentation streams get distinct seeds while
    remaining reproducible across runs.
    """
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def resolve_device(device_cfg: str, local_rank: int = 0) -> torch.device:
    """Resolve runtime device from config string.

    When `local_rank > 0` (DDP child process) and CUDA is available, returns
    `cuda:{local_rank}` so each rank pins its own GPU. Single-process callers
    can omit `local_rank` and behavior is identical to the original.
    """
    value = device_cfg.lower().strip()
    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = bool(mps_backend is not None and mps_backend.is_available())

    if value == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(local_rank)}")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")

    if value == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device(f"cuda:{int(local_rank)}")

    if value == "mps":
        if not has_mps:
            print("[WARN] MPS requested but not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    if value == "cpu":
        return torch.device("cpu")

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
        backend=str(dcfg.get("backend", "turbsim_gpu_v1")),
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
        cn2_range=(float(dcfg.get("cn2_range", [1e-16, 5e-14])[0]), float(dcfg.get("cn2_range", [1e-16, 5e-14])[1])),
        focal_length_range=(
            float(dcfg.get("focal_length_range", [35.0, 300.0])[0]),
            float(dcfg.get("focal_length_range", [35.0, 300.0])[1]),
        ),
        wind_speed_range=(
            float(dcfg.get("wind_speed_range", [1.0, 20.0])[0]),
            float(dcfg.get("wind_speed_range", [1.0, 20.0])[1]),
        ),
        sequence_time_step=float(dcfg.get("sequence_time_step", 0.03)),
        aperture_diameter=float(dcfg.get("aperture_diameter", 0.2)),
        wavelength=float(dcfg.get("wavelength", 0.525e-6)),
        object_size=float(dcfg.get("object_size", 2.06)),
        turbulence_strength=float(dcfg.get("turbulence_strength", 1.8)),
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
    data_mode = str(data_cfg.get("mode", "single")).lower().strip()

    train_root = Path(data_cfg["train_root"])
    val_root_str = str(data_cfg.get("val_root", "")).strip()

    if data_mode == "sequence":
        sequence_storage = str(data_cfg.get("sequence_storage", "folder")).lower().strip()
        train_ds_params = SequenceDatasetParams(
            image_size=int(data_cfg["image_size"]),
            num_frames=int(data_cfg.get("num_frames", 7)),
            random_crop=bool(data_cfg["random_crop"]),
            horizontal_flip_prob=float(data_cfg["horizontal_flip_prob"]),
        )
        val_ds_params = SequenceDatasetParams(
            image_size=int(data_cfg["image_size"]),
            num_frames=int(data_cfg.get("num_frames", 7)),
            random_crop=False,
            horizontal_flip_prob=0.0,
        )
        if sequence_storage == "lmdb":
            train_full = TurbulenceSequenceLmdbDataset(
                lmdb_root=train_root,
                dataset_params=train_ds_params,
                seed=seed,
            )
        else:
            train_full = TurbulenceSequenceDataset(
                root_dir=train_root,
                dataset_params=train_ds_params,
                seed=seed,
            )
    else:
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
        if data_mode == "sequence":
            sequence_storage = str(data_cfg.get("sequence_storage", "folder")).lower().strip()
            if sequence_storage == "lmdb":
                val_ds = TurbulenceSequenceLmdbDataset(
                    lmdb_root=val_root,
                    dataset_params=val_ds_params,
                    seed=seed + 1,
                )
            else:
                val_ds = TurbulenceSequenceDataset(
                    root_dir=val_root,
                    dataset_params=val_ds_params,
                    seed=seed + 1,
                )
        else:
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

    if data_mode == "sequence":
        sequence_storage = str(data_cfg.get("sequence_storage", "folder")).lower().strip()
        if sequence_storage == "lmdb":
            val_full = TurbulenceSequenceLmdbDataset(
                lmdb_root=train_root,
                dataset_params=val_ds_params,
                seed=seed + 1,
            )
        else:
            val_full = TurbulenceSequenceDataset(
                root_dir=train_root,
                dataset_params=val_ds_params,
                seed=seed + 1,
            )
    else:
        val_full = TurbulencePairDataset(
            root_dir=train_root,
            dataset_params=val_ds_params,
            turbulence_params=turbulence_params,
            seed=seed + 1,
        )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    return train_ds, val_ds


def build_dataloaders(
    cfg: dict[str, Any],
    seed: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train and validation dataloaders.

    When `distributed=True`, wraps both datasets with `DistributedSampler` and
    disables shuffle on the loader (the sampler handles shuffling). Callers
    must invoke `train_loader.sampler.set_epoch(epoch)` at the start of each
    epoch to ensure proper cross-rank shuffling.

    Note: `batch_size` in config is interpreted as the per-rank batch size.
    Effective global batch = batch_size * world_size.
    """
    data_cfg = cfg["data"]
    train_ds, val_ds = build_datasets(cfg=cfg, seed=seed)
    num_workers = int(data_cfg["num_workers"])
    pin_memory = bool(data_cfg["pin_memory"])

    loader_extra_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_extra_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_extra_kwargs["prefetch_factor"] = max(1, int(data_cfg.get("prefetch_factor", 2)))

    if distributed:
        train_sampler: DistributedSampler[Any] | None = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(seed),
            drop_last=True,
        )
        val_sampler: DistributedSampler[Any] | None = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **loader_extra_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("val_batch_size", data_cfg["batch_size"])),
        shuffle=False,
        sampler=val_sampler,
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


def resolve_cond_channels(cfg: dict[str, Any]) -> int:
    """Resolve degraded-input channels used by model condition branch."""
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    base_in = int(model_cfg.get("in_channels", 3))

    data_mode = str(data_cfg.get("mode", "single")).lower().strip()
    if data_mode != "sequence":
        return base_in

    strategy = str(data_cfg.get("sequence_input", "stack_channels")).lower().strip()
    num_frames = max(1, int(data_cfg.get("num_frames", 1)))
    if strategy == "stack_channels":
        return base_in * num_frames
    return base_in


def adapt_degraded_for_model(degraded: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Adapt degraded input to 4D tensors expected by current model backbones."""
    if degraded.ndim != 5:
        return degraded

    data_cfg = cfg.get("data", {})
    strategy = str(data_cfg.get("sequence_input", "stack_channels")).lower().strip()
    if strategy == "stack_channels":
        bsz, frames, channels, height, width = degraded.shape
        return degraded.reshape(bsz, frames * channels, height, width)
    if strategy == "mean":
        return degraded.mean(dim=1)
    if strategy == "center_frame":
        center_idx = degraded.shape[1] // 2
        return degraded[:, center_idx, ...]

    raise ValueError(f"Unsupported data.sequence_input strategy: {strategy}")


# ---------------------------------------------------------------------------
# Distributed (DDP) utilities
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def is_dist_available_and_initialized() -> bool:
    """Return True if torch.distributed is available and a process group is up."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank; 0 when distributed is not active."""
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """World size; 1 when distributed is not active."""
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """True for rank 0 (and for single-process runs)."""
    return get_rank() == 0


def init_distributed_mode() -> tuple[bool, int, int, int]:
    """Initialise torch.distributed if launched via torchrun.

    Detects ``LOCAL_RANK``/``RANK``/``WORLD_SIZE`` environment variables. If
    present and ``WORLD_SIZE > 1``, initialises the NCCL process group and
    pins the current CUDA device to ``local_rank``.

    Returns:
        (distributed, rank, world_size, local_rank)
        - For single-process runs returns (False, 0, 1, 0).
    """
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1 or not dist.is_available():
        return False, 0, 1, 0

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP launch detected (WORLD_SIZE>1) but CUDA is not available. "
            "DDP currently requires GPUs in this project."
        )

    backend = "nccl"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Tear down the process group if it was initialised. Safe to call always."""
    try:
        if is_dist_available_and_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        print(f"[WARN] cleanup_distributed encountered an error: {exc}")


def reduce_dict(stats: dict[str, float], world_size: int | None = None) -> dict[str, float]:
    """Average a dict of scalar metrics across ranks via all_reduce.

    No-op when distributed is not initialised. Keys must be identical on all
    ranks (callers should ensure stable ordering).
    """
    if not is_dist_available_and_initialized():
        return dict(stats)
    if world_size is None:
        world_size = get_world_size()
    if world_size <= 1 or not stats:
        return dict(stats)

    keys = sorted(stats.keys())
    values = torch.tensor(
        [float(stats[k]) for k in keys],
        dtype=torch.float64,
        device=torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu"),
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(world_size)
    return {k: float(values[i].item()) for i, k in enumerate(keys)}


def reduce_sum_count(
    sum_stats: dict[str, float],
    count: int,
) -> tuple[dict[str, float], int]:
    """Reduce per-rank running sums and a sample count via all_reduce(SUM).

    Used by validation loops where each rank processes a disjoint shard of the
    val set; we need the global sum and the global number of batches/samples
    to compute a true mean.
    """
    if not is_dist_available_and_initialized():
        return dict(sum_stats), int(count)

    keys = sorted(sum_stats.keys())
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    if keys:
        values = torch.tensor([float(sum_stats[k]) for k in keys], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        reduced = {k: float(values[i].item()) for i, k in enumerate(keys)}
    else:
        reduced = {}

    count_t = torch.tensor([int(count)], dtype=torch.long, device=device)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    return reduced, int(count_t.item())
