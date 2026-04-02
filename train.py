"""Unified training script for turbulence restoration.

Supported model types:
- unet: baseline supervised restoration with L1 objective.
- gan: Pix2Pix-style conditional GAN with optional physics consistency.

Usage:
    python train.py --config configs/default.yaml
    
20260402 1507
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
import yaml

from data.dataset import DatasetParams, TurbulencePairDataset
from modules.baseline_unet import build_baseline_unet
from modules.gan_models import (
    GANLossWeights,
    build_pix2pix_models,
    discriminator_loss,
    generator_total_loss,
)
from modules.diffusion import ConditionalDiffusionModel, DiffusionConfig
from utils.degradation import TurbulenceParams
from utils.metrics import RestorationMetrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train restoration models for atmospheric turbulence removal.")
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


def _mean_stats(sum_stats: dict[str, float], count: int | dict[str, int]) -> dict[str, float]:
    """Convert sum stats to mean stats."""
    if isinstance(count, int):
        if count <= 0:
            return {k: 0.0 for k in sum_stats}
        return {k: v / float(count) for k, v in sum_stats.items()}

    out: dict[str, float] = {}
    for k, v in sum_stats.items():
        denom = int(count.get(k, 0))
        out[k] = (v / float(denom)) if denom > 0 else 0.0
    return out


def _safe_log(message: str, use_tqdm: bool) -> None:
    """Log message without breaking active tqdm bars."""
    if use_tqdm:
        tqdm.write(message)
    else:
        print(message)


def train_one_epoch_unet(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    metrics: RestorationMetrics,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_grad_norm: float,
    amp_enabled: bool,
    scaler: GradScaler | None,
    metric_interval: int,
    metric_on_log: bool,
    use_tqdm: bool,
) -> dict[str, float]:
    """Train U-Net for one epoch."""
    model.train()
    use_amp = amp_enabled and device.type == "cuda"

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(
        loader,
        total=len(loader),
        disable=not use_tqdm,
        desc=f"Train UNet e{epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar, start=1):
        degraded, clear = batch
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        with autocast(enabled=use_amp):
            pred = model(degraded)
            pred_01 = pred.clamp(0.0, 1.0)

            loss = criterion(pred_01, clear)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        batch_metrics: dict[str, float] | None = None
        if compute_metrics_now:
            batch_metrics = metrics.compute_batch(pred=pred_01.detach(), target=clear.detach())

        sum_stats["loss"] += float(loss.detach().item())
        sum_counts["loss"] += 1
        if batch_metrics is not None:
            for k, v in batch_metrics.items():
                sum_stats[k] += float(v)
                sum_counts[k] += 1

        postfix: dict[str, str] = {
            "loss": f"{loss.detach().item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        if batch_metrics is not None:
            if "psnr" in batch_metrics:
                postfix["psnr"] = f"{batch_metrics['psnr']:.2f}"
            if "ssim" in batch_metrics:
                postfix["ssim"] = f"{batch_metrics['ssim']:.3f}"
        pbar.set_postfix(postfix)

        if should_log:
            log_msg = f"[Train][UNet] epoch={epoch} step={step}/{len(loader)} loss={loss.detach().item():.4f}"
            if batch_metrics is not None:
                if "psnr" in batch_metrics and "ssim" in batch_metrics:
                    log_msg += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
                if "lpips" in batch_metrics:
                    log_msg += f" lpips={batch_metrics['lpips']:.4f}"
            _safe_log(log_msg, use_tqdm=use_tqdm)

    return _mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_unet(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    metrics: RestorationMetrics,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate U-Net model."""
    model.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        pred = model(degraded)
        pred_01 = pred.clamp(0.0, 1.0)

        loss = criterion(pred_01, clear)
        batch_metrics = metrics.compute_batch(pred=pred_01, target=clear)

        sum_stats["loss"] += float(loss.detach().item())
        for k, v in batch_metrics.items():
            sum_stats[k] += float(v)
        n_batches += 1

    return _mean_stats(sum_stats, n_batches)


def train_one_epoch_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader[Any],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    gan_weights: GANLossWeights,
    metrics: RestorationMetrics,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_grad_norm: float,
    amp_enabled: bool,
    scaler: GradScaler | None,
    metric_interval: int,
    metric_on_log: bool,
    use_tqdm: bool,
) -> dict[str, float]:
    """Train conditional GAN for one epoch."""
    generator.train()
    discriminator.train()
    use_amp = amp_enabled and device.type == "cuda"

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(
        loader,
        total=len(loader),
        disable=not use_tqdm,
        desc=f"Train GAN e{epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar, start=1):
        degraded, clear = batch
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        degraded_n = to_minus1_1(degraded)
        clear_n = to_minus1_1(clear)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        # Update discriminator.
        with autocast(enabled=use_amp):
            with torch.no_grad():
                fake_clear_detached = generator(degraded_n)

            pred_real = discriminator(degraded_n, clear_n)
            pred_fake = discriminator(degraded_n, fake_clear_detached)
            loss_d = discriminator_loss(pred_real_logits=pred_real, pred_fake_logits=pred_fake)

        optimizer_d.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(loss_d).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer_d)
        else:
            loss_d.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
            optimizer_d.step()

        # Update generator.
        with autocast(enabled=use_amp):
            fake_clear = generator(degraded_n)
            pred_fake_for_g = discriminator(degraded_n, fake_clear)

            loss_g_total, g_stats = generator_total_loss(
                fake_clear=fake_clear,
                real_clear=clear_n,
                degraded_input=degraded_n,
                pred_fake_logits=pred_fake_for_g,
                weights=gan_weights,
            )

        optimizer_g.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(loss_g_total).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer_g)
            scaler.update()
        else:
            loss_g_total.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
            optimizer_g.step()

        batch_metrics: dict[str, float] | None = None
        if compute_metrics_now:
            fake_01 = to_0_1(fake_clear.detach())
            batch_metrics = metrics.compute_batch(pred=fake_01, target=clear.detach())

        sum_stats["d_loss"] += float(loss_d.detach().item())
        sum_counts["d_loss"] += 1
        sum_stats["g_total"] += float(loss_g_total.detach().item())
        sum_counts["g_total"] += 1
        sum_stats["g_adv"] += float(g_stats["g_adv"])
        sum_counts["g_adv"] += 1
        sum_stats["g_l1"] += float(g_stats["g_l1"])
        sum_counts["g_l1"] += 1
        sum_stats["g_phy"] += float(g_stats["g_phy"])
        sum_counts["g_phy"] += 1
        if batch_metrics is not None:
            for k, v in batch_metrics.items():
                sum_stats[k] += float(v)
                sum_counts[k] += 1

        postfix: dict[str, str] = {
            "d": f"{loss_d.detach().item():.4f}",
            "g": f"{loss_g_total.detach().item():.4f}",
            "g_adv": f"{g_stats['g_adv']:.3f}",
            "g_l1": f"{g_stats['g_l1']:.3f}",
            "g_phy": f"{g_stats['g_phy']:.3f}",
            "lr_g": f"{optimizer_g.param_groups[0]['lr']:.2e}",
        }
        if batch_metrics is not None:
            if "psnr" in batch_metrics:
                postfix["psnr"] = f"{batch_metrics['psnr']:.2f}"
            if "ssim" in batch_metrics:
                postfix["ssim"] = f"{batch_metrics['ssim']:.3f}"
        pbar.set_postfix(postfix)

        if should_log:
            log_msg = (
                f"[Train][GAN] epoch={epoch} step={step}/{len(loader)} "
                f"d={loss_d.detach().item():.4f} g={loss_g_total.detach().item():.4f}"
            )
            if batch_metrics is not None:
                if "psnr" in batch_metrics and "ssim" in batch_metrics:
                    log_msg += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
                if "lpips" in batch_metrics:
                    log_msg += f" lpips={batch_metrics['lpips']:.4f}"
            _safe_log(log_msg, use_tqdm=use_tqdm)

    return _mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader[Any],
    gan_weights: GANLossWeights,
    metrics: RestorationMetrics,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate GAN generator/discriminator on validation set."""
    generator.eval()
    discriminator.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        degraded_n = to_minus1_1(degraded)
        clear_n = to_minus1_1(clear)

        fake_clear = generator(degraded_n)
        pred_real = discriminator(degraded_n, clear_n)
        pred_fake = discriminator(degraded_n, fake_clear)

        d_loss = discriminator_loss(pred_real_logits=pred_real, pred_fake_logits=pred_fake)
        g_total, g_stats = generator_total_loss(
            fake_clear=fake_clear,
            real_clear=clear_n,
            degraded_input=degraded_n,
            pred_fake_logits=pred_fake,
            weights=gan_weights,
        )

        fake_01 = to_0_1(fake_clear)
        batch_metrics = metrics.compute_batch(pred=fake_01, target=clear)

        sum_stats["d_loss"] += float(d_loss.detach().item())
        sum_stats["g_total"] += float(g_total.detach().item())
        sum_stats["g_adv"] += float(g_stats["g_adv"])
        sum_stats["g_l1"] += float(g_stats["g_l1"])
        sum_stats["g_phy"] += float(g_stats["g_phy"])
        for k, v in batch_metrics.items():
            sum_stats[k] += float(v)
        n_batches += 1

    return _mean_stats(sum_stats, n_batches)


def train_one_epoch_diffusion(
    model: ConditionalDiffusionModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    noise_weight: float,
    metrics: RestorationMetrics,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_grad_norm: float,
    amp_enabled: bool,
    scaler: GradScaler | None,
    metric_interval: int,
    metric_on_log: bool,
    ddim_steps: int,
    use_tqdm: bool,
) -> dict[str, float]:
    """Train diffusion model for one epoch."""
    model.train()
    use_amp = amp_enabled and device.type == "cuda"

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(
        loader,
        total=len(loader),
        disable=not use_tqdm,
        desc=f"Train Diff e{epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar, start=1):
        degraded, clear = batch
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        # Convert to [-1, 1] range for diffusion model
        degraded_n = to_minus1_1(degraded)
        clear_n = to_minus1_1(clear)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        # Sample random timesteps
        batch_size = clear.shape[0]
        t = torch.randint(0, model.config.timesteps, (batch_size,), device=device)

        with autocast(enabled=use_amp):
            # Compute noise prediction loss
            loss = model.p_losses(x_start=clear_n, cond=degraded_n, t=t)
            loss = loss * noise_weight

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        batch_metrics: dict[str, float] | None = None
        if compute_metrics_now:
            # Generate sample for metrics (expensive, so only when logging)
            with torch.no_grad():
                pbar.set_postfix({"loss": f"{loss.detach().item():.4f}", "stage": "ddim_eval"})
                model.eval()
                pred_n = model.sample_ddim(cond=degraded_n, steps=ddim_steps)
                model.train()
                pred_01 = to_0_1(pred_n)
                batch_metrics = metrics.compute_batch(pred=pred_01, target=clear)

        sum_stats["loss"] += float(loss.detach().item())
        sum_counts["loss"] += 1
        if batch_metrics is not None:
            for k, v in batch_metrics.items():
                sum_stats[k] += float(v)
                sum_counts[k] += 1

        postfix: dict[str, str] = {
            "loss": f"{loss.detach().item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "stage": "train",
        }
        if batch_metrics is not None:
            if "psnr" in batch_metrics:
                postfix["psnr"] = f"{batch_metrics['psnr']:.2f}"
            if "ssim" in batch_metrics:
                postfix["ssim"] = f"{batch_metrics['ssim']:.3f}"
        pbar.set_postfix(postfix)

        if should_log:
            log_msg = f"[Train][Diffusion] epoch={epoch} step={step}/{len(loader)} loss={loss.detach().item():.4f}"
            if batch_metrics is not None:
                if "psnr" in batch_metrics and "ssim" in batch_metrics:
                    log_msg += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
                if "lpips" in batch_metrics:
                    log_msg += f" lpips={batch_metrics['lpips']:.4f}"
            _safe_log(log_msg, use_tqdm=use_tqdm)

    return _mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_diffusion(
    model: ConditionalDiffusionModel,
    loader: DataLoader[Any],
    metrics: RestorationMetrics,
    device: torch.device,
    ddim_steps: int,
) -> dict[str, float]:
    """Evaluate diffusion model."""
    model.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        degraded_n = to_minus1_1(degraded)
        pred_n = model.sample_ddim(cond=degraded_n, steps=ddim_steps)
        pred_01 = to_0_1(pred_n)

        batch_metrics = metrics.compute_batch(pred=pred_01, target=clear)

        for k, v in batch_metrics.items():
            sum_stats[k] += float(v)
        n_batches += 1

    return _mean_stats(sum_stats, n_batches)


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Save checkpoint dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    cfg = load_config(args.config)
    validate_data_protocol(cfg)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    runtime_cfg = cfg.get("runtime", {})
    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    amp_requested = bool(runtime_cfg.get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and not amp_enabled:
        print("[WARN] AMP requested but CUDA is unavailable. AMP is disabled.")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] AMP enabled: {amp_enabled}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders(cfg=cfg, seed=seed)
    print(f"[INFO] train batches={len(train_loader)}, val batches={len(val_loader)}")

    metrics_cfg = cfg["metrics"]
    metric_computer = RestorationMetrics(
        compute_lpips=bool(metrics_cfg.get("compute_lpips", True)),
        lpips_net=str(metrics_cfg.get("lpips_net", "alex")),
        device=device,
    )
    if metric_computer.lpips.enabled and not metric_computer.lpips.available:
        print(f"[WARN] LPIPS disabled at runtime: {metric_computer.lpips.error_message}")

    model_cfg = cfg["model"]
    model_type = str(model_cfg["type"]).lower()

    opt_cfg = cfg["optimizer"]
    lr = float(opt_cfg["learning_rate"])
    betas = (float(opt_cfg["beta1"]), float(opt_cfg["beta2"]))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    log_interval = int(train_cfg.get("log_interval", 20))
    val_interval = int(train_cfg.get("val_interval", 1))
    save_interval = int(train_cfg.get("save_interval", 5))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    train_metric_interval = max(0, int(train_cfg.get("train_metric_interval", 0)))
    train_metric_on_log = bool(train_cfg.get("train_metric_on_log", True))
    use_tqdm = bool(train_cfg.get("tqdm", True))
    fast_train = bool(train_cfg.get("fast_train", False))

    if fast_train:
        # Throughput-first mode: avoid expensive train-time metrics unless explicitly requested.
        train_metric_on_log = bool(train_cfg.get("train_metric_on_log_fast", False))
        train_metric_interval = max(0, int(train_cfg.get("train_metric_interval_fast", 0)))

    ckpt_cfg = cfg["checkpoint"]
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor_name = str(ckpt_cfg.get("monitor", "psnr"))
    best_value = -1e9

    scheduler_cfg = cfg.get("scheduler", {})
    use_scheduler = bool(scheduler_cfg.get("enabled", False))
    step_size = int(scheduler_cfg.get("step_size", 30))
    gamma = float(scheduler_cfg.get("gamma", 0.5))

    if model_type == "unet":
        model = build_baseline_unet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 64)),
        ).to(device)

        criterion = nn.L1Loss()
        optimizer = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scaler = GradScaler(enabled=amp_enabled)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) if use_scheduler else None

        for epoch in range(1, epochs + 1):
            train_stats = train_one_epoch_unet(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=metric_computer,
                device=device,
                epoch=epoch,
                log_interval=log_interval,
                max_grad_norm=max_grad_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
                metric_interval=train_metric_interval,
                metric_on_log=train_metric_on_log,
                use_tqdm=use_tqdm,
            )

            if scheduler is not None:
                scheduler.step()

            print(f"[Epoch {epoch}] train={train_stats}")

            do_val = (val_interval > 0) and (epoch % val_interval == 0)
            val_stats: dict[str, float] | None = None
            if do_val:
                val_stats = evaluate_unet(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    metrics=metric_computer,
                    device=device,
                )
                print(f"[Epoch {epoch}] val={val_stats}")

                metric_value = float(val_stats.get(monitor_name, -1e9))
                if metric_value > best_value:
                    best_value = metric_value
                    best_path = ckpt_dir / "best_unet.pt"
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_type": model_type,
                            "config": copy.deepcopy(cfg),
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                            "val_stats": val_stats,
                        },
                        best_path,
                    )
                    print(f"[INFO] Saved new best checkpoint: {best_path} ({monitor_name}={metric_value:.4f})")

            if save_interval > 0 and (epoch % save_interval == 0 or epoch == epochs):
                last_path = ckpt_dir / f"unet_epoch_{epoch}.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": model_type,
                        "config": copy.deepcopy(cfg),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "train_stats": train_stats,
                    },
                    last_path,
                )
                print(f"[INFO] Saved checkpoint: {last_path}")

        return

    if model_type == "gan":
        generator, discriminator = build_pix2pix_models(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 64)),
        )
        generator = generator.to(device)
        discriminator = discriminator.to(device)

        optimizer_g = Adam(generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        optimizer_d = Adam(discriminator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scaler = GradScaler(enabled=amp_enabled)

        scheduler_g = StepLR(optimizer_g, step_size=step_size, gamma=gamma) if use_scheduler else None
        scheduler_d = StepLR(optimizer_d, step_size=step_size, gamma=gamma) if use_scheduler else None

        loss_cfg = cfg["loss"]
        gan_cfg = loss_cfg.get("gan", {})
        gan_l1 = float(gan_cfg.get("lambda_l1", 100.0))
        legacy_l1 = loss_cfg.get("l1_weight", None)
        if legacy_l1 is not None and abs(float(legacy_l1) - gan_l1) > 1e-8:
            print(
                "[WARN] loss.l1_weight is deprecated for GAN and ignored. "
                "Using loss.gan.lambda_l1 as the only GAN L1 weight."
            )
        gan_weights = GANLossWeights(
            lambda_l1=gan_l1,
            lambda_physics=float(gan_cfg.get("lambda_physics", 10.0)),
        )

        for epoch in range(1, epochs + 1):
            train_stats = train_one_epoch_gan(
                generator=generator,
                discriminator=discriminator,
                loader=train_loader,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                gan_weights=gan_weights,
                metrics=metric_computer,
                device=device,
                epoch=epoch,
                log_interval=log_interval,
                max_grad_norm=max_grad_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
                metric_interval=train_metric_interval,
                metric_on_log=train_metric_on_log,
                use_tqdm=use_tqdm,
            )

            if scheduler_g is not None:
                scheduler_g.step()
            if scheduler_d is not None:
                scheduler_d.step()

            print(f"[Epoch {epoch}] train={train_stats}")

            do_val = (val_interval > 0) and (epoch % val_interval == 0)
            val_stats: dict[str, float] | None = None
            if do_val:
                val_stats = evaluate_gan(
                    generator=generator,
                    discriminator=discriminator,
                    loader=val_loader,
                    gan_weights=gan_weights,
                    metrics=metric_computer,
                    device=device,
                )
                print(f"[Epoch {epoch}] val={val_stats}")

                metric_value = float(val_stats.get(monitor_name, -1e9))
                if metric_value > best_value:
                    best_value = metric_value
                    best_path = ckpt_dir / "best_gan.pt"
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_type": model_type,
                            "config": copy.deepcopy(cfg),
                            "generator_state": generator.state_dict(),
                            "discriminator_state": discriminator.state_dict(),
                            "optimizer_g_state": optimizer_g.state_dict(),
                            "optimizer_d_state": optimizer_d.state_dict(),
                            "scheduler_g_state": scheduler_g.state_dict() if scheduler_g is not None else None,
                            "scheduler_d_state": scheduler_d.state_dict() if scheduler_d is not None else None,
                            "val_stats": val_stats,
                        },
                        best_path,
                    )
                    print(f"[INFO] Saved new best checkpoint: {best_path} ({monitor_name}={metric_value:.4f})")

            if save_interval > 0 and (epoch % save_interval == 0 or epoch == epochs):
                last_path = ckpt_dir / f"gan_epoch_{epoch}.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": model_type,
                        "config": copy.deepcopy(cfg),
                        "generator_state": generator.state_dict(),
                        "discriminator_state": discriminator.state_dict(),
                        "optimizer_g_state": optimizer_g.state_dict(),
                        "optimizer_d_state": optimizer_d.state_dict(),
                        "scheduler_g_state": scheduler_g.state_dict() if scheduler_g is not None else None,
                        "scheduler_d_state": scheduler_d.state_dict() if scheduler_d is not None else None,
                        "train_stats": train_stats,
                    },
                    last_path,
                )
                print(f"[INFO] Saved checkpoint: {last_path}")

        return

    if model_type == "diffusion":
        # Build diffusion configuration
        diffusion_cfg = model_cfg.get("diffusion", {})
        diff_config = DiffusionConfig(
            image_channels=int(model_cfg.get("in_channels", 3)),
            cond_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            timesteps=int(diffusion_cfg.get("timesteps", 1000)),
            beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
            beta_end=float(diffusion_cfg.get("beta_end", 2e-2)),
        )

        model = ConditionalDiffusionModel(config=diff_config).to(device)

        optimizer = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scaler = GradScaler(enabled=amp_enabled)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) if use_scheduler else None

        loss_cfg = cfg["loss"]
        diffusion_loss_cfg = loss_cfg.get("diffusion", {})
        noise_weight = float(diffusion_loss_cfg.get("noise_weight", 1.0))

        # Get DDIM sampling steps for validation
        ddim_steps = int(diffusion_cfg.get("ddim_steps", 50))

        for epoch in range(1, epochs + 1):
            train_stats = train_one_epoch_diffusion(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                noise_weight=noise_weight,
                metrics=metric_computer,
                device=device,
                epoch=epoch,
                log_interval=log_interval,
                max_grad_norm=max_grad_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
                metric_interval=train_metric_interval,
                metric_on_log=train_metric_on_log,
                ddim_steps=ddim_steps,
                use_tqdm=use_tqdm,
            )

            if scheduler is not None:
                scheduler.step()

            print(f"[Epoch {epoch}] train={train_stats}")

            do_val = (val_interval > 0) and (epoch % val_interval == 0)
            val_stats: dict[str, float] | None = None
            if do_val:
                val_stats = evaluate_diffusion(
                    model=model,
                    loader=val_loader,
                    metrics=metric_computer,
                    device=device,
                    ddim_steps=ddim_steps,
                )
                print(f"[Epoch {epoch}] val={val_stats}")

                metric_value = float(val_stats.get(monitor_name, -1e9))
                if metric_value > best_value:
                    best_value = metric_value
                    best_path = ckpt_dir / "best_diffusion.pt"
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_type": model_type,
                            "config": copy.deepcopy(cfg),
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                            "val_stats": val_stats,
                        },
                        best_path,
                    )
                    print(f"[INFO] Saved new best checkpoint: {best_path} ({monitor_name}={metric_value:.4f})")

            if save_interval > 0 and (epoch % save_interval == 0 or epoch == epochs):
                last_path = ckpt_dir / f"diffusion_epoch_{epoch}.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": model_type,
                        "config": copy.deepcopy(cfg),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "train_stats": train_stats,
                    },
                    last_path,
                )
                print(f"[INFO] Saved checkpoint: {last_path}")

        return

    raise ValueError(f"Unsupported model type: {model_type}. Expected 'unet', 'gan', or 'diffusion'.")


if __name__ == "__main__":
    main()
