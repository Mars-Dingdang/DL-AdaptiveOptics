"""Evaluation metrics for image restoration.

This module provides:
- PSNR and SSIM based on scikit-image.
- Optional LPIPS perceptual metric.

All interfaces assume tensors are in [0, 1] unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch


@dataclass
class LPIPSConfig:
    """Configuration for optional LPIPS metric."""

    enabled: bool = True
    net: str = "alex"


class LPIPSMetric:
    """Lazy LPIPS wrapper with graceful fallback when dependency is missing."""

    def __init__(self, config: LPIPSConfig, device: torch.device) -> None:
        self.enabled = bool(config.enabled)
        self.net = config.net
        self.device = device
        self.available = False
        self._model: Any | None = None
        self.error_message: str | None = None

        if not self.enabled:
            return

        try:
            import lpips  # type: ignore

            self._model = lpips.LPIPS(net=self.net).to(self.device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
            self.available = True
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            self.available = False
            self.error_message = str(exc)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float | None:
        """Compute LPIPS score for a batch. Returns None if unavailable."""
        if not self.enabled or not self.available or self._model is None:
            return None

        pred_11 = pred * 2.0 - 1.0
        target_11 = target * 2.0 - 1.0
        score = self._model(pred_11, target_11)
        return float(score.mean().item())


def tensor_to_numpy_image(image_chw: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [0,1] to HWC float32 numpy array."""
    if image_chw.ndim != 3:
        raise ValueError("Expected CHW tensor")

    image = image_chw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(image, dtype=np.float32)


def batch_psnr_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> tuple[float, float]:
    """Compute average PSNR and SSIM over a batch.

    Args:
        pred: Predicted images in [0,1], shape [B, C, H, W].
        target: Ground-truth images in [0,1], shape [B, C, H, W].
        data_range: Value range used by metrics.

    Returns:
        (mean_psnr, mean_ssim)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.ndim != 4:
        raise ValueError("Expected tensors with shape [B, C, H, W]")

    b = pred.shape[0]
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []

    for i in range(b):
        pred_np = tensor_to_numpy_image(pred[i])
        target_np = tensor_to_numpy_image(target[i])

        psnr_i = peak_signal_noise_ratio(target_np, pred_np, data_range=data_range)
        ssim_i = structural_similarity(
            target_np,
            pred_np,
            channel_axis=-1,
            data_range=data_range,
        )
        psnr_vals.append(float(psnr_i))
        ssim_vals.append(float(ssim_i))

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


class RestorationMetrics:
    """Metric computer for restoration tasks."""

    def __init__(self, compute_lpips: bool, lpips_net: str, device: torch.device) -> None:
        self.lpips = LPIPSMetric(config=LPIPSConfig(enabled=compute_lpips, net=lpips_net), device=device)

    @torch.no_grad()
    def compute_batch(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Compute metrics for one batch.

        Inputs are expected in [0, 1].
        """
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        psnr, ssim = batch_psnr_ssim(pred=pred, target=target, data_range=1.0)
        out: dict[str, float] = {
            "psnr": psnr,
            "ssim": ssim,
        }

        lpips_val = self.lpips(pred=pred, target=target)
        if lpips_val is not None:
            out["lpips"] = lpips_val

        return out
