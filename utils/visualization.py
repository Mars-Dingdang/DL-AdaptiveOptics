"""Visualization helpers for restoration experiments.

This module provides utilities to save side-by-side comparisons of:
- degraded input,
- ground-truth clear image,
- model prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


def tensor_chw_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [0,1] to uint8 HWC RGB."""
    if image.ndim != 3:
        raise ValueError("Expected CHW tensor")
    arr = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def numpy_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Convert numpy image to uint8 HWC RGB."""
    if image.ndim != 3:
        raise ValueError("Expected HWC image")

    if image.dtype == np.uint8:
        out = image
    else:
        out = image.astype(np.float32)
        if out.max() > 1.0:
            out = out / 255.0
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)

    return out


def _put_label(image: np.ndarray, text: str) -> np.ndarray:
    """Add a small top-left label to image."""
    out = image.copy()
    cv2.rectangle(out, (0, 0), (180, 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def save_image_rgb(image: np.ndarray, path: str | Path) -> None:
    """Save RGB uint8 image to disk using OpenCV."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    image_u8 = numpy_to_uint8_hwc(image)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path_obj), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path_obj}")


def save_triplet_comparison(
    degraded: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    path: str | Path,
    with_labels: bool = True,
) -> None:
    """Save a horizontal triplet: degraded | target | prediction."""
    d = numpy_to_uint8_hwc(degraded)
    t = numpy_to_uint8_hwc(target)
    p = numpy_to_uint8_hwc(pred)

    if with_labels:
        d = _put_label(d, "Input")
        t = _put_label(t, "Ground Truth")
        p = _put_label(p, "Prediction")

    canvas = np.concatenate([d, t, p], axis=1)
    save_image_rgb(canvas, path)


def save_batch_triplets(
    degraded_batch: torch.Tensor,
    target_batch: torch.Tensor,
    pred_batch: torch.Tensor,
    out_dir: str | Path,
    prefix: str,
    start_index: int = 0,
    max_items: int | None = None,
) -> int:
    """Save triplet comparisons from batched tensors.

    Returns:
        Number of saved samples.
    """
    if degraded_batch.ndim != 4 or target_batch.ndim != 4 or pred_batch.ndim != 4:
        raise ValueError("All inputs must be [B, C, H, W]")

    bsz = degraded_batch.shape[0]
    n = bsz if max_items is None else min(bsz, max_items)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(n):
        d = tensor_chw_to_uint8_hwc(degraded_batch[i])
        t = tensor_chw_to_uint8_hwc(target_batch[i])
        p = tensor_chw_to_uint8_hwc(pred_batch[i])

        filename = f"{prefix}_{start_index + i:05d}.png"
        save_triplet_comparison(d, t, p, out_path / filename, with_labels=True)
        count += 1

    return count


def stack_images_h(images: Iterable[np.ndarray]) -> np.ndarray:
    """Stack multiple RGB images horizontally."""
    arrs = [numpy_to_uint8_hwc(img) for img in images]
    if not arrs:
        raise ValueError("images cannot be empty")
    return np.concatenate(arrs, axis=1)
