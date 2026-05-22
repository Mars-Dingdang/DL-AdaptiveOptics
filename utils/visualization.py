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
from PIL import Image
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


def save_sequence_triplet_gif(
    degraded_sequence: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    path: str | Path,
    duration_ms: int = 220,
    loop: int = 0,
    with_labels: bool = True,
) -> None:
    """Save an animated comparison: degraded sequence | target | prediction."""
    if degraded_sequence.ndim != 4:
        raise ValueError("degraded_sequence must be [T, C, H, W]")
    if target.ndim != 3 or pred.ndim != 3:
        raise ValueError("target and pred must be [C, H, W]")

    target_img = tensor_chw_to_uint8_hwc(target)
    pred_img = tensor_chw_to_uint8_hwc(pred)
    if with_labels:
        target_img = _put_label(target_img, "Ground Truth")
        pred_img = _put_label(pred_img, "Prediction")

    frames: list[Image.Image] = []
    for frame_idx in range(degraded_sequence.shape[0]):
        degraded_img = tensor_chw_to_uint8_hwc(degraded_sequence[frame_idx])
        if with_labels:
            degraded_img = _put_label(degraded_img, "Input")
        canvas = np.concatenate([degraded_img, target_img, pred_img], axis=1)
        frames.append(Image.fromarray(canvas, mode="RGB"))

    if not frames:
        raise ValueError("degraded_sequence must contain at least one frame")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path_obj,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=int(loop),
        optimize=False,
    )


def save_batch_sequence_triplet_gifs(
    degraded_batch: torch.Tensor,
    target_batch: torch.Tensor,
    pred_batch: torch.Tensor,
    out_dir: str | Path,
    prefix: str,
    start_index: int = 0,
    max_items: int | None = None,
    duration_ms: int = 220,
) -> int:
    """Save animated triplet GIFs from sequence batches."""
    if degraded_batch.ndim != 5 or target_batch.ndim != 4 or pred_batch.ndim != 4:
        raise ValueError("Expected degraded [B,T,C,H,W], target/pred [B,C,H,W]")

    bsz = degraded_batch.shape[0]
    n = bsz if max_items is None else min(bsz, max_items)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(n):
        filename = f"{prefix}_{start_index + i:05d}.gif"
        save_sequence_triplet_gif(
            degraded_sequence=degraded_batch[i],
            target=target_batch[i],
            pred=pred_batch[i],
            path=out_path / filename,
            duration_ms=duration_ms,
            with_labels=True,
        )
        count += 1

    return count


def stack_images_h(images: Iterable[np.ndarray]) -> np.ndarray:
    """Stack multiple RGB images horizontally."""
    arrs = [numpy_to_uint8_hwc(img) for img in images]
    if not arrs:
        raise ValueError("images cannot be empty")
    return np.concatenate(arrs, axis=1)
