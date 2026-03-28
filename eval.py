"""Unified evaluation script for restoration checkpoints.

This script evaluates either U-Net or GAN checkpoints on a validation split and
reports PSNR/SSIM/LPIPS metrics. It can also export side-by-side result images.

Usage examples:
    python eval.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt
    python eval.py --config configs/default.yaml --checkpoint checkpoints/best_gan.pt --save-images
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import DatasetParams, TurbulencePairDataset
from modules.baseline_unet import build_baseline_unet
from modules.gan_models import build_pix2pix_models
from train import (
    build_turbulence_params,
    load_config,
    resolve_device,
    set_seed,
    to_0_1,
    to_minus1_1,
)
from utils.metrics import RestorationMetrics
from utils.visualization import save_batch_triplets


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate restoration model checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file path")
    parser.add_argument(
        "--model-type",
        type=str,
        default="",
        choices=["", "unet", "gan"],
        help="Optional model type override. If empty, infer from checkpoint/config.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Override eval batch size if > 0")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override dataloader workers if >= 0")
    parser.add_argument("--save-images", action="store_true", help="Save visual triplets")
    parser.add_argument("--max-save", type=int, default=30, help="Maximum number of images to save")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"), help="Evaluation output dir")
    return parser.parse_args()


def build_eval_loader(cfg: dict[str, Any], seed: int, batch_size_override: int, workers_override: int) -> DataLoader[Any]:
    """Build evaluation dataloader from val_root or split from train_root."""
    data_cfg = cfg["data"]
    train_root = Path(data_cfg["train_root"])
    val_root_str = str(data_cfg.get("val_root", "")).strip()

    turbulence_params = build_turbulence_params(cfg)
    val_params = DatasetParams(
        image_size=int(data_cfg["image_size"]),
        random_crop=False,
        horizontal_flip_prob=0.0,
    )

    if val_root_str:
        val_ds = TurbulencePairDataset(
            root_dir=Path(val_root_str),
            dataset_params=val_params,
            turbulence_params=turbulence_params,
            seed=seed + 1,
        )
    else:
        full_ds = TurbulencePairDataset(
            root_dir=train_root,
            dataset_params=val_params,
            turbulence_params=turbulence_params,
            seed=seed + 1,
        )
        n_total = len(full_ds)
        if n_total < 2:
            raise RuntimeError("Need at least 2 images for validation split.")

        val_ratio = float(data_cfg.get("val_ratio", 0.1))
        val_ratio = min(max(val_ratio, 0.01), 0.5)
        n_val = max(1, int(round(n_total * val_ratio)))

        indices = np.arange(n_total)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        val_idx = indices[:n_val].tolist()
        val_ds = Subset(full_ds, val_idx)

    batch_size = int(data_cfg.get("val_batch_size", data_cfg["batch_size"]))
    if batch_size_override > 0:
        batch_size = batch_size_override

    num_workers = int(data_cfg["num_workers"])
    if workers_override >= 0:
        num_workers = workers_override

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    return loader


def infer_model_type(ckpt: dict[str, Any], cfg: dict[str, Any], arg_type: str) -> str:
    """Infer model type with priority: CLI arg > ckpt > config."""
    if arg_type:
        return arg_type
    ckpt_type = str(ckpt.get("model_type", "")).lower().strip()
    if ckpt_type in {"unet", "gan"}:
        return ckpt_type
    return str(cfg["model"]["type"]).lower().strip()


def main() -> None:
    """Main evaluation entrypoint."""
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(cfg["runtime"].get("device", "auto")))
    print(f"[INFO] Eval device: {device}")

    loader = build_eval_loader(
        cfg=cfg,
        seed=seed,
        batch_size_override=int(args.batch_size),
        workers_override=int(args.num_workers),
    )
    print(f"[INFO] Eval batches: {len(loader)}")

    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = infer_model_type(ckpt=ckpt, cfg=cfg, arg_type=args.model_type)
    print(f"[INFO] Model type: {model_type}")

    metrics_cfg = cfg.get("metrics", {})
    metric_computer = RestorationMetrics(
        compute_lpips=bool(metrics_cfg.get("compute_lpips", True)),
        lpips_net=str(metrics_cfg.get("lpips_net", "alex")),
        device=device,
    )
    if metric_computer.lpips.enabled and not metric_computer.lpips.available:
        print(f"[WARN] LPIPS disabled at runtime: {metric_computer.lpips.error_message}")

    model_cfg = cfg["model"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0
    saved_count = 0

    if model_type == "unet":
        model = build_baseline_unet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 64)),
        ).to(device)

        state = ckpt.get("model_state")
        if state is None:
            raise RuntimeError("Checkpoint missing 'model_state' for unet evaluation.")
        model.load_state_dict(state, strict=True)
        model.eval()

        with torch.no_grad():
            for batch_idx, (degraded, clear) in enumerate(loader):
                degraded = degraded.to(device, non_blocking=True)
                clear = clear.to(device, non_blocking=True)

                pred = model(degraded).clamp(0.0, 1.0)
                batch_metrics = metric_computer.compute_batch(pred=pred, target=clear)

                for k, v in batch_metrics.items():
                    sum_stats[k] += float(v)
                n_batches += 1

                if args.save_images and saved_count < args.max_save:
                    can_save = min(int(args.max_save - saved_count), pred.shape[0])
                    saved_count += save_batch_triplets(
                        degraded_batch=degraded.detach().cpu(),
                        target_batch=clear.detach().cpu(),
                        pred_batch=pred.detach().cpu(),
                        out_dir=out_dir / "samples",
                        prefix="unet",
                        start_index=saved_count,
                        max_items=can_save,
                    )

    elif model_type == "gan":
        generator, _discriminator = build_pix2pix_models(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 64)),
        )
        generator = generator.to(device)

        state = ckpt.get("generator_state")
        if state is None:
            raise RuntimeError("Checkpoint missing 'generator_state' for gan evaluation.")
        generator.load_state_dict(state, strict=True)
        generator.eval()

        with torch.no_grad():
            for batch_idx, (degraded, clear) in enumerate(loader):
                degraded = degraded.to(device, non_blocking=True)
                clear = clear.to(device, non_blocking=True)

                pred_n = generator(to_minus1_1(degraded))
                pred = to_0_1(pred_n)
                batch_metrics = metric_computer.compute_batch(pred=pred, target=clear)

                for k, v in batch_metrics.items():
                    sum_stats[k] += float(v)
                n_batches += 1

                if args.save_images and saved_count < args.max_save:
                    can_save = min(int(args.max_save - saved_count), pred.shape[0])
                    saved_count += save_batch_triplets(
                        degraded_batch=degraded.detach().cpu(),
                        target_batch=clear.detach().cpu(),
                        pred_batch=pred.detach().cpu(),
                        out_dir=out_dir / "samples",
                        prefix="gan",
                        start_index=saved_count,
                        max_items=can_save,
                    )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if n_batches <= 0:
        raise RuntimeError("No evaluation batches were processed.")

    mean_stats = {k: v / float(n_batches) for k, v in sum_stats.items()}
    print(f"[RESULT] {mean_stats}")

    report_path = out_dir / "metrics.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"model_type: {model_type}\n")
        f.write(f"batches: {n_batches}\n")
        for k, v in sorted(mean_stats.items()):
            f.write(f"{k}: {v:.6f}\n")
    print(f"[INFO] Metrics saved to: {report_path}")

    if args.save_images:
        print(f"[INFO] Saved sample images: {saved_count}")


if __name__ == "__main__":
    main()
