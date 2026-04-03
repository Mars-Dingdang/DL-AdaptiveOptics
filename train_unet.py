"""Standalone U-Net training entrypoint for turbulence restoration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.baseline_unet import build_baseline_unet
from train_common import (
    adapt_degraded_for_model,
    build_dataloaders,
    compute_mean_stats,
    load_config,
    parse_train_args,
    resolve_device,
    resolve_cond_channels,
    save_checkpoint,
    set_seed,
    validate_data_protocol,
)
from utils.metrics import RestorationMetrics


def _safe_log(message: str, use_tqdm: bool) -> None:
    if use_tqdm:
        tqdm.write(message)
    else:
        print(message)


def train_one_epoch_unet(
    model: nn.Module,
    loader: DataLoader,
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
    cfg_for_adapt: dict[str, object],
) -> dict[str, float]:
    model.train()
    use_amp = amp_enabled and device.type == "cuda"

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(loader, total=len(loader), disable=not use_tqdm, desc=f"Train UNet e{epoch}", leave=False, dynamic_ncols=True)

    for step, (degraded, clear) in enumerate(pbar, start=1):
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        with autocast(enabled=use_amp):
            pred = model(adapt_degraded_for_model(degraded=degraded, cfg=cfg_for_adapt))
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

        postfix: dict[str, str] = {"loss": f"{loss.detach().item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
        if batch_metrics is not None:
            if "psnr" in batch_metrics:
                postfix["psnr"] = f"{batch_metrics['psnr']:.2f}"
            if "ssim" in batch_metrics:
                postfix["ssim"] = f"{batch_metrics['ssim']:.3f}"
        pbar.set_postfix(postfix)

        if should_log:
            log_msg = f"[Train][UNet] epoch={epoch} step={step}/{len(loader)} loss={loss.detach().item():.4f}"
            if batch_metrics is not None and "psnr" in batch_metrics and "ssim" in batch_metrics:
                log_msg += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
            _safe_log(log_msg, use_tqdm=use_tqdm)

    return compute_mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_unet(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metrics: RestorationMetrics,
    device: torch.device,
    cfg_for_adapt: dict[str, object],
) -> dict[str, float]:
    model.eval()
    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        pred = model(adapt_degraded_for_model(degraded=degraded, cfg=cfg_for_adapt))
        pred_01 = pred.clamp(0.0, 1.0)
        loss = criterion(pred_01, clear)
        batch_metrics = metrics.compute_batch(pred=pred_01, target=clear)

        sum_stats["loss"] += float(loss.detach().item())
        for k, v in batch_metrics.items():
            sum_stats[k] += float(v)
        n_batches += 1

    return compute_mean_stats(sum_stats, n_batches)


def main() -> None:
    args = parse_train_args("Train U-Net model for atmospheric turbulence removal.")
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

    cond_channels = resolve_cond_channels(cfg)
    model_cfg = cfg["model"]
    model = build_baseline_unet(
        in_channels=cond_channels,
        out_channels=int(model_cfg.get("out_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    ).to(device)

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
            cfg_for_adapt=cfg,
        )

        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch}] train={train_stats}")

        if val_interval > 0 and epoch % val_interval == 0:
            val_stats = evaluate_unet(
                model=model,
                loader=val_loader,
                criterion=criterion,
                metrics=metric_computer,
                device=device,
                cfg_for_adapt=cfg,
            )
            print(f"[Epoch {epoch}] val={val_stats}")
            metric_value = float(val_stats.get(monitor_name, -1e9))
            if metric_value > best_value:
                best_value = metric_value
                best_path = ckpt_dir / "best_unet.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": "unet",
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
                    "model_type": "unet",
                    "config": copy.deepcopy(cfg),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "train_stats": train_stats,
                },
                last_path,
            )
            print(f"[INFO] Saved checkpoint: {last_path}")


if __name__ == "__main__":
    main()
