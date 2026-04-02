"""Standalone VAE training entrypoint for turbulence restoration."""

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

from modules.vae import VAEConfig, build_conditional_vae, kl_divergence
from train_common import (
    build_dataloaders,
    compute_mean_stats,
    load_config,
    parse_train_args,
    resolve_device,
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


def train_one_epoch_vae(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    recon_criterion: nn.Module,
    metrics: RestorationMetrics,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_grad_norm: float,
    amp_enabled: bool,
    scaler: GradScaler | None,
    metric_interval: int,
    metric_on_log: bool,
    kl_weight: float,
    use_tqdm: bool,
) -> dict[str, float]:
    model.train()
    use_amp = amp_enabled and device.type == "cuda"

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(loader, total=len(loader), disable=not use_tqdm, desc=f"Train VAE e{epoch}", leave=False, dynamic_ncols=True)

    for step, (degraded, clear) in enumerate(pbar, start=1):
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        with autocast(enabled=use_amp):
            recon, mu, logvar = model(degraded=degraded, clear=clear)
            recon_01 = recon.clamp(0.0, 1.0)
            loss_recon = recon_criterion(recon_01, clear)
            loss_kl = kl_divergence(mu=mu, logvar=logvar)
            loss_total = loss_recon + kl_weight * loss_kl

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(loss_total).backward()
            if max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        batch_metrics: dict[str, float] | None = None
        if compute_metrics_now:
            batch_metrics = metrics.compute_batch(pred=recon_01.detach(), target=clear.detach())

        sum_stats["loss"] += float(loss_total.detach().item())
        sum_counts["loss"] += 1
        sum_stats["recon"] += float(loss_recon.detach().item())
        sum_counts["recon"] += 1
        sum_stats["kl"] += float(loss_kl.detach().item())
        sum_counts["kl"] += 1
        if batch_metrics is not None:
            for k, v in batch_metrics.items():
                sum_stats[k] += float(v)
                sum_counts[k] += 1

        postfix: dict[str, str] = {
            "loss": f"{loss_total.detach().item():.4f}",
            "recon": f"{loss_recon.detach().item():.4f}",
            "kl": f"{loss_kl.detach().item():.4f}",
            "klw": f"{kl_weight:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        if batch_metrics is not None:
            if "psnr" in batch_metrics:
                postfix["psnr"] = f"{batch_metrics['psnr']:.2f}"
            if "ssim" in batch_metrics:
                postfix["ssim"] = f"{batch_metrics['ssim']:.3f}"
        pbar.set_postfix(postfix)

        if should_log:
            log_msg = (
                f"[Train][VAE] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_total.detach().item():.4f} recon={loss_recon.detach().item():.4f} kl={loss_kl.detach().item():.4f}"
            )
            if batch_metrics is not None and "psnr" in batch_metrics and "ssim" in batch_metrics:
                log_msg += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
            _safe_log(log_msg, use_tqdm=use_tqdm)

    return compute_mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_vae(
    model: nn.Module,
    loader: DataLoader,
    recon_criterion: nn.Module,
    metrics: RestorationMetrics,
    device: torch.device,
    kl_weight: float,
) -> dict[str, float]:
    model.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)

        recon, mu, logvar = model(degraded=degraded, clear=clear)
        recon_01 = recon.clamp(0.0, 1.0)
        loss_recon = recon_criterion(recon_01, clear)
        loss_kl = kl_divergence(mu=mu, logvar=logvar)
        loss_total = loss_recon + kl_weight * loss_kl

        batch_metrics = metrics.compute_batch(pred=recon_01, target=clear)

        sum_stats["loss"] += float(loss_total.detach().item())
        sum_stats["recon"] += float(loss_recon.detach().item())
        sum_stats["kl"] += float(loss_kl.detach().item())
        for k, v in batch_metrics.items():
            sum_stats[k] += float(v)
        n_batches += 1

    return compute_mean_stats(sum_stats, n_batches)


def main() -> None:
    args = parse_train_args("Train conditional VAE for atmospheric turbulence removal.")
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

    model_cfg = cfg["model"]
    vae_cfg = model_cfg.get("vae", {})
    vae_config = VAEConfig(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        cond_channels=int(model_cfg.get("in_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        latent_channels=int(vae_cfg.get("latent_channels", 128)),
    )
    model = build_conditional_vae(config=vae_config).to(device)

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

    loss_cfg = cfg["loss"]
    vae_loss_cfg = loss_cfg.get("vae", {})
    kl_weight_max = float(vae_loss_cfg.get("kl_weight", 0.01))
    # `kl_warmup_epochs=1` means no warmup (full KL weight from first epoch).
    kl_warmup_epochs = max(1, int(vae_loss_cfg.get("kl_warmup_epochs", 10)))

    recon_criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp_enabled)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) if use_scheduler else None

    for epoch in range(1, epochs + 1):
        current_kl_weight = min(1.0, float(epoch) / float(kl_warmup_epochs)) * kl_weight_max

        train_stats = train_one_epoch_vae(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            recon_criterion=recon_criterion,
            metrics=metric_computer,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
            amp_enabled=amp_enabled,
            scaler=scaler,
            metric_interval=train_metric_interval,
            metric_on_log=train_metric_on_log,
            kl_weight=current_kl_weight,
            use_tqdm=use_tqdm,
        )

        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch}] train={train_stats}")

        if val_interval > 0 and epoch % val_interval == 0:
            val_stats = evaluate_vae(
                model=model,
                loader=val_loader,
                recon_criterion=recon_criterion,
                metrics=metric_computer,
                device=device,
                kl_weight=current_kl_weight,
            )
            print(f"[Epoch {epoch}] val={val_stats}")
            metric_value = float(val_stats.get(monitor_name, -1e9))
            if metric_value > best_value:
                best_value = metric_value
                best_path = ckpt_dir / "best_vae.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": "vae",
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
            last_path = ckpt_dir / f"vae_epoch_{epoch}.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_type": "vae",
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
