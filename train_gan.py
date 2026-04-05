"""Standalone GAN training entrypoint for turbulence restoration."""

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

from modules.gan_models import (
    GANLossWeights,
    build_pix2pix_models,
    discriminator_loss,
    generator_total_loss,
)
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
    to_0_1,
    to_minus1_1,
    validate_data_protocol,
)
from utils.metrics import RestorationMetrics


def _safe_log(message: str, use_tqdm: bool) -> None:
    if use_tqdm:
        tqdm.write(message)
    else:
        print(message)


def _toggle_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def train_one_epoch_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
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
    d_steps_per_g: int,
    real_label_smooth: float,
    use_tqdm: bool,
    cfg_for_adapt: dict[str, object],
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    use_amp = amp_enabled and device.type == "cuda"

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(loader, total=len(loader), disable=not use_tqdm, desc=f"Train GAN e{epoch}", leave=False, dynamic_ncols=True)

    for step, (degraded, clear) in enumerate(pbar, start=1):
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        degraded_model = adapt_degraded_for_model(degraded=degraded, cfg=cfg_for_adapt)

        degraded_n = to_minus1_1(degraded_model)
        clear_n = to_minus1_1(clear)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        # Update discriminator (possibly multiple steps / TTUR-style balancing).
        _toggle_requires_grad(discriminator, True)
        fake_clear = None
        loss_d = torch.tensor(0.0, device=device)
        for _ in range(max(1, d_steps_per_g)):
            with autocast(enabled=use_amp):
                with torch.no_grad():
                    fake_clear_detached = generator(degraded_n)

                pred_real = discriminator(degraded_n, clear_n)
                pred_fake = discriminator(degraded_n, fake_clear_detached)
                loss_d = discriminator_loss(
                    pred_real_logits=pred_real,
                    pred_fake_logits=pred_fake,
                    real_label=real_label_smooth,
                    fake_label=0.0,
                )

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
        _toggle_requires_grad(discriminator, False)
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
        if compute_metrics_now and fake_clear is not None:
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
            "lr_d": f"{optimizer_d.param_groups[0]['lr']:.2e}",
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
            if batch_metrics is not None and "psnr" in batch_metrics and "ssim" in batch_metrics:
                log_msg += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
            _safe_log(log_msg, use_tqdm=use_tqdm)

    return compute_mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    gan_weights: GANLossWeights,
    metrics: RestorationMetrics,
    device: torch.device,
    cfg_for_adapt: dict[str, object],
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        degraded_model = adapt_degraded_for_model(degraded=degraded, cfg=cfg_for_adapt)

        degraded_n = to_minus1_1(degraded_model)
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

    return compute_mean_stats(sum_stats, n_batches)


def main() -> None:
    args = parse_train_args("Train Pix2Pix GAN for atmospheric turbulence removal.")
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
    cond_channels = resolve_cond_channels(cfg)
    generator, discriminator = build_pix2pix_models(
        in_channels=cond_channels,
        out_channels=int(model_cfg.get("out_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    )
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    opt_cfg = cfg["optimizer"]
    lr = float(opt_cfg["learning_rate"])
    lr_g = float(opt_cfg.get("gan_lr_g", lr))
    lr_d = float(opt_cfg.get("gan_lr_d", lr))
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
    gan_cfg = loss_cfg.get("gan", {})
    gan_l1 = float(gan_cfg.get("lambda_l1", 100.0))
    legacy_l1 = loss_cfg.get("l1_weight", None)
    if legacy_l1 is not None and abs(float(legacy_l1) - gan_l1) > 1e-8:
        print("[WARN] loss.l1_weight is deprecated for GAN and ignored. Using loss.gan.lambda_l1 only.")

    gan_weights = GANLossWeights(
        lambda_l1=gan_l1,
        lambda_physics=float(gan_cfg.get("lambda_physics", 10.0)),
    )
    d_steps_per_g = int(gan_cfg.get("d_steps_per_g", 1))
    real_label_smooth = float(gan_cfg.get("real_label_smooth", 0.9))

    optimizer_g = Adam(generator.parameters(), lr=lr_g, betas=betas, weight_decay=weight_decay)
    optimizer_d = Adam(discriminator.parameters(), lr=lr_d, betas=betas, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp_enabled)

    scheduler_g = StepLR(optimizer_g, step_size=step_size, gamma=gamma) if use_scheduler else None
    scheduler_d = StepLR(optimizer_d, step_size=step_size, gamma=gamma) if use_scheduler else None

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
            d_steps_per_g=d_steps_per_g,
            real_label_smooth=real_label_smooth,
            use_tqdm=use_tqdm,
            cfg_for_adapt=cfg,
        )

        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        print(f"[Epoch {epoch}] train={train_stats}")

        if val_interval > 0 and epoch % val_interval == 0:
            val_stats = evaluate_gan(
                generator=generator,
                discriminator=discriminator,
                loader=val_loader,
                gan_weights=gan_weights,
                metrics=metric_computer,
                device=device,
                cfg_for_adapt=cfg,
            )
            print(f"[Epoch {epoch}] val={val_stats}")

            metric_value = float(val_stats.get(monitor_name, -1e9))
            if metric_value > best_value:
                best_value = metric_value
                best_path = ckpt_dir / "best_gan.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": "gan",
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
                    "model_type": "gan",
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


if __name__ == "__main__":
    main()
