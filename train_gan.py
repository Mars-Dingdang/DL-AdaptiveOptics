"""TSR-WGAN training entrypoint for turbulence sequence restoration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from modules.gan_models import (
    GANLossWeights,
    PerceptualLoss,
    build_fake_discriminator_input,
    build_tsr_wgan_models,
    critic_total_loss,
    generator_total_loss,
    select_discriminator_frame_indices,
)
from train_common import (
    adapt_degraded_for_model,
    build_dataloaders,
    cleanup_distributed,
    compute_mean_stats,
    init_distributed_mode,
    is_main_process,
    load_config,
    parse_train_args,
    reduce_dict,
    reduce_sum_count,
    resolve_device,
    save_checkpoint,
    set_seed,
    to_0_1,
    to_minus1_1,
    validate_data_protocol,
)
from utils.metrics import RestorationMetrics


def _safe_log(message: str, use_tqdm: bool) -> None:
    if not is_main_process():
        return
    if use_tqdm:
        tqdm.write(message)
    else:
        print(message)


def _toggle_requires_grad(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def _build_critic_inputs(
    degraded_sequence: torch.Tensor,
    real_clear: torch.Tensor,
    fake_clear: torch.Tensor,
    frame_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    real_input = build_fake_discriminator_input(
        degraded_sequence=degraded_sequence,
        fake_clear=real_clear,
        frame_indices=frame_indices,
    )
    fake_input = build_fake_discriminator_input(
        degraded_sequence=degraded_sequence,
        fake_clear=fake_clear,
        frame_indices=frame_indices,
    )
    return real_input, fake_input


def train_one_epoch_gan(
    generator: nn.Module,
    critic: nn.Module,
    perceptual_loss_fn: nn.Module,
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss_weights: GANLossWeights,
    metrics: RestorationMetrics,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_grad_norm: float,
    metric_interval: int,
    metric_on_log: bool,
    d_steps_per_g: int,
    frame_indices: list[int],
    use_tqdm: bool,
    cfg_for_adapt: dict[str, object],
) -> dict[str, float]:
    generator.train()
    critic.train()
    perceptual_loss_fn.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    sum_counts: dict[str, int] = defaultdict(int)

    pbar = tqdm(loader, total=len(loader), disable=not use_tqdm, desc=f"Train TSR-WGAN e{epoch}", leave=False, dynamic_ncols=True)

    for step, (degraded, clear) in enumerate(pbar, start=1):
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        degraded_model = adapt_degraded_for_model(degraded=degraded, cfg=cfg_for_adapt)
        if degraded_model.ndim != 5:
            raise ValueError("TSR-WGAN expects sequence tensors shaped [B, T, C, H, W].")

        degraded_n = to_minus1_1(degraded_model)
        clear_n = to_minus1_1(clear)

        should_log = log_interval > 0 and (step % log_interval == 0 or step == len(loader))
        compute_metrics_now = (metric_on_log and should_log) or (metric_interval > 0 and step % metric_interval == 0)

        _toggle_requires_grad(critic, True)
        d_stats: dict[str, float] = {}
        fake_clear = None
        for _ in range(max(1, d_steps_per_g)):
            with torch.no_grad():
                fake_clear = generator(degraded_n)
            real_input, fake_input = _build_critic_inputs(
                degraded_sequence=degraded_n,
                real_clear=clear_n,
                fake_clear=fake_clear,
                frame_indices=frame_indices,
            )
            loss_d, d_stats = critic_total_loss(
                critic=critic,
                fake_discriminator_input=fake_input,
                real_discriminator_input=real_input,
                lambda_gp=loss_weights.lambda_gp,
            )

            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
            optimizer_d.step()

        _toggle_requires_grad(critic, False)
        fake_clear = generator(degraded_n)
        _real_input, fake_input = _build_critic_inputs(
            degraded_sequence=degraded_n,
            real_clear=clear_n,
            fake_clear=fake_clear,
            frame_indices=frame_indices,
        )
        loss_g, g_stats = generator_total_loss(
            critic=critic,
            fake_discriminator_input=fake_input,
            fake_clear=fake_clear,
            real_clear=clear_n,
            perceptual_loss_fn=perceptual_loss_fn,
            weights=loss_weights,
        )

        optimizer_g.zero_grad(set_to_none=True)
        loss_g.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
        optimizer_g.step()

        batch_metrics: dict[str, float] | None = None
        if compute_metrics_now:
            fake_01 = to_0_1(fake_clear.detach())
            batch_metrics = metrics.compute_batch(pred=fake_01, target=clear.detach())

        for key, value in d_stats.items():
            sum_stats[key] += float(value)
            sum_counts[key] += 1
        for key, value in g_stats.items():
            sum_stats[key] += float(value)
            sum_counts[key] += 1
        if batch_metrics is not None:
            for key, value in batch_metrics.items():
                sum_stats[key] += float(value)
                sum_counts[key] += 1

        postfix: dict[str, str] = {
            "d": f"{d_stats.get('d_total', 0.0):.4f}",
            "g": f"{g_stats.get('g_total', 0.0):.4f}",
            "gp": f"{d_stats.get('d_gp', 0.0):.3f}",
            "percep": f"{g_stats.get('g_percep', 0.0):.3f}",
            "pixel": f"{g_stats.get('g_pixel', 0.0):.3f}",
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
            message = (
                f"[Train][TSR-WGAN] epoch={epoch} step={step}/{len(loader)} "
                f"d={d_stats.get('d_total', 0.0):.4f} g={g_stats.get('g_total', 0.0):.4f}"
            )
            if batch_metrics is not None and "psnr" in batch_metrics and "ssim" in batch_metrics:
                message += f" psnr={batch_metrics['psnr']:.3f} ssim={batch_metrics['ssim']:.4f}"
            _safe_log(message, use_tqdm=use_tqdm)

    return compute_mean_stats(sum_stats, sum_counts)


@torch.no_grad()
def evaluate_gan(
    generator: nn.Module,
    critic: nn.Module,
    perceptual_loss_fn: nn.Module,
    loader: DataLoader,
    loss_weights: GANLossWeights,
    metrics: RestorationMetrics,
    device: torch.device,
    frame_indices: list[int],
    cfg_for_adapt: dict[str, object],
    distributed: bool = False,
) -> dict[str, float]:
    generator.eval()
    critic.eval()
    perceptual_loss_fn.eval()

    sum_stats: dict[str, float] = defaultdict(float)
    n_batches = 0

    for degraded, clear in loader:
        degraded = degraded.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        degraded_model = adapt_degraded_for_model(degraded=degraded, cfg=cfg_for_adapt)
        if degraded_model.ndim != 5:
            raise ValueError("TSR-WGAN expects sequence tensors shaped [B, T, C, H, W].")

        degraded_n = to_minus1_1(degraded_model)
        clear_n = to_minus1_1(clear)

        fake_clear = generator(degraded_n)
        real_input, fake_input = _build_critic_inputs(
            degraded_sequence=degraded_n,
            real_clear=clear_n,
            fake_clear=fake_clear,
            frame_indices=frame_indices,
        )
        d_loss, d_stats = critic_total_loss(
            critic=critic,
            fake_discriminator_input=fake_input,
            real_discriminator_input=real_input,
            lambda_gp=loss_weights.lambda_gp,
            include_gp=False,
        )
        del d_loss
        g_total, g_stats = generator_total_loss(
            critic=critic,
            fake_discriminator_input=fake_input,
            fake_clear=fake_clear,
            real_clear=clear_n,
            perceptual_loss_fn=perceptual_loss_fn,
            weights=loss_weights,
        )
        del g_total

        fake_01 = to_0_1(fake_clear)
        batch_metrics = metrics.compute_batch(pred=fake_01, target=clear)

        for key, value in d_stats.items():
            sum_stats[key] += float(value)
        for key, value in g_stats.items():
            sum_stats[key] += float(value)
        for key, value in batch_metrics.items():
            sum_stats[key] += float(value)
        n_batches += 1

    if distributed:
        reduced_sums, reduced_count = reduce_sum_count(dict(sum_stats), n_batches)
        return compute_mean_stats(reduced_sums, reduced_count)
    return compute_mean_stats(sum_stats, n_batches)


def _build_schedulers(
    cfg: dict[str, object],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> tuple[object | None, object | None, str]:
    scheduler_cfg = cfg.get("scheduler", {})
    if not bool(scheduler_cfg.get("enabled", False)):
        return None, None, "none"

    scheduler_type = str(scheduler_cfg.get("type", "step")).lower().strip()
    if scheduler_type == "plateau":
        scheduler_g = ReduceLROnPlateau(
            optimizer_g,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("gamma", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
        )
        scheduler_d = ReduceLROnPlateau(
            optimizer_d,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("gamma", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
        )
        return scheduler_g, scheduler_d, "plateau"

    scheduler_g = StepLR(
        optimizer_g,
        step_size=int(scheduler_cfg.get("step_size", 20)),
        gamma=float(scheduler_cfg.get("gamma", 0.5)),
    )
    scheduler_d = StepLR(
        optimizer_d,
        step_size=int(scheduler_cfg.get("step_size", 20)),
        gamma=float(scheduler_cfg.get("gamma", 0.5)),
    )
    return scheduler_g, scheduler_d, "step"


def main() -> None:
    args = parse_train_args("Train TSR-WGAN for atmospheric turbulence removal.")
    cfg = load_config(args.config)

    distributed, rank, world_size, local_rank = init_distributed_mode()

    try:
        _run(cfg=cfg, distributed=distributed, rank=rank, world_size=world_size, local_rank=local_rank)
    finally:
        cleanup_distributed()


def _run(
    cfg: dict[str, object],
    distributed: bool,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    if is_main_process():
        validate_data_protocol(cfg)

    seed = int(cfg.get("seed", 42))
    set_seed(seed, rank=rank)

    runtime_cfg = cfg.get("runtime", {})
    device = resolve_device(str(runtime_cfg.get("device", "auto")), local_rank=local_rank)
    amp_requested = bool(runtime_cfg.get("amp", False))
    if amp_requested and is_main_process():
        print("[WARN] runtime.amp is ignored for TSR-WGAN because WGAN-GP is unstable under mixed precision in this path.")
    if is_main_process():
        if distributed:
            print(f"[INFO] Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"[INFO] Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders(
        cfg=cfg,
        seed=seed,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process():
        print(f"[INFO] train batches/rank={len(train_loader)}, val batches/rank={len(val_loader)}")

    metrics_cfg = cfg["metrics"]
    metric_computer = RestorationMetrics(
        compute_lpips=bool(metrics_cfg.get("compute_lpips", True)),
        lpips_net=str(metrics_cfg.get("lpips_net", "alex")),
        device=device,
    )

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    gan_model_cfg = model_cfg.get("gan", {}) if isinstance(model_cfg.get("gan", {}), dict) else {}
    num_frames = int(data_cfg.get("num_frames", 7))
    critic_clip_frames = int(gan_model_cfg.get("critic_clip_frames", 5))
    frame_indices = select_discriminator_frame_indices(frames_num=num_frames, clip_frames=critic_clip_frames)

    generator, critic = build_tsr_wgan_models(
        frames_num=num_frames,
        critic_clip_frames=critic_clip_frames,
        align_channels=int(gan_model_cfg.get("align_channels", 8)),
        base_channels=int(gan_model_cfg.get("generator_base_channels", 16)),
        critic_base_channels=int(gan_model_cfg.get("critic_base_channels", 64)),
        downsample_stages=int(gan_model_cfg.get("downsample_stages", 2)),
        num_resblocks=int(gan_model_cfg.get("num_resblocks", 9)),
        learn_residual=bool(gan_model_cfg.get("learn_residual", True)),
    )
    perceptual_loss_fn = PerceptualLoss().to(device)
    generator = generator.to(device)
    critic = critic.to(device)

    if distributed:
        generator = DDP(
            generator,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        critic = DDP(
            critic,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    def _unwrap(module: nn.Module) -> nn.Module:
        return module.module if isinstance(module, DDP) else module

    optimizer_cfg = cfg["optimizer"]
    optimizer_g = Adam(
        generator.parameters(),
        lr=float(optimizer_cfg.get("gan_lr_g", optimizer_cfg.get("learning_rate", 3e-4))),
        betas=(float(optimizer_cfg.get("gan_beta1", 0.9)), float(optimizer_cfg.get("gan_beta2", 0.999))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    optimizer_d = Adam(
        critic.parameters(),
        lr=float(optimizer_cfg.get("gan_lr_d", 1e-5)),
        betas=(float(optimizer_cfg.get("gan_beta1", 0.9)), float(optimizer_cfg.get("gan_beta2", 0.999))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    scheduler_g, scheduler_d, scheduler_mode = _build_schedulers(cfg=cfg, optimizer_g=optimizer_g, optimizer_d=optimizer_d)

    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    log_interval = int(train_cfg.get("log_interval", 20))
    val_interval = int(train_cfg.get("val_interval", 1))
    save_interval = int(train_cfg.get("save_interval", 5))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    train_metric_interval = max(0, int(train_cfg.get("train_metric_interval", 0)))
    train_metric_on_log = bool(train_cfg.get("train_metric_on_log", True))
    use_tqdm = bool(train_cfg.get("tqdm", True)) and is_main_process()
    fast_train = bool(train_cfg.get("fast_train", False))
    if fast_train:
        train_metric_on_log = bool(train_cfg.get("train_metric_on_log_fast", False))
        train_metric_interval = max(0, int(train_cfg.get("train_metric_interval_fast", 0)))

    ckpt_cfg = cfg["checkpoint"]
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor_name = str(ckpt_cfg.get("monitor", "psnr"))
    best_value = -1e9

    loss_cfg = cfg.get("loss", {})
    gan_loss_cfg = loss_cfg.get("gan", {}) if isinstance(loss_cfg.get("gan", {}), dict) else {}
    loss_weights = GANLossWeights(
        lambda_perceptual=float(gan_loss_cfg.get("lambda_perceptual", 10.0)),
        lambda_pixel=float(gan_loss_cfg.get("lambda_pixel", 10000.0)),
        lambda_gp=float(gan_loss_cfg.get("lambda_gp", 10.0)),
    )
    d_steps_per_g = int(gan_loss_cfg.get("d_steps_per_g", 1))

    model_type = str(model_cfg.get("type", "tsr_wgan")).lower().strip()
    if model_type not in {"tsr_wgan", "gan"}:
        if is_main_process():
            print(f"[WARN] train_gan.py is being used with model.type='{model_type}'. Continuing with TSR-WGAN training path.")

    for epoch in range(1, epochs + 1):
        if distributed and isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_gan(
            generator=generator,
            critic=critic,
            perceptual_loss_fn=perceptual_loss_fn,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            loss_weights=loss_weights,
            metrics=metric_computer,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
            metric_interval=train_metric_interval,
            metric_on_log=train_metric_on_log,
            d_steps_per_g=d_steps_per_g,
            frame_indices=frame_indices,
            use_tqdm=use_tqdm,
            cfg_for_adapt=cfg,
        )
        if distributed:
            train_stats = reduce_dict(train_stats, world_size=world_size)
        if is_main_process():
            print(f"[Epoch {epoch}] train={train_stats}")

        val_stats: dict[str, float] | None = None
        if val_interval > 0 and epoch % val_interval == 0:
            val_stats = evaluate_gan(
                generator=generator,
                critic=critic,
                perceptual_loss_fn=perceptual_loss_fn,
                loader=val_loader,
                loss_weights=loss_weights,
                metrics=metric_computer,
                device=device,
                frame_indices=frame_indices,
                cfg_for_adapt=cfg,
                distributed=distributed,
            )
            if is_main_process():
                print(f"[Epoch {epoch}] val={val_stats}")

            metric_value = float(val_stats.get(monitor_name, -1e9))
            if is_main_process() and metric_value > best_value:
                best_value = metric_value
                best_path = ckpt_dir / "best_tsr_wgan.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_type": "tsr_wgan",
                        "config": copy.deepcopy(cfg),
                        "generator_state": _unwrap(generator).state_dict(),
                        "critic_state": _unwrap(critic).state_dict(),
                        "optimizer_g_state": optimizer_g.state_dict(),
                        "optimizer_d_state": optimizer_d.state_dict(),
                        "scheduler_g_state": scheduler_g.state_dict() if scheduler_g is not None else None,
                        "scheduler_d_state": scheduler_d.state_dict() if scheduler_d is not None else None,
                        "val_stats": val_stats,
                    },
                    best_path,
                )
                print(f"[INFO] Saved new best checkpoint: {best_path} ({monitor_name}={metric_value:.4f})")

        if scheduler_g is not None and scheduler_d is not None:
            if scheduler_mode == "plateau":
                monitor_value = float((val_stats or train_stats).get(monitor_name, (val_stats or train_stats).get("g_total", 0.0)))
                scheduler_g.step(monitor_value)
                scheduler_d.step(monitor_value)
            else:
                scheduler_g.step()
                scheduler_d.step()

        if is_main_process() and save_interval > 0 and (epoch % save_interval == 0 or epoch == epochs):
            last_path = ckpt_dir / f"tsr_wgan_epoch_{epoch}.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_type": "tsr_wgan",
                    "config": copy.deepcopy(cfg),
                    "generator_state": _unwrap(generator).state_dict(),
                    "critic_state": _unwrap(critic).state_dict(),
                    "optimizer_g_state": optimizer_g.state_dict(),
                    "optimizer_d_state": optimizer_d.state_dict(),
                    "scheduler_g_state": scheduler_g.state_dict() if scheduler_g is not None else None,
                    "scheduler_d_state": scheduler_d.state_dict() if scheduler_d is not None else None,
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                },
                last_path,
            )
            print(f"[INFO] Saved checkpoint: {last_path}")


if __name__ == "__main__":
    main()