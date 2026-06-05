"""Compatibility training launcher.

This file keeps the historical `python train.py` interface while delegating
real training loops to dedicated scripts:
- train_unet.py
- train_gan.py
- train_diffusion.py
- train_vae.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
from typing import Callable

from train_common import load_config
from train_diffusion import main as train_diffusion_main
from train_gan import main as train_gan_main
from train_unet import main as train_unet_main
from train_vae import main as train_vae_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch to model-specific training entrypoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def _select_training_main(model_type: str) -> Callable[[], None]:
    mt = model_type.lower().strip()
    if mt == "unet":
        return train_unet_main
    if mt in {"gan", "tsr_wgan"}:
        return train_gan_main
    if mt == "diffusion":
        return train_diffusion_main
    if mt == "vae":
        return train_vae_main
    raise ValueError(f"Unsupported model type: {model_type}. Expected 'unet', 'gan', 'tsr_wgan', 'diffusion', or 'vae'.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_type = str(cfg.get("model", {}).get("type", "unet"))

    # Safety rail: only model types with explicit distributed support may run under torchrun.
    # If launched via torchrun with WORLD_SIZE > 1 against another model type,
    # fail fast rather than silently running redundant single-GPU training on
    # every rank (which would also race on checkpoint files).
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    if world_size > 1 and model_type.lower().strip() not in {"unet", "gan", "tsr_wgan"}:
        raise RuntimeError(
            f"Multi-GPU (DDP) training is currently only supported for model.type='unet', 'gan', or 'tsr_wgan', "
            f"but got model.type='{model_type}'. Either set model.type to 'unet', or launch "
            f"with a single process (e.g. plain `python train.py`) for other model types."
        )

    entrypoint = _select_training_main(model_type)
    entrypoint()


if __name__ == "__main__":
    main()
