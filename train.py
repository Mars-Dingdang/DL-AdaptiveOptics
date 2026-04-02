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


def _resolve_entrypoint(model_type: str) -> Callable[[], None]:
    mt = model_type.lower().strip()
    if mt == "unet":
        return train_unet_main
    if mt == "gan":
        return train_gan_main
    if mt == "diffusion":
        return train_diffusion_main
    if mt == "vae":
        return train_vae_main
    raise ValueError(f"Unsupported model type: {model_type}. Expected 'unet', 'gan', 'diffusion', or 'vae'.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_type = str(cfg.get("model", {}).get("type", "unet"))
    entrypoint = _resolve_entrypoint(model_type)
    entrypoint()


if __name__ == "__main__":
    main()
