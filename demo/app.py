"""Gradio demo for turbulence removal.

This app loads a trained checkpoint and provides an interactive interface:
- Upload degraded image.
- Click restore button.
- Compare before/after with an image slider (with fallback).

Usage:
    python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt
    python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_gan.pt --model-type gan
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.baseline_unet import build_baseline_unet
from modules.gan_models import build_pix2pix_models
from train import load_config, resolve_device, to_0_1, to_minus1_1


try:
    from gradio_imageslider import ImageSlider  # type: ignore

    HAS_IMAGE_SLIDER = True
except Exception:
    ImageSlider = None
    HAS_IMAGE_SLIDER = False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Gradio demo for turbulence restoration.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml", help="Config path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--model-type",
        type=str,
        default="",
        choices=["", "unet", "gan"],
        help="Optional model type override.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Launch host")
    parser.add_argument("--port", type=int, default=7860, help="Launch port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()


def infer_model_type(ckpt: dict[str, Any], cfg: dict[str, Any], arg_type: str) -> str:
    """Infer model type with priority: CLI arg > checkpoint > config."""
    if arg_type:
        return arg_type
    ckpt_type = str(ckpt.get("model_type", "")).lower().strip()
    if ckpt_type in {"unet", "gan"}:
        return ckpt_type
    return str(cfg["model"]["type"]).lower().strip()


def _resize_to_multiple_of_16(image: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize image for stable model inference while preserving aspect ratio."""
    h, w = image.shape[:2]

    scale = 1.0
    long_side = max(h, w)
    if long_side > max_side:
        scale = float(max_side) / float(long_side)

    new_h = max(16, int(round(h * scale)))
    new_w = max(16, int(round(w * scale)))

    # U-Net/GAN downsample path expects spatial sizes compatible with /16.
    new_h = (new_h // 16) * 16
    new_w = (new_w // 16) * 16
    new_h = max(16, new_h)
    new_w = max(16, new_w)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    return resized, (h, w)


def _restore_original_size(image: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Resize restored image back to original size."""
    oh, ow = original_hw
    return cv2.resize(image, (ow, oh), interpolation=cv2.INTER_CUBIC)


class InferenceEngine:
    """Model loader and inference pipeline."""

    def __init__(self, config_path: Path, checkpoint_path: Path, model_type_override: str = "") -> None:
        self.cfg = load_config(config_path)
        self.device = resolve_device(str(self.cfg["runtime"].get("device", "auto")))

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model_type = infer_model_type(ckpt=ckpt, cfg=self.cfg, arg_type=model_type_override)

        model_cfg = self.cfg["model"]
        in_channels = int(model_cfg.get("in_channels", 3))
        out_channels = int(model_cfg.get("out_channels", 3))
        base_channels = int(model_cfg.get("base_channels", 64))

        if self.model_type == "unet":
            self.model = build_baseline_unet(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
            ).to(self.device)
            state = ckpt.get("model_state")
            if state is None:
                raise RuntimeError("Checkpoint missing 'model_state' for U-Net model.")
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            self.generator = None

        elif self.model_type == "gan":
            generator, _disc = build_pix2pix_models(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
            )
            generator = generator.to(self.device)
            state = ckpt.get("generator_state")
            if state is None:
                raise RuntimeError("Checkpoint missing 'generator_state' for GAN model.")
            generator.load_state_dict(state, strict=True)
            generator.eval()
            self.model = None
            self.generator = generator

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        print(f"[INFO] Loaded model type: {self.model_type} on {self.device}")

    @torch.no_grad()
    def infer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run restoration inference and return RGB uint8 image."""
        if image_rgb is None:
            raise ValueError("Input image is None")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input must be an RGB image (H, W, 3)")

        image_u8 = image_rgb.astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb
        resized, original_hw = _resize_to_multiple_of_16(image_u8)

        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x_t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(self.device)

        if self.model_type == "unet":
            assert self.model is not None
            pred = self.model(x_t).clamp(0.0, 1.0)
        else:
            assert self.generator is not None
            pred_n = self.generator(to_minus1_1(x_t))
            pred = to_0_1(pred_n)

        pred_np = pred.squeeze(0).detach().cpu().numpy()
        pred_np = np.transpose(pred_np, (1, 2, 0))
        pred_np = (np.clip(pred_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        pred_np = _restore_original_size(pred_np, original_hw=original_hw)
        return pred_np


def build_app(engine: InferenceEngine) -> gr.Blocks:
    """Build Gradio UI."""
    css = """
    .main-title {text-align:center; font-size: 2rem; font-weight: 700; margin-bottom: 0.4rem;}
    .sub-title {text-align:center; opacity: 0.85; margin-bottom: 1rem;}
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="teal", secondary_hue="amber")) as demo:
        gr.HTML('<div class="main-title">Computational Adaptive Optics Demo</div>')
        gr.HTML('<div class="sub-title">Upload a degraded remote-sensing image and restore it with a trained model.</div>')

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Input (Degraded RGB Image)")
                btn = gr.Button("Remove Turbulence", variant="primary")
                model_info = gr.Markdown(
                    f"**Model**: {engine.model_type.upper()}  |  **Device**: {str(engine.device)}"
                )

            with gr.Column(scale=1):
                if HAS_IMAGE_SLIDER and ImageSlider is not None:
                    out_slider = ImageSlider(label="Before / After", type="numpy")
                    out_before = None
                    out_after = None
                else:
                    out_slider = None
                    out_before = gr.Image(type="numpy", label="Before")
                    out_after = gr.Image(type="numpy", label="After")
                    gr.Markdown(
                        "`gradio-imageslider` not found. Showing side-by-side images as fallback."
                    )

        def run_restore(image: np.ndarray | None):
            if image is None:
                raise gr.Error("Please upload an image first.")

            image_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
            restored_u8 = engine.infer(image_u8)

            if out_slider is not None:
                return (image_u8, restored_u8)
            return image_u8, restored_u8

        if HAS_IMAGE_SLIDER and ImageSlider is not None and out_slider is not None:
            btn.click(fn=run_restore, inputs=inp, outputs=out_slider)
        else:
            assert out_before is not None and out_after is not None
            btn.click(fn=run_restore, inputs=inp, outputs=[out_before, out_after])

        _ = model_info

    return demo


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    engine = InferenceEngine(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_type_override=args.model_type,
    )

    app = build_app(engine)
    app.launch(server_name=args.host, server_port=int(args.port), share=bool(args.share))


if __name__ == "__main__":
    main()
