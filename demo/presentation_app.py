"""Gallery-first Gradio presentation demo for turbulence restoration.

The app loads the single-frame and 7-frame U-Net checkpoints once, then serves
prepared examples and upload-based inference from one presentation-friendly UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.baseline_unet import build_baseline_unet
from train_common import adapt_degraded_for_model, load_config, resolve_cond_channels, resolve_device
from utils.metrics import batch_psnr_ssim


try:
    from gradio_imageslider import ImageSlider  # type: ignore

    HAS_IMAGE_SLIDER = True
except Exception:
    ImageSlider = None
    HAS_IMAGE_SLIDER = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_SINGLE_METRICS = {"psnr": 31.48, "ssim": 0.9581}
DEFAULT_SEQUENCE_METRICS = {"psnr": 35.37, "ssim": 0.9806}
APP_CSS = """
.app-title {font-size: 2rem; font-weight: 700; margin-bottom: 0.15rem;}
.app-subtitle {font-size: 1rem; opacity: 0.78; margin-bottom: 0.9rem;}
.metric-card {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px;}
.compact-note {font-size: 0.9rem; opacity: 0.78;}
"""


@dataclass(frozen=True)
class ExampleSample:
    """Prepared sequence sample discovered from disk."""

    name: str
    root: Path
    frames: tuple[Path, ...]
    clean: Path | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run presentation demo for single-frame and 7-frame U-Net models.")
    parser.add_argument(
        "--single-config",
        type=Path,
        default=PROJECT_ROOT / "configs/train_single_lmdb_center_tuned.yaml",
        help="Single-frame model config path.",
    )
    parser.add_argument(
        "--single-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/single_lmdb_center_tuned/best_unet.pt",
        help="Single-frame U-Net checkpoint path.",
    )
    parser.add_argument(
        "--sequence-config",
        type=Path,
        default=PROJECT_ROOT / "configs/default.yaml",
        help="7-frame stack model config path.",
    )
    parser.add_argument(
        "--sequence-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/best_unet.pt",
        help="7-frame U-Net checkpoint path.",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=PROJECT_ROOT / "data/turbulence_seq_nwpu_mild50_test",
        help="Folder containing sample_* prepared sequence examples.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device override: auto, cuda, cpu, or mps.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Launch host.")
    parser.add_argument("--port", type=int, default=7860, help="Launch port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    """Resolve CLI paths relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_to_multiple_of_16(image: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize one image for stable U-Net inference."""
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    new_h = max(16, int(round(h * scale)))
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, (new_h // 16) * 16)
    new_w = max(16, (new_w // 16) * 16)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    return resized, (h, w)


def _restore_original_size(image: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Resize model output back to the input display size."""
    oh, ow = original_hw
    return cv2.resize(image, (ow, oh), interpolation=cv2.INTER_CUBIC)


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HWC RGB uint8 image to BCHW float tensor in [0, 1]."""
    x = image.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)


def _from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert BCHW or CHW [0, 1] tensor to RGB uint8."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    image = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    image = np.transpose(image, (1, 2, 0))
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _metric_text(pred: np.ndarray, clean: np.ndarray | None, model_metrics: dict[str, float]) -> str:
    """Build a compact metrics markdown string."""
    if clean is None:
        return (
            f"Validation PSNR: **{model_metrics['psnr']:.2f} dB**  \n"
            f"Validation SSIM: **{model_metrics['ssim']:.4f}**"
        )

    if clean.shape[:2] != pred.shape[:2]:
        clean = cv2.resize(clean, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)

    pred_t = _to_tensor(pred, torch.device("cpu"))
    clean_t = _to_tensor(clean, torch.device("cpu"))
    psnr, ssim = batch_psnr_ssim(pred=pred_t, target=clean_t)
    return (
        f"Sample PSNR: **{psnr:.2f} dB**  \n"
        f"Sample SSIM: **{ssim:.4f}**  \n"
        f"Validation PSNR/SSIM: **{model_metrics['psnr']:.2f} / {model_metrics['ssim']:.4f}**"
    )


def discover_examples(examples_root: Path) -> list[ExampleSample]:
    """Discover prepared sample folders with seven frames."""
    root = _resolve_path(examples_root)
    if not root.exists():
        return []

    samples: list[ExampleSample] = []
    for sample_root in sorted(root.glob("sample_*")):
        if not sample_root.is_dir():
            continue
        frames = tuple(sample_root / f"frame_{idx:03d}.png" for idx in range(7))
        if not all(path.exists() for path in frames):
            continue
        clean = sample_root / "clean.png"
        samples.append(
            ExampleSample(
                name=sample_root.name,
                root=sample_root,
                frames=frames,
                clean=clean if clean.exists() else None,
            )
        )
    return samples


class UNetEngine:
    """U-Net model wrapper for one configured input strategy."""

    def __init__(self, *, name: str, config_path: Path, checkpoint_path: Path, device_override: str) -> None:
        self.name = name
        self.config_path = _resolve_path(config_path)
        self.checkpoint_path = _resolve_path(checkpoint_path)
        self.cfg = load_config(self.config_path)
        if device_override:
            self.cfg.setdefault("runtime", {})["device"] = device_override
        self.device = resolve_device(str(self.cfg["runtime"].get("device", "auto")))

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state")
        if state is None:
            raise RuntimeError(f"{name} checkpoint is missing model_state.")

        model_cfg = self.cfg["model"]
        self.cond_channels = resolve_cond_channels(self.cfg)
        self.model = build_baseline_unet(
            in_channels=self.cond_channels,
            out_channels=int(model_cfg.get("out_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 64)),
        ).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.epoch = ckpt.get("epoch", "unknown")

    @torch.no_grad()
    def infer_single(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run a single-image model on one RGB image."""
        image_u8 = image_rgb.astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb
        resized, original_hw = _resize_to_multiple_of_16(image_u8)
        pred = self.model(_to_tensor(resized, self.device)).clamp(0.0, 1.0)
        return _restore_original_size(_from_tensor(pred), original_hw)

    @torch.no_grad()
    def infer_sequence(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        """Run a sequence model on exactly seven RGB frames."""
        if len(frames_rgb) != 7:
            raise ValueError("7-frame inference requires exactly seven frames.")

        center = frames_rgb[len(frames_rgb) // 2]
        center_u8 = center.astype(np.uint8) if center.dtype != np.uint8 else center
        center_resized, original_hw = _resize_to_multiple_of_16(center_u8)
        target_h, target_w = center_resized.shape[:2]

        resized_frames = [
            cv2.resize(frame.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_AREA)
            for frame in frames_rgb
        ]
        frame_tensors = [_to_tensor(frame, self.device).squeeze(0) for frame in resized_frames]
        degraded = torch.stack(frame_tensors, dim=0).unsqueeze(0)
        degraded_model = adapt_degraded_for_model(degraded=degraded, cfg=self.cfg)
        pred = self.model(degraded_model).clamp(0.0, 1.0)
        return _restore_original_size(_from_tensor(pred), original_hw)


class DemoState:
    """Shared state and UI callbacks for the presentation app."""

    def __init__(self, *, single_engine: UNetEngine, sequence_engine: UNetEngine, examples: list[ExampleSample]) -> None:
        self.single_engine = single_engine
        self.sequence_engine = sequence_engine
        self.examples = {sample.name: sample for sample in examples}
        self.example_names = list(self.examples)

    def _get_sample(self, sample_name: str | None) -> ExampleSample:
        if not sample_name:
            raise gr.Error("Select a prepared example.")
        sample = self.examples.get(sample_name)
        if sample is None:
            raise gr.Error(f"Prepared example not found: {sample_name}")
        return sample

    def preview_example(self, sample_name: str | None) -> tuple[list[str], np.ndarray | None, np.ndarray | None]:
        """Return frame gallery, center frame, and clean target for a selected prepared sample."""
        if not sample_name:
            return [], None, None
        sample = self._get_sample(sample_name)
        center = _read_rgb(sample.frames[3])
        clean = _read_rgb(sample.clean) if sample.clean is not None else None
        return [str(path) for path in sample.frames], center, clean

    def run_prepared_single(self, sample_name: str | None) -> tuple[np.ndarray, str]:
        """Run single-frame inference on the center frame of a prepared sample."""
        sample = self._get_sample(sample_name)
        center = _read_rgb(sample.frames[3])
        clean = _read_rgb(sample.clean) if sample.clean is not None else None
        restored = self.single_engine.infer_single(center)
        return restored, _metric_text(restored, clean, DEFAULT_SINGLE_METRICS)

    def run_prepared_sequence(self, sample_name: str | None) -> tuple[np.ndarray, str]:
        """Run 7-frame inference on a prepared sample."""
        sample = self._get_sample(sample_name)
        frames = [_read_rgb(path) for path in sample.frames]
        clean = _read_rgb(sample.clean) if sample.clean is not None else None
        restored = self.sequence_engine.infer_sequence(frames)
        return restored, _metric_text(restored, clean, DEFAULT_SEQUENCE_METRICS)

    def run_prepared_both(self, sample_name: str | None) -> tuple[np.ndarray, str, np.ndarray, str]:
        """Run both models on a prepared sample."""
        single = self.run_prepared_single(sample_name)
        sequence = self.run_prepared_sequence(sample_name)
        return single + sequence

    def preview_uploads(
        self,
        mode: str,
        single_image: np.ndarray | None,
        sequence_files: list[str] | None,
    ) -> tuple[list[str], np.ndarray | None]:
        """Preview the current upload selection."""
        if mode == "Single Image":
            return [], single_image
        paths = _validate_sequence_files(sequence_files)
        center = _read_rgb(paths[3]) if paths else None
        return [str(path) for path in paths], center

    def run_upload_single(self, image: np.ndarray | None) -> tuple[np.ndarray, str]:
        """Run single-frame inference from an uploaded image."""
        if image is None:
            raise gr.Error("Upload one image for single-frame inference.")
        image_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
        restored = self.single_engine.infer_single(image_u8)
        return restored, _metric_text(restored, None, DEFAULT_SINGLE_METRICS)

    def run_upload_sequence(self, sequence_files: list[str] | None) -> tuple[np.ndarray, str]:
        """Run 7-frame inference from uploaded files."""
        paths = _validate_sequence_files(sequence_files)
        frames = [_read_rgb(path) for path in paths]
        restored = self.sequence_engine.infer_sequence(frames)
        return restored, _metric_text(restored, None, DEFAULT_SEQUENCE_METRICS)


def _validate_sequence_files(sequence_files: list[str] | None) -> list[Path]:
    """Validate and sort uploaded sequence files."""
    if not sequence_files:
        return []

    paths = sorted(Path(path) for path in sequence_files)
    bad = [path.name for path in paths if path.suffix.lower() not in IMAGE_EXTENSIONS]
    if bad:
        raise gr.Error(f"Unsupported image file: {bad[0]}")
    if len(paths) != 7:
        raise gr.Error(f"Upload exactly seven frames. Received {len(paths)}.")
    return paths


def _model_card(single_engine: UNetEngine, sequence_engine: UNetEngine) -> str:
    """Build static model summary markdown."""
    return (
        "### Model Summary\n"
        f"**Single-frame U-Net**: epoch `{single_engine.epoch}`, validation PSNR "
        f"`{DEFAULT_SINGLE_METRICS['psnr']:.2f}`, SSIM `{DEFAULT_SINGLE_METRICS['ssim']:.4f}`  \n"
        f"**7-frame U-Net**: epoch `{sequence_engine.epoch}`, validation PSNR "
        f"`{DEFAULT_SEQUENCE_METRICS['psnr']:.2f}`, SSIM `{DEFAULT_SEQUENCE_METRICS['ssim']:.4f}`  \n"
        f"**Device**: `{sequence_engine.device}`"
    )


def _result_component(label: str):
    """Create a before/after result component."""
    if HAS_IMAGE_SLIDER and ImageSlider is not None:
        return ImageSlider(label=label, type="numpy")
    return gr.Image(type="numpy", label=label)


def build_app(state: DemoState) -> gr.Blocks:
    """Build the Gradio presentation UI."""
    with gr.Blocks(title="Computational Adaptive Optics Demo") as demo:
        gr.HTML('<div class="app-title">Computational Adaptive Optics Demo</div>')
        gr.HTML('<div class="app-subtitle">Remote-sensing turbulence restoration with single-frame and 7-frame U-Nets.</div>')

        with gr.Tabs():
            with gr.Tab("Prepared Examples"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        example_dropdown = gr.Dropdown(
                            choices=state.example_names,
                            value=state.example_names[0] if state.example_names else None,
                            label="Example",
                            interactive=True,
                        )
                        with gr.Row():
                            prepared_single_btn = gr.Button("Run Single-Frame", variant="secondary")
                            prepared_sequence_btn = gr.Button("Run 7-Frame", variant="primary")
                        prepared_both_btn = gr.Button("Run Both")
                        gr.Markdown(_model_card(state.single_engine, state.sequence_engine), elem_classes="metric-card")

                    with gr.Column(scale=3, min_width=520):
                        prepared_strip = gr.Gallery(
                            label="7-frame input",
                            columns=7,
                            rows=1,
                            height=130,
                            object_fit="contain",
                            allow_preview=True,
                        )
                        gr.Markdown("### Inference results")
                        with gr.Row():
                            prepared_center = gr.Image(type="numpy", label="Input center frame", height=260)
                            prepared_single_result = gr.Image(type="numpy", label="Generated: single-frame U-Net", height=260)
                            prepared_sequence_result = gr.Image(type="numpy", label="Generated: 7-frame U-Net", height=260)
                            prepared_clean = gr.Image(type="numpy", label="Clean target", height=260)
                        with gr.Row():
                            prepared_single_metrics = gr.Markdown()
                            prepared_sequence_metrics = gr.Markdown()

            with gr.Tab("Upload"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        upload_mode = gr.Radio(
                            choices=["Single Image", "7-Frame Sequence"],
                            value="Single Image",
                            label="Mode",
                            interactive=True,
                        )
                        upload_single = gr.Image(type="numpy", label="Single image")
                        upload_sequence = gr.Files(
                            label="Seven ordered frames",
                            file_types=["image"],
                            type="filepath",
                            visible=False,
                        )
                        with gr.Row():
                            upload_single_btn = gr.Button("Run Single-Frame", variant="primary")
                            upload_sequence_btn = gr.Button("Run 7-Frame", variant="primary", visible=False)
                    with gr.Column(scale=2, min_width=360):
                        upload_strip = gr.Gallery(
                            label="Uploaded sequence",
                            columns=7,
                            rows=1,
                            height=130,
                            object_fit="contain",
                            allow_preview=True,
                            visible=False,
                        )
                        upload_center = gr.Image(type="numpy", label="Input preview")
                        upload_result = gr.Image(type="numpy", label="Generated output", height=320)
                        upload_metrics = gr.Markdown(_model_card(state.single_engine, state.sequence_engine))

            with gr.Tab("About Models"):
                gr.Markdown(
                    "### Checkpoints\n"
                    "`checkpoints/best_unet.pt` restores a 7-frame sequence by stacking channels.  \n"
                    "`checkpoints/single_lmdb_center_tuned/best_unet.pt` restores the center frame only.\n\n"
                    "### Presentation Metrics\n"
                    f"7-frame validation PSNR/SSIM: **{DEFAULT_SEQUENCE_METRICS['psnr']:.2f} / "
                    f"{DEFAULT_SEQUENCE_METRICS['ssim']:.4f}**  \n"
                    f"single-frame validation PSNR/SSIM: **{DEFAULT_SINGLE_METRICS['psnr']:.2f} / "
                    f"{DEFAULT_SINGLE_METRICS['ssim']:.4f}**"
                )

        def switch_upload_mode(mode: str):
            is_sequence = mode == "7-Frame Sequence"
            return (
                gr.update(visible=not is_sequence),
                gr.update(visible=is_sequence),
                gr.update(visible=not is_sequence),
                gr.update(visible=is_sequence),
                gr.update(visible=is_sequence),
            )

        example_change = example_dropdown.change(
            fn=state.preview_example,
            inputs=example_dropdown,
            outputs=[prepared_strip, prepared_center, prepared_clean],
        )
        example_change.then(
            fn=state.run_prepared_both,
            inputs=example_dropdown,
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        initial_load = demo.load(
            fn=state.preview_example,
            inputs=example_dropdown,
            outputs=[prepared_strip, prepared_center, prepared_clean],
        )
        initial_load.then(
            fn=state.run_prepared_both,
            inputs=example_dropdown,
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        prepared_single_btn.click(
            fn=state.run_prepared_single,
            inputs=example_dropdown,
            outputs=[prepared_single_result, prepared_single_metrics],
        )
        prepared_sequence_btn.click(
            fn=state.run_prepared_sequence,
            inputs=example_dropdown,
            outputs=[prepared_sequence_result, prepared_sequence_metrics],
        )
        prepared_both_btn.click(
            fn=state.run_prepared_both,
            inputs=example_dropdown,
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        upload_mode.change(
            fn=switch_upload_mode,
            inputs=upload_mode,
            outputs=[upload_single, upload_sequence, upload_single_btn, upload_sequence_btn, upload_strip],
        )
        upload_single.change(
            fn=lambda image: ([], image),
            inputs=upload_single,
            outputs=[upload_strip, upload_center],
        )
        upload_sequence.change(
            fn=lambda files: state.preview_uploads("7-Frame Sequence", None, files),
            inputs=upload_sequence,
            outputs=[upload_strip, upload_center],
        )
        upload_single_btn.click(
            fn=state.run_upload_single,
            inputs=upload_single,
            outputs=[upload_result, upload_metrics],
        )
        upload_sequence_btn.click(
            fn=state.run_upload_sequence,
            inputs=upload_sequence,
            outputs=[upload_result, upload_metrics],
        )

    return demo


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    single_engine = UNetEngine(
        name="single-frame",
        config_path=args.single_config,
        checkpoint_path=args.single_checkpoint,
        device_override=args.device,
    )
    sequence_engine = UNetEngine(
        name="7-frame",
        config_path=args.sequence_config,
        checkpoint_path=args.sequence_checkpoint,
        device_override=args.device,
    )
    examples = discover_examples(args.examples_root)
    if not examples:
        print(f"[WARN] No prepared examples found under {_resolve_path(args.examples_root)}")
    print(f"[INFO] Loaded {len(examples)} prepared examples.")
    print(f"[INFO] Single-frame model device: {single_engine.device}")
    print(f"[INFO] 7-frame model device: {sequence_engine.device}")

    app = build_app(DemoState(single_engine=single_engine, sequence_engine=sequence_engine, examples=examples))
    app.launch(
        server_name=args.host,
        server_port=int(args.port),
        share=bool(args.share),
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="amber"),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
