"""GIF-first Gradio presentation demo for turbulence restoration.

The app loads the single-frame and 7-frame U-Net checkpoints once, then serves
prepared examples and upload-based inference from one presentation-friendly UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import tempfile
from typing import Any

import cv2
import gradio as gr
import lmdb
import numpy as np
from PIL import Image, ImageSequence
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.baseline_unet import build_baseline_unet
from train_common import adapt_degraded_for_model, load_config, resolve_cond_channels, resolve_device
from utils.metrics import batch_psnr_ssim
from data.dataset import SequenceDatasetParams, TurbulenceSequenceLmdbDataset


try:
    from gradio_imageslider import ImageSlider  # type: ignore

    HAS_IMAGE_SLIDER = True
except Exception:
    ImageSlider = None
    HAS_IMAGE_SLIDER = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
GIF_EXTENSIONS = {".gif"}
MODE_METRICS = {
    "0.5 turbulence": {
        "single": {"psnr": 31.48, "ssim": 0.9581},
        "sequence": {"psnr": 35.37, "ssim": 0.9806},
    },
    "0.75 turbulence": {
        "single": {"psnr": 29.46, "ssim": 0.9256},
        "sequence": {"psnr": 31.78, "ssim": 0.9557},
    },
}
MODE_LABELS = tuple(MODE_METRICS)
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
    gif: Path | None
    clean: Path | None


@dataclass(frozen=True)
class ModeBundle:
    """All runtime assets for one turbulence strength."""

    label: str
    single_engine: "UNetEngine"
    sequence_engine: "UNetEngine"
    examples: dict[str, ExampleSample]
    single_metrics: dict[str, float]
    sequence_metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run presentation demo for single-frame and 7-frame U-Net models.")
    parser.add_argument(
        "--mild50-single-config",
        type=Path,
        default=PROJECT_ROOT / "configs/train_single_lmdb_center_tuned.yaml",
        help="0.5 turbulence single-frame model config path.",
    )
    parser.add_argument(
        "--mild50-single-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/single_lmdb_center_tuned/best_unet.pt",
        help="0.5 turbulence single-frame U-Net checkpoint path.",
    )
    parser.add_argument(
        "--mild50-sequence-config",
        type=Path,
        default=PROJECT_ROOT / "configs/eval_lmdb_stack_mild50.yaml",
        help="0.5 turbulence 7-frame stack model config path.",
    )
    parser.add_argument(
        "--mild50-sequence-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/best_unet.pt",
        help="0.5 turbulence 7-frame U-Net checkpoint path.",
    )
    parser.add_argument(
        "--mild50-demo-examples",
        type=Path,
        default=PROJECT_ROOT / "demo/examples/mild50",
        help="Tracked GIF examples for 0.5 turbulence.",
    )
    parser.add_argument(
        "--mild50-lmdb-root",
        type=Path,
        default=PROJECT_ROOT / "data/turbulence_seq_nwpu_mild50_lmdb",
        help="LMDB root used to materialize prepared GIF examples for 0.5 turbulence.",
    )
    parser.add_argument(
        "--mild75-single-config",
        type=Path,
        default=PROJECT_ROOT / "configs/train_single_mild75.yaml",
        help="0.75 turbulence single-frame model config path.",
    )
    parser.add_argument(
        "--mild75-single-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/single_mild75/best_unet.pt",
        help="0.75 turbulence single-frame U-Net checkpoint path.",
    )
    parser.add_argument(
        "--mild75-sequence-config",
        type=Path,
        default=PROJECT_ROOT / "configs/mild75.yaml",
        help="0.75 turbulence 7-frame stack model config path.",
    )
    parser.add_argument(
        "--mild75-sequence-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/mild75/best_unet.pt",
        help="0.75 turbulence 7-frame U-Net checkpoint path.",
    )
    parser.add_argument(
        "--mild75-demo-examples",
        type=Path,
        default=PROJECT_ROOT / "demo/examples/mild75",
        help="Tracked GIF examples for 0.75 turbulence.",
    )
    parser.add_argument(
        "--mild75-lmdb-root",
        type=Path,
        default=PROJECT_ROOT / "data/turbulence_seq_nwpu_mild75_lmdb",
        help="LMDB root used to materialize prepared GIF examples for 0.75 turbulence.",
    )
    parser.add_argument(
        "--raw-categories-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/NWPU-RESISC45",
        help="Raw NWPU category root used to label LMDB examples by exact image match.",
    )
    parser.add_argument("--examples-per-mode", type=int, default=45, help="Prepared examples to expose per mode.")
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


def _read_gif_frames(path: Path, *, target_frames: int = 7) -> list[np.ndarray]:
    """Read a GIF as exactly target_frames RGB frames."""
    with Image.open(path) as image:
        frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in ImageSequence.Iterator(image)]

    if not frames:
        raise gr.Error("The uploaded GIF has no readable frames.")
    if len(frames) < target_frames:
        raise gr.Error(f"Upload a GIF with at least {target_frames} frames. Received {len(frames)}.")
    if len(frames) == target_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, target_frames).round().astype(int).tolist()
    return [frames[idx] for idx in indices]


def _write_sequence_gif(frames: list[np.ndarray], path: Path, *, duration_ms: int = 220) -> None:
    """Write RGB frames to an animated GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _tensor_image_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW [0, 1] tensor to HWC uint8."""
    image = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    image = np.transpose(image, (1, 2, 0))
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


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
    """Discover prepared example folders from tracked GIFs or frame PNGs."""
    root = _resolve_path(examples_root)
    if not root.exists():
        return []

    samples: list[ExampleSample] = []
    for sample_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if not sample_root.is_dir():
            continue
        frames = tuple(sample_root / f"frame_{idx:03d}.png" for idx in range(7))
        gif = sample_root / "turbulence.gif"
        has_frames = all(path.exists() for path in frames)
        if not has_frames and not gif.exists():
            continue
        clean = sample_root / "clean.png"
        samples.append(
            ExampleSample(
                name=sample_root.name.replace("_", " "),
                root=sample_root,
                frames=frames if has_frames else tuple(),
                gif=gif if gif.exists() else None,
                clean=clean if clean.exists() else None,
            )
        )
    return samples


def _image_hash(image: np.ndarray) -> str:
    """Hash decoded RGB pixel content."""
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


def _raw_hash_cache_path(raw_root: Path) -> Path:
    """Return a local cache path for decoded raw-image category hashes."""
    key = hashlib.sha1(str(raw_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "dl_adaptive_optics_demo" / f"raw_hash_categories_{key}.json"


def _load_raw_category_hashes(raw_root: Path) -> dict[str, str]:
    """Map decoded raw NWPU image hashes to category names."""
    root = _resolve_path(raw_root)
    if not root.exists():
        return {}

    cache_path = _raw_hash_cache_path(root)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("root") == str(root.resolve()):
            return {str(k): str(v) for k, v in cached.get("hash_to_category", {}).items()}

    hash_to_category: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = _read_rgb(path)
        hash_to_category[_image_hash(image)] = path.parent.name

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "root": str(root.resolve()),
                "hash_to_category": hash_to_category,
            },
            handle,
            ensure_ascii=True,
        )
    return hash_to_category


def _select_lmdb_indices_by_category(
    lmdb_root: Path,
    raw_hash_categories: dict[str, str],
    *,
    limit: int,
) -> list[tuple[int, str]]:
    """Select one LMDB sample index for each raw NWPU category."""
    if not raw_hash_categories:
        return []

    env = lmdb.open(
        str(lmdb_root),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    selected: dict[str, int] = {}
    with env.begin(write=False) as txn:
        raw_len = txn.get(b"__len__")
        if raw_len is None:
            return []
        total = int(raw_len.decode("utf-8"))
        for idx in range(total):
            clean_raw = txn.get(f"sample-{idx:07d}-clean".encode("utf-8"))
            if clean_raw is None:
                continue
            clean = np.asarray(Image.open(io.BytesIO(clean_raw)).convert("RGB"), dtype=np.uint8)
            category = raw_hash_categories.get(_image_hash(clean))
            if category and category not in selected:
                selected[category] = idx
                if limit > 0 and len(selected) >= limit:
                    break

    return [(idx, category) for category, idx in sorted(selected.items(), key=lambda item: item[0])]


def materialize_lmdb_examples(
    lmdb_root: Path,
    cache_root: Path,
    *,
    limit: int,
    raw_hash_categories: dict[str, str] | None = None,
) -> list[ExampleSample]:
    """Create prepared GIF examples from an LMDB sequence dataset."""
    root = _resolve_path(lmdb_root)
    if not root.exists():
        return []

    params = SequenceDatasetParams(
        image_size=256,
        num_frames=7,
        random_crop=False,
        horizontal_flip_prob=0.0,
    )
    dataset = TurbulenceSequenceLmdbDataset(lmdb_root=root, dataset_params=params, seed=42)
    selected = _select_lmdb_indices_by_category(
        root,
        raw_hash_categories or {},
        limit=int(limit),
    )
    if not selected:
        count = min(max(0, int(limit)), len(dataset))
        selected = [(idx, f"sample_{idx:07d}") for idx in range(count)]

    samples: list[ExampleSample] = []
    for idx, category in selected:
        safe_category = category.replace("/", "_").replace(" ", "_")
        sample_root = cache_root / f"{safe_category}_{idx:07d}"
        sample_root.mkdir(parents=True, exist_ok=True)
        frame_paths = tuple(sample_root / f"frame_{frame_idx:03d}.png" for frame_idx in range(7))
        clean_path = sample_root / "clean.png"
        gif_path = sample_root / "turbulence.gif"

        if not (gif_path.exists() and clean_path.exists() and all(path.exists() for path in frame_paths)):
            seq_tensor, clean_tensor = dataset[idx]
            frames = [_tensor_image_to_uint8(seq_tensor[frame_idx]) for frame_idx in range(seq_tensor.shape[0])]
            clean = _tensor_image_to_uint8(clean_tensor)
            for frame, frame_path in zip(frames, frame_paths):
                Image.fromarray(frame, mode="RGB").save(frame_path)
            Image.fromarray(clean, mode="RGB").save(clean_path)
            _write_sequence_gif(frames, gif_path)

        samples.append(
            ExampleSample(
                name=category.replace("_", " "),
                root=sample_root,
                frames=frame_paths,
                gif=gif_path,
                clean=clean_path,
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

    def __init__(self, *, bundles: dict[str, ModeBundle]) -> None:
        self.bundles = bundles

    def _get_bundle(self, mode: str | None) -> ModeBundle:
        if not mode:
            raise gr.Error("Select a turbulence mode.")
        bundle = self.bundles.get(mode)
        if bundle is None:
            raise gr.Error(f"Turbulence mode not found: {mode}")
        return bundle

    def example_names(self, mode: str | None) -> list[str]:
        """Return prepared example names for one mode."""
        return list(self._get_bundle(mode).examples)

    def switch_mode(self, mode: str | None) -> str:
        """Update model card when turbulence mode changes."""
        bundle = self._get_bundle(mode)
        return _model_card(bundle)

    def _get_sample(self, mode: str | None, sample_name: str | None) -> ExampleSample:
        if not sample_name:
            raise gr.Error("Select a prepared example.")
        bundle = self._get_bundle(mode)
        sample = bundle.examples.get(sample_name)
        if sample is None:
            raise gr.Error(f"Prepared example not found: {sample_name}")
        return sample

    def _sample_frames(self, sample: ExampleSample) -> list[np.ndarray]:
        """Read seven frames from a prepared example."""
        if sample.frames:
            return [_read_rgb(path) for path in sample.frames]
        if sample.gif is not None:
            return _read_gif_frames(sample.gif)
        raise gr.Error(f"Prepared example has no frames or GIF: {sample.name}")

    def preview_example(
        self,
        mode: str | None,
        sample_name: str | None,
    ) -> tuple[str | None, np.ndarray | None, np.ndarray | None]:
        """Return GIF, center frame, and clean target for a selected sample."""
        if not sample_name:
            return None, None, None
        sample = self._get_sample(mode, sample_name)
        frames = self._sample_frames(sample)
        center = frames[3]
        clean = _read_rgb(sample.clean) if sample.clean is not None else None
        return str(sample.gif) if sample.gif is not None else None, center, clean

    def run_prepared_single(self, mode: str | None, sample_name: str | None) -> tuple[np.ndarray, str]:
        """Run single-frame inference on the center frame of a prepared sample."""
        bundle = self._get_bundle(mode)
        sample = self._get_sample(mode, sample_name)
        center = self._sample_frames(sample)[3]
        clean = _read_rgb(sample.clean) if sample.clean is not None else None
        restored = bundle.single_engine.infer_single(center)
        return restored, _metric_text(restored, clean, bundle.single_metrics)

    def run_prepared_sequence(self, mode: str | None, sample_name: str | None) -> tuple[np.ndarray, str]:
        """Run 7-frame inference on a prepared sample."""
        bundle = self._get_bundle(mode)
        sample = self._get_sample(mode, sample_name)
        frames = self._sample_frames(sample)
        clean = _read_rgb(sample.clean) if sample.clean is not None else None
        restored = bundle.sequence_engine.infer_sequence(frames)
        return restored, _metric_text(restored, clean, bundle.sequence_metrics)

    def run_prepared_both(self, mode: str | None, sample_name: str | None) -> tuple[np.ndarray, str, np.ndarray, str]:
        """Run both models on a prepared sample."""
        single = self.run_prepared_single(mode, sample_name)
        sequence = self.run_prepared_sequence(mode, sample_name)
        return single + sequence

    def preview_gif_upload(self, gif_file: str | None) -> np.ndarray | None:
        """Preview the center frame of an uploaded GIF."""
        frames = _validate_gif_file(gif_file)
        return frames[3] if frames else None

    def run_upload_single(self, mode: str | None, gif_file: str | None) -> tuple[np.ndarray, str]:
        """Run single-frame inference from the center frame of an uploaded GIF."""
        bundle = self._get_bundle(mode)
        frames = _validate_gif_file(gif_file)
        restored = bundle.single_engine.infer_single(frames[3])
        return restored, _metric_text(restored, None, bundle.single_metrics)

    def run_upload_sequence(self, mode: str | None, gif_file: str | None) -> tuple[np.ndarray, str]:
        """Run 7-frame inference from an uploaded GIF."""
        bundle = self._get_bundle(mode)
        frames = _validate_gif_file(gif_file)
        restored = bundle.sequence_engine.infer_sequence(frames)
        return restored, _metric_text(restored, None, bundle.sequence_metrics)

    def run_upload_both(self, mode: str | None, gif_file: str | None) -> tuple[np.ndarray, str, np.ndarray, str]:
        """Run both models from one uploaded GIF."""
        single = self.run_upload_single(mode, gif_file)
        sequence = self.run_upload_sequence(mode, gif_file)
        return single + sequence


def _validate_gif_file(gif_file: str | None) -> list[np.ndarray]:
    """Validate an uploaded GIF and return seven RGB frames."""
    if not gif_file:
        return []
    path = Path(gif_file)
    if path.suffix.lower() not in GIF_EXTENSIONS:
        raise gr.Error("Upload one animated GIF containing the 7-frame degraded sequence.")
    return _read_gif_frames(path)


def _model_card(bundle: ModeBundle) -> str:
    """Build static model summary markdown."""
    return (
        "### Model Summary\n"
        f"**Mode**: `{bundle.label}`  \n"
        f"**Single-frame U-Net**: epoch `{bundle.single_engine.epoch}`, validation PSNR "
        f"`{bundle.single_metrics['psnr']:.2f}`, SSIM `{bundle.single_metrics['ssim']:.4f}`  \n"
        f"**7-frame U-Net**: epoch `{bundle.sequence_engine.epoch}`, validation PSNR "
        f"`{bundle.sequence_metrics['psnr']:.2f}`, SSIM `{bundle.sequence_metrics['ssim']:.4f}`  \n"
        f"**Device**: `{bundle.sequence_engine.device}`"
    )


def _result_component(label: str):
    """Create a before/after result component."""
    if HAS_IMAGE_SLIDER and ImageSlider is not None:
        return ImageSlider(label=label, type="numpy")
    return gr.Image(type="numpy", label=label)


def build_app(state: DemoState) -> gr.Blocks:
    """Build the Gradio presentation UI."""
    initial_mode = MODE_LABELS[0]
    initial_bundle = state.bundles[initial_mode]
    example_sets = [set(bundle.examples) for bundle in state.bundles.values()]
    all_examples = sorted(set.intersection(*example_sets)) if example_sets else []
    with gr.Blocks(title="Computational Adaptive Optics Demo") as demo:
        gr.HTML('<div class="app-title">Computational Adaptive Optics Demo</div>')
        gr.HTML('<div class="app-subtitle">Remote-sensing turbulence restoration from 7-frame GIF input.</div>')
        mode_dropdown = gr.Dropdown(
            choices=list(MODE_LABELS),
            value=initial_mode,
            label="Turbulence mode",
            interactive=True,
        )

        with gr.Tabs():
            with gr.Tab("Prepared Examples"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        example_dropdown = gr.Dropdown(
                            choices=all_examples,
                            value=all_examples[0] if all_examples else None,
                            label="Example",
                            interactive=True,
                        )
                        with gr.Row():
                            prepared_single_btn = gr.Button("Run Single-Frame", variant="secondary")
                            prepared_sequence_btn = gr.Button("Run 7-Frame", variant="primary")
                        prepared_both_btn = gr.Button("Run Both")
                        model_summary = gr.Markdown(_model_card(initial_bundle), elem_classes="metric-card")

                    with gr.Column(scale=3, min_width=520):
                        prepared_gif = gr.Image(
                            type="filepath",
                            label="7-frame degraded GIF input",
                            height=260,
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
                        upload_gif = gr.File(
                            label="7-frame degraded GIF",
                            file_types=[".gif"],
                            type="filepath",
                        )
                        upload_both_btn = gr.Button("Run Both", variant="primary")
                        upload_single_btn = gr.Button("Run Single-Frame", variant="secondary")
                        upload_sequence_btn = gr.Button("Run 7-Frame", variant="secondary")
                    with gr.Column(scale=2, min_width=360):
                        upload_center = gr.Image(type="numpy", label="Input preview")
                        with gr.Row():
                            upload_single_result = gr.Image(type="numpy", label="Generated: single-frame U-Net", height=300)
                            upload_sequence_result = gr.Image(type="numpy", label="Generated: 7-frame U-Net", height=300)
                        with gr.Row():
                            upload_single_metrics = gr.Markdown()
                            upload_sequence_metrics = gr.Markdown()

            with gr.Tab("About Models"):
                gr.Markdown(
                    "### Checkpoints\n"
                    "**0.5 turbulence** uses `checkpoints/best_unet.pt` for 7-frame inference and "
                    "`checkpoints/single_lmdb_center_tuned/best_unet.pt` for center-frame inference.  \n"
                    "**0.75 turbulence** uses `checkpoints/mild75/best_unet.pt` for 7-frame inference and "
                    "`checkpoints/single_mild75/best_unet.pt` for center-frame inference.\n\n"
                    "### Presentation Metrics\n"
                    f"0.5 single-frame PSNR/SSIM: **{MODE_METRICS['0.5 turbulence']['single']['psnr']:.2f} / "
                    f"{MODE_METRICS['0.5 turbulence']['single']['ssim']:.4f}**  \n"
                    f"0.5 7-frame PSNR/SSIM: **{MODE_METRICS['0.5 turbulence']['sequence']['psnr']:.2f} / "
                    f"{MODE_METRICS['0.5 turbulence']['sequence']['ssim']:.4f}**  \n"
                    f"0.75 single-frame PSNR/SSIM: **{MODE_METRICS['0.75 turbulence']['single']['psnr']:.2f} / "
                    f"{MODE_METRICS['0.75 turbulence']['single']['ssim']:.4f}**  \n"
                    f"0.75 7-frame PSNR/SSIM: **{MODE_METRICS['0.75 turbulence']['sequence']['psnr']:.2f} / "
                    f"{MODE_METRICS['0.75 turbulence']['sequence']['ssim']:.4f}**\n\n"
                    "### Prepared Examples\n"
                    "Prepared examples are selected by exact pixel match between each LMDB clean target and "
                    "the local NWPU category folders. The cloud category is not present in these LMDBs."
                )

        mode_change = mode_dropdown.change(
            fn=state.switch_mode,
            inputs=mode_dropdown,
            outputs=model_summary,
        )
        mode_change.then(
            fn=state.preview_example,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[prepared_gif, prepared_center, prepared_clean],
        )
        mode_change.then(
            fn=state.run_prepared_both,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        example_change = example_dropdown.change(
            fn=state.preview_example,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[prepared_gif, prepared_center, prepared_clean],
        )
        example_change.then(
            fn=state.run_prepared_both,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        initial_load = demo.load(
            fn=state.preview_example,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[prepared_gif, prepared_center, prepared_clean],
        )
        initial_load.then(
            fn=state.run_prepared_both,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        prepared_single_btn.click(
            fn=state.run_prepared_single,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[prepared_single_result, prepared_single_metrics],
        )
        prepared_sequence_btn.click(
            fn=state.run_prepared_sequence,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[prepared_sequence_result, prepared_sequence_metrics],
        )
        prepared_both_btn.click(
            fn=state.run_prepared_both,
            inputs=[mode_dropdown, example_dropdown],
            outputs=[
                prepared_single_result,
                prepared_single_metrics,
                prepared_sequence_result,
                prepared_sequence_metrics,
            ],
        )
        upload_gif.change(
            fn=state.preview_gif_upload,
            inputs=upload_gif,
            outputs=upload_center,
        )
        upload_single_btn.click(
            fn=state.run_upload_single,
            inputs=[mode_dropdown, upload_gif],
            outputs=[upload_single_result, upload_single_metrics],
        )
        upload_sequence_btn.click(
            fn=state.run_upload_sequence,
            inputs=[mode_dropdown, upload_gif],
            outputs=[upload_sequence_result, upload_sequence_metrics],
        )
        upload_both_btn.click(
            fn=state.run_upload_both,
            inputs=[mode_dropdown, upload_gif],
            outputs=[
                upload_single_result,
                upload_single_metrics,
                upload_sequence_result,
                upload_sequence_metrics,
            ],
        )

    return demo


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    mild50_single = UNetEngine(
        name="0.5 single-frame",
        config_path=args.mild50_single_config,
        checkpoint_path=args.mild50_single_checkpoint,
        device_override=args.device,
    )
    mild50_sequence = UNetEngine(
        name="0.5 7-frame",
        config_path=args.mild50_sequence_config,
        checkpoint_path=args.mild50_sequence_checkpoint,
        device_override=args.device,
    )
    mild75_single = UNetEngine(
        name="0.75 single-frame",
        config_path=args.mild75_single_config,
        checkpoint_path=args.mild75_single_checkpoint,
        device_override=args.device,
    )
    mild75_sequence = UNetEngine(
        name="0.75 7-frame",
        config_path=args.mild75_sequence_config,
        checkpoint_path=args.mild75_sequence_checkpoint,
        device_override=args.device,
    )

    raw_hash_categories: dict[str, str] | None = None

    mild50_examples = discover_examples(args.mild50_demo_examples)
    if not mild50_examples:
        raw_hash_categories = _load_raw_category_hashes(args.raw_categories_root)
        if not raw_hash_categories:
            print(f"[WARN] No raw category mapping loaded from {_resolve_path(args.raw_categories_root)}")
        mild50_cache_root = Path(tempfile.gettempdir()) / "dl_adaptive_optics_demo" / "mild50_raw_categories_v2"
        mild50_examples = materialize_lmdb_examples(
            args.mild50_lmdb_root,
            mild50_cache_root,
            limit=int(args.examples_per_mode),
            raw_hash_categories=raw_hash_categories,
        )
    if not mild50_examples:
        print(f"[WARN] No prepared 0.5 examples found under {_resolve_path(args.mild50_demo_examples)}")

    mild75_examples = discover_examples(args.mild75_demo_examples)
    if not mild75_examples:
        raw_hash_categories = raw_hash_categories or _load_raw_category_hashes(args.raw_categories_root)
        if not raw_hash_categories:
            print(f"[WARN] No raw category mapping loaded from {_resolve_path(args.raw_categories_root)}")
        cache_root = Path(tempfile.gettempdir()) / "dl_adaptive_optics_demo" / "mild75_raw_categories_v2"
        mild75_examples = materialize_lmdb_examples(
            args.mild75_lmdb_root,
            cache_root,
            limit=int(args.examples_per_mode),
            raw_hash_categories=raw_hash_categories,
        )
    if not mild75_examples:
        print(f"[WARN] No prepared 0.75 examples found under {_resolve_path(args.mild75_demo_examples)}")

    bundles = {
        "0.5 turbulence": ModeBundle(
            label="0.5 turbulence",
            single_engine=mild50_single,
            sequence_engine=mild50_sequence,
            examples={sample.name: sample for sample in mild50_examples},
            single_metrics=MODE_METRICS["0.5 turbulence"]["single"],
            sequence_metrics=MODE_METRICS["0.5 turbulence"]["sequence"],
        ),
        "0.75 turbulence": ModeBundle(
            label="0.75 turbulence",
            single_engine=mild75_single,
            sequence_engine=mild75_sequence,
            examples={sample.name: sample for sample in mild75_examples},
            single_metrics=MODE_METRICS["0.75 turbulence"]["single"],
            sequence_metrics=MODE_METRICS["0.75 turbulence"]["sequence"],
        ),
    }
    print(f"[INFO] Loaded {len(mild50_examples)} prepared 0.5 category examples.")
    print(f"[INFO] Loaded {len(mild75_examples)} prepared 0.75 category examples.")
    print("[INFO] Demo examples are loaded from demo/examples when available; LMDB materialization is a fallback.")
    print(f"[INFO] Demo device: {mild50_sequence.device}")

    app = build_app(DemoState(bundles=bundles))
    app.launch(
        server_name=args.host,
        server_port=int(args.port),
        share=bool(args.share),
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="amber"),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
