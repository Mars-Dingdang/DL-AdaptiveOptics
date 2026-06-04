"""Build a stronger seven-frame evaluation LMDB for sequence-trained models.

This script selects a contiguous slice of clean images, generates a stronger
seven-frame degradation sequence per image, writes the result to LMDB, and
exports a contact-sheet preview for visual inspection.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys

import cv2
import lmdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_common import build_turbulence_params, load_config
from utils.degradation import TurbulenceParams, add_atmospheric_turbulence_sequence, sample_turbulence_context
from utils.visualization import save_image_rgb


IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stronger seven-frame eval LMDB and degradation previews.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config path")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/clean_patches/nwpu_parquet/images"),
        help="Root directory containing clean images.",
    )
    parser.add_argument(
        "--output-lmdb-root",
        type=Path,
        default=Path("data/eval_seven_frame"),
        help="LMDB output directory.",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=Path("outputs/eval_seven_frame_degradation"),
        help="Directory for exported multi-frame preview PNGs.",
    )
    parser.add_argument("--start-index", type=int, default=2000, help="Inclusive start index in sorted clean files.")
    parser.add_argument("--count", type=int, default=30, help="Number of clean images to include.")
    parser.add_argument("--num-frames", type=int, default=7, help="Sequence length stored in LMDB.")
    parser.add_argument("--seed", type=int, default=2029, help="Base RNG seed for reproducibility.")
    parser.add_argument(
        "--backend",
        type=str,
        default="",
        choices=["", "turbsim_gpu_v1", "turbsim_cpu_v1", "simple_parametric"],
        help="Optional override for degradation backend.",
    )
    parser.add_argument(
        "--image-codec",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "webp"],
        help="Image codec for LMDB payloads.",
    )
    parser.add_argument("--image-quality", type=int, default=92, help="Quality for jpg/webp LMDB payloads.")
    parser.add_argument("--map-size-mb", type=int, default=512, help="LMDB map size in MB.")
    parser.add_argument("--force", action="store_true", help="Delete existing output directories before rebuilding.")
    return parser.parse_args()


def _scan_image_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input root does not exist: {root_dir}")

    files = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    if not files:
        raise RuntimeError(f"No image files found under {root_dir}")
    return files


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _resize_if_needed(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    short_side = min(h, w)
    if short_side >= target_size:
        return image

    scale = float(target_size) / float(short_side)
    new_w = max(target_size, int(round(w * scale)))
    new_h = max(target_size, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    src = _resize_if_needed(image, target_size=crop_size)
    h, w = src.shape[:2]
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)
    return src[top : top + crop_size, left : left + crop_size, :]


def _encode_rgb_image_bytes(image_rgb: np.ndarray, codec: str, quality: int) -> bytes:
    codec_norm = str(codec).lower().strip()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if codec_norm in {"jpg", "jpeg"}:
        ext = ".jpg"
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    elif codec_norm == "webp":
        ext = ".webp"
        encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(np.clip(quality, 1, 100))]
    else:
        ext = ".png"
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    ok, encoded = cv2.imencode(ext, image_bgr, encode_params)
    if not ok:
        raise RuntimeError(f"Failed to encode image with codec={codec_norm}.")
    return encoded.tobytes()


def _prepare_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (180, 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _build_stronger_params(cfg: dict[str, object], backend_override: str) -> TurbulenceParams:
    params = build_turbulence_params(cfg)
    # if backend_override:
    #     params.backend = backend_override

    # params.turbulence_strength = max(1.0, float(params.turbulence_strength))
    # params.turbsim_luma_only = False
    # params.turbsim_patch_grid_downsample = 1
    # params.turbsim_psf_resolution = max(32, int(params.turbsim_psf_resolution))
    # params.turbsim_reuse_psf_per_frame = False

    # cn2_lo, cn2_hi = params.cn2_range
    # params.cn2_range = (max(float(cn2_lo), 5.0e-16), max(float(cn2_hi), 1.0e-14))
    # wind_lo, wind_hi = params.wind_speed_range
    # params.wind_speed_range = (max(float(wind_lo), 1.0), max(float(wind_hi), 3.0))
    return params


def _save_preview(clean_rgb: np.ndarray, degraded_frames: np.ndarray, out_path: Path) -> None:
    tiles = [_put_label(clean_rgb, "Clean")]
    for frame_idx in range(degraded_frames.shape[0]):
        tiles.append(_put_label(degraded_frames[frame_idx], f"Frame {frame_idx}"))
    preview = np.concatenate(tiles, axis=1)
    save_image_rgb(preview, out_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    params = _build_stronger_params(cfg, backend_override=str(args.backend))

    image_size = int(cfg.get("data", {}).get("image_size", 256))
    files = _scan_image_files(args.input_root)
    end_index = int(args.start_index) + int(args.count)
    selected = files[int(args.start_index) : end_index]
    if len(selected) != int(args.count):
        raise RuntimeError(
            f"Requested {args.count} images from index {args.start_index}, but only found {len(selected)} files."
        )

    _prepare_dir(args.output_lmdb_root, force=bool(args.force))
    _prepare_dir(args.viz_dir, force=bool(args.force))

    env = lmdb.open(
        str(args.output_lmdb_root),
        map_size=int(args.map_size_mb) * 1024 * 1024,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    codec = str(args.image_codec).lower().strip()
    num_frames = int(args.num_frames)
    try:
        with env.begin(write=True) as txn:
            for sample_idx, src_path in enumerate(selected):
                clean_rgb = _read_rgb(src_path)
                clean_rgb = _center_crop(clean_rgb, crop_size=image_size)
                clean_f = clean_rgb.astype(np.float32) / 255.0

                sample_rng = np.random.default_rng(int(args.seed) + int(args.start_index) + sample_idx)
                context = sample_turbulence_context(params=params, rng=sample_rng)
                degraded_seq, frame_metas = add_atmospheric_turbulence_sequence(
                    image=clean_f,
                    num_frames=num_frames,
                    params=params,
                    rng=sample_rng,
                    context=context,
                    return_meta=True,
                )

                degraded_uint8_seq = np.clip(np.round(degraded_seq * 255.0), 0.0, 255.0).astype(np.uint8)

                sample_key = f"sample-{sample_idx:07d}"
                txn.put(
                    f"{sample_key}-clean".encode("utf-8"),
                    _encode_rgb_image_bytes(clean_rgb, codec=codec, quality=int(args.image_quality)),
                )
                for frame_idx, frame_uint8 in enumerate(degraded_uint8_seq):
                    txn.put(
                        f"{sample_key}-frame-{frame_idx:03d}".encode("utf-8"),
                        _encode_rgb_image_bytes(frame_uint8, codec=codec, quality=int(args.image_quality)),
                    )

                meta = {
                    "sample_id": f"{sample_idx:07d}",
                    "source_index": int(args.start_index) + sample_idx,
                    "source": str(src_path.as_posix()),
                    "num_frames": int(num_frames),
                    "mode": "stronger_seven_frame_sequence",
                    "backend": str(params.backend),
                    "context": context,
                    "frame_metas": frame_metas,
                    "strength_overrides": {
                        "turbulence_strength": float(params.turbulence_strength),
                        "turbsim_luma_only": bool(params.turbsim_luma_only),
                        "turbsim_patch_grid_downsample": int(params.turbsim_patch_grid_downsample),
                        "turbsim_psf_resolution": int(params.turbsim_psf_resolution),
                        "turbsim_reuse_psf_per_frame": bool(params.turbsim_reuse_psf_per_frame),
                        "cn2_range": [float(params.cn2_range[0]), float(params.cn2_range[1])],
                        "wind_speed_range": [float(params.wind_speed_range[0]), float(params.wind_speed_range[1])],
                    },
                }
                txn.put(f"{sample_key}-meta".encode("utf-8"), json.dumps(meta, ensure_ascii=True).encode("utf-8"))

                preview_name = f"unet_{sample_idx:05d}.png"
                _save_preview(clean_rgb=clean_rgb, degraded_frames=degraded_uint8_seq, out_path=args.viz_dir / preview_name)

            txn.put("__len__".encode("utf-8"), str(len(selected)).encode("utf-8"))
            txn.put(
                "__meta__".encode("utf-8"),
                json.dumps(
                    {
                        "num_samples": len(selected),
                        "num_frames": int(num_frames),
                        "image_size": image_size,
                        "source_root": str(args.input_root.as_posix()),
                        "source_start_index": int(args.start_index),
                        "backend": str(params.backend),
                        "build_mode": "stronger_seven_frame_eval",
                        "image_codec": codec,
                        "seed": int(args.seed),
                    },
                    ensure_ascii=True,
                ).encode("utf-8"),
            )
        env.sync()
    finally:
        env.close()

    print(f"[INFO] Built stronger seven-frame eval LMDB at: {args.output_lmdb_root}")
    print(f"[INFO] Exported sequence degradation previews to: {args.viz_dir}")
    print(f"[INFO] Samples: {len(selected)}, source indices: {args.start_index}-{end_index - 1}")


if __name__ == "__main__":
    main()