"""Dataset preparation and offline turbulence-sequence generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
from queue import Full, Queue
import shutil
import tarfile
from threading import Thread
import urllib.request
import zipfile

import cv2
import lmdb
import numpy as np
import yaml
from tqdm.auto import tqdm

from utils.degradation import (
    TurbulenceParams,
    add_atmospheric_turbulence_sequence,
    sample_turbulence_context,
)


IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSource:
    """Description of a dataset source."""

    name: str
    url: str | None
    archive_name: str | None
    extracted_dir_name: str | None
    note: str


DATASET_SOURCES: dict[str, DatasetSource] = {
    "uc_merced": DatasetSource(
        name="UC Merced Land Use",
        url="http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
        archive_name="UCMerced_LandUse.zip",
        extracted_dir_name="UCMerced_LandUse",
        note="2100 images of 21 land-use classes, 256x256 RGB.",
    ),
    "nwpu_resisc45": DatasetSource(
        name="NWPU-RESISC45",
        url=None,
        archive_name=None,
        extracted_dir_name="NWPU-RESISC45",
        note=(
            "Use the HuggingFace mirror and place extracted files in data/raw/NWPU-RESISC45. "
            "Recommended source: https://huggingface.co/datasets/jonathan-roberts1/NWPU-RESISC45"
        ),
    ),
    "whu_rs19": DatasetSource(
        name="WHU-RS19",
        url=None,
        archive_name=None,
        extracted_dir_name="WHU-RS19",
        note=(
            "No stable direct link is bundled. Download manually from the official "
            "dataset page and place extracted files in data/raw/WHU-RS19."
        ),
    ),
    "nwpu_vhr10": DatasetSource(
        name="NWPU VHR-10",
        url=None,
        archive_name=None,
        extracted_dir_name="NWPU-VHR10",
        note=(
            "No stable direct link is bundled. Download manually from the official "
            "dataset page and place extracted files in data/raw/NWPU-VHR10."
        ),
    ),
}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
        return

    suffixes = {s.lower() for s in archive_path.suffixes}
    if {".tar", ".gz"}.issubset(suffixes) or ".tgz" in suffixes:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def prepare_dataset(dataset_key: str, data_root: Path, force_download: bool = False) -> Path:
    """Prepare one dataset and return its expected directory path."""
    if dataset_key not in DATASET_SOURCES:
        raise KeyError(f"Unknown dataset key: {dataset_key}. Available: {list(DATASET_SOURCES)}")

    source = DATASET_SOURCES[dataset_key]
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = raw_root / (source.extracted_dir_name or source.name.replace(" ", "_"))

    if source.url is None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {source.name}: manual download required.")
        print(f"[INFO] {source.note}")
        print(f"[INFO] Place extracted files under: {dataset_dir}")
        return dataset_dir

    assert source.archive_name is not None
    archive_path = raw_root / source.archive_name

    if force_download or not archive_path.exists():
        print(f"[INFO] Downloading {source.name} from {source.url}")
        _download(source.url, archive_path)

    if force_download or not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print(f"[INFO] Extracting archive to {raw_root}")
        _extract_archive(archive_path, raw_root)

    print(f"[INFO] Dataset ready at: {dataset_dir}")
    return dataset_dir


def _iter_image_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input root does not exist: {root_dir}")
    files = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    if not files:
        raise RuntimeError(f"No image files found under {root_dir}")
    return files


def _iter_parquet_files(root_dir: Path) -> list[Path]:
    files = [p for p in root_dir.rglob("*.parquet") if p.is_file()]
    files.sort()
    return files


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def _decode_hf_image_cell(image_cell: object, parquet_root: Path) -> np.ndarray:
    """Decode HuggingFace image cell into RGB ndarray."""
    if isinstance(image_cell, dict):
        if image_cell.get("bytes") is not None:
            buf = image_cell["bytes"]
            if isinstance(buf, memoryview):
                buf = buf.tobytes()
            if not isinstance(buf, (bytes, bytearray)):
                raise RuntimeError("Unsupported bytes payload in parquet image cell.")
            arr = np.frombuffer(buf, dtype=np.uint8)
            dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if dec is None:
                raise RuntimeError("Failed to decode parquet image bytes.")
            return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        rel_path = image_cell.get("path")
        if rel_path:
            path = (parquet_root / str(rel_path)).resolve()
            if path.exists():
                return _read_rgb(path)

    raise RuntimeError("Unsupported parquet image format. Expected dict with bytes/path.")


def _iter_parquet_images(parquet_path: Path) -> list[tuple[str, np.ndarray, int | None]]:
    """Read images from HuggingFace parquet file as (id, image_rgb, label)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("pyarrow is required to read parquet datasets. Install with: pip install pyarrow") from exc

    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    out: list[tuple[str, np.ndarray, int | None]] = []

    for idx, row in enumerate(records):
        image_cell = row.get("image")
        label_val = row.get("label")
        label_int: int | None = int(label_val) if label_val is not None else None
        image_rgb = _decode_hf_image_cell(image_cell=image_cell, parquet_root=parquet_path.parent)
        out.append((f"{parquet_path.name}:{idx}", image_rgb, label_int))

    return out


def _cloud_score(image_rgb: np.ndarray) -> float:
    """Heuristic cloud score in [0,1], higher means more cloud-like."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    bright_low_sat = (val > 0.8) & (sat < 0.18)
    flat_texture = cv2.Laplacian(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() < 18.0
    score = float(bright_low_sat.mean())
    if flat_texture:
        score = min(1.0, score + 0.08)
    return score


def _save_manifest_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        return
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_nwpu_clean_patches(
    input_root: Path,
    output_root: Path,
    patch_size: int,
    stride: int,
    max_patches: int,
    cloud_threshold: float,
    seed: int,
) -> int:
    """Extract cloud-light clean 256x256 patches from NWPU image pool."""
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if max_patches <= 0:
        raise ValueError("max_patches must be > 0")

    image_files: list[Path] = []
    parquet_files = _iter_parquet_files(input_root)
    if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in input_root.rglob("*")):
        image_files = _iter_image_files(input_root)

    rng = np.random.default_rng(seed)
    if image_files:
        rng.shuffle(image_files)

    patches_dir = output_root / "images"
    patches_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    patch_idx = 0

    def emit_patches(image_rgb: np.ndarray, src_id: str, label: int | None) -> bool:
        nonlocal patch_idx
        if label == 9:
            return False
        if _cloud_score(image_rgb) > cloud_threshold:
            return False

        h, w = image_rgb.shape[:2]
        if h < patch_size or w < patch_size:
            return False

        for top in range(0, h - patch_size + 1, stride):
            for left in range(0, w - patch_size + 1, stride):
                patch = image_rgb[top : top + patch_size, left : left + patch_size, :]
                out_path = patches_dir / f"patch_{patch_idx:07d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

                rows.append(
                    {
                        "patch_id": f"{patch_idx:07d}",
                        "src_image": src_id,
                        "top": str(top),
                        "left": str(left),
                        "height": str(patch_size),
                        "width": str(patch_size),
                        "label": "" if label is None else str(label),
                        "cloud_score": f"{_cloud_score(patch):.6f}",
                    }
                )
                patch_idx += 1
                if patch_idx >= max_patches:
                    return True
        return False

    if image_files:
        for image_path in image_files:
            image_rgb = _read_rgb(image_path)
            reached = emit_patches(image_rgb=image_rgb, src_id=str(image_path.as_posix()), label=None)
            if reached:
                _save_manifest_rows(output_root / "manifest_clean.csv", rows)
                print(f"[INFO] Extracted clean patches: {patch_idx}")
                return patch_idx
    elif parquet_files:
        for parquet_path in parquet_files:
            print(f"[INFO] Reading parquet source: {parquet_path}")
            samples = _iter_parquet_images(parquet_path)
            rng.shuffle(samples)
            for src_id, image_rgb, label in samples:
                reached = emit_patches(image_rgb=image_rgb, src_id=src_id, label=label)
                if reached:
                    _save_manifest_rows(output_root / "manifest_clean.csv", rows)
                    print(f"[INFO] Extracted clean patches: {patch_idx}")
                    return patch_idx
    else:
        raise RuntimeError(f"No image files or parquet files found under {input_root}")

    _save_manifest_rows(output_root / "manifest_clean.csv", rows)
    print(f"[INFO] Extracted clean patches: {patch_idx}")
    return patch_idx


def _ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _encode_rgb_image_bytes(image_rgb: np.ndarray, codec: str, quality: int) -> bytes:
    """Encode RGB uint8 image to bytes for LMDB storage."""
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
        raise RuntimeError(f"Failed to encode image for LMDB export with codec={codec_norm}.")
    return encoded.tobytes()


def _estimate_direct_lmdb_map_size_bytes(
    clean_files: list[Path],
    target_samples: int,
    num_frames: int,
    codec: str,
) -> int:
    """Estimate a practical LMDB map_size for direct build to avoid massive over-allocation."""
    if target_samples <= 0:
        n_samples = len(clean_files)
    else:
        n_samples = int(target_samples)

    sample_files = clean_files[: min(256, len(clean_files))]
    avg_clean_bytes = 0.0
    if sample_files:
        avg_clean_bytes = float(np.mean([p.stat().st_size for p in sample_files]))

    codec_norm = str(codec).lower().strip()
    # Approximate frame size ratio vs source clean PNG/JPG files.
    if codec_norm in {"jpg", "jpeg"}:
        frame_ratio = 0.55
    elif codec_norm == "webp":
        frame_ratio = 0.45
    else:
        frame_ratio = 1.0

    per_sample_bytes = avg_clean_bytes * (1.0 + float(num_frames) * frame_ratio) + 8192.0
    estimated = int(max(256.0 * (1024**2), per_sample_bytes * float(n_samples) * 1.35))
    return estimated


def _load_yaml_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a dictionary")
    return cfg


def _pick_value(cli_val: object, cfg_val: object, default_val: object) -> object:
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default_val


def _sample_seed(base_seed: int, sample_idx: int) -> int:
    """Derive a stable per-sample seed for resumable chunk generation."""
    # Mix with a large odd multiplier to reduce neighboring-index correlation.
    return int((int(base_seed) + int(sample_idx) * 1000003) % (2**63 - 1))


def _infer_existing_lmdb_num_samples(txn: lmdb.Transaction) -> int:
    """Infer existing sample count from __len__ and sample keys."""
    existing_from_len = 0
    raw_len = txn.get("__len__".encode("utf-8"))
    if raw_len is not None:
        try:
            existing_from_len = int(raw_len.decode("utf-8"))
        except Exception:
            existing_from_len = 0

    max_idx = -1
    cursor = txn.cursor()
    for key_b, _ in cursor:
        try:
            key = key_b.decode("utf-8")
        except Exception:
            continue
        if not key.startswith("sample-") or not key.endswith("-clean"):
            continue
        # sample-0000123-clean -> 0000123
        parts = key.split("-")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[1])
        except Exception:
            continue
        if idx > max_idx:
            max_idx = idx

    existing_from_keys = max_idx + 1 if max_idx >= 0 else 0
    return max(existing_from_len, existing_from_keys)


def _resolve_build_sequence_from_config(args: argparse.Namespace) -> dict[str, object]:
    cfg = _load_yaml_config(args.config)
    bcfg = cfg.get("build_sequence", {}) if isinstance(cfg, dict) else {}
    dcfg = cfg.get("degradation", {}) if isinstance(cfg, dict) else {}
    if not isinstance(bcfg, dict):
        bcfg = {}
    if not isinstance(dcfg, dict):
        dcfg = {}

    data_root_default = Path(__file__).resolve().parents[1] / "data"
    data_root = Path(_pick_value(args.data_root, bcfg.get("data_root"), data_root_default))

    clean_root = Path(
        _pick_value(
            args.input_root,
            bcfg.get("input_root"),
            data_root / "clean_patches" / "nwpu" / "images",
        )
    )
    output_root = Path(
        _pick_value(
            args.output_root,
            bcfg.get("output_root"),
            data_root / "processed" / "train",
        )
    )

    lmdb_cfg = bcfg.get("lmdb", {}) if isinstance(bcfg, dict) else {}
    gif_cfg = bcfg.get("gif", {}) if isinstance(bcfg, dict) else {}
    if not isinstance(lmdb_cfg, dict):
        lmdb_cfg = {}
    if not isinstance(gif_cfg, dict):
        gif_cfg = {}
    lmdb_output_root = Path(
        _pick_value(
            args.lmdb_output_root,
            lmdb_cfg.get("output_root"),
            data_root / "processed" / "train_lmdb",
        )
    )

    resolved: dict[str, object] = {
        "clean_root": clean_root,
        "output_root": output_root,
        "sample_start_index": int(_pick_value(args.sample_start_index, bcfg.get("sample_start_index"), 0)),
        "target_samples": int(_pick_value(args.target_samples, bcfg.get("target_samples"), 50000)),
        "num_frames": int(_pick_value(args.num_frames, bcfg.get("num_frames"), 7)),
        "seed": int(_pick_value(args.seed, bcfg.get("seed"), 42)),
        "lmdb_only": bool(_pick_value(args.lmdb_only, lmdb_cfg.get("only"), False)),
        "lmdb_output_root": lmdb_output_root,
        "lmdb_map_size_gb": int(_pick_value(args.lmdb_map_size_gb, lmdb_cfg.get("map_size_gb"), 0)),
        "lmdb_image_codec": str(_pick_value(args.lmdb_image_codec, lmdb_cfg.get("image_codec"), "png")),
        "lmdb_image_quality": int(_pick_value(args.lmdb_image_quality, lmdb_cfg.get("image_quality"), 95)),
        "lmdb_prefetch_queue_size": int(_pick_value(args.lmdb_prefetch_queue_size, lmdb_cfg.get("prefetch_queue_size"), 8)),
        "lmdb_allow_overwrite": bool(_pick_value(args.lmdb_allow_overwrite, lmdb_cfg.get("allow_overwrite"), False)),
        "gif_enabled": bool(_pick_value(args.gif_enabled, gif_cfg.get("enabled", bool(gif_cfg)), False)),
        "gif_duration_ms": int(_pick_value(args.gif_duration_ms, gif_cfg.get("duration_ms"), 220)),
        "gif_loop": int(_pick_value(args.gif_loop, gif_cfg.get("loop"), 0)),
        "gif_optimize": bool(_pick_value(args.gif_optimize, gif_cfg.get("optimize"), True)),
        "gif_name": str(_pick_value(args.gif_name, gif_cfg.get("name"), "turbulence.gif")),
        "backend": str(_pick_value(args.backend, dcfg.get("backend"), "turbsim_gpu_v1")),
        "zernike_order": int(_pick_value(args.zernike_order, dcfg.get("zernike_order"), 6)),
        "phase_strength": float(_pick_value(args.phase_strength, dcfg.get("phase_strength"), 0.12)),
        "psf_kernel_size": int(_pick_value(args.psf_kernel_size, dcfg.get("psf_kernel_size"), 5)),
        "cn2_min": float(_pick_value(args.cn2_min, (dcfg.get("cn2_range") or [1e-16, 5e-15])[0], 1e-16)),
        "cn2_max": float(_pick_value(args.cn2_max, (dcfg.get("cn2_range") or [1e-16, 5e-15])[1], 5e-15)),
        "focal_length_min": float(
            _pick_value(args.focal_length_min, (dcfg.get("focal_length_range") or [6000.0, 18000.0])[0], 6000.0)
        ),
        "focal_length_max": float(
            _pick_value(args.focal_length_max, (dcfg.get("focal_length_range") or [6000.0, 18000.0])[1], 18000.0)
        ),
        "wind_speed_min": float(_pick_value(args.wind_speed_min, (dcfg.get("wind_speed_range") or [0.5, 2.0])[0], 0.5)),
        "wind_speed_max": float(_pick_value(args.wind_speed_max, (dcfg.get("wind_speed_range") or [0.5, 2.0])[1], 2.0)),
        "sequence_time_step": float(_pick_value(args.sequence_time_step, dcfg.get("sequence_time_step"), 0.03)),
        "aperture_diameter": float(_pick_value(args.aperture_diameter, dcfg.get("aperture_diameter"), 0.2)),
        "wavelength": float(_pick_value(args.wavelength, dcfg.get("wavelength"), 0.525e-6)),
        "object_size": float(_pick_value(args.object_size, dcfg.get("object_size"), 2.06)),
        "turbulence_strength": float(_pick_value(args.turbulence_strength, dcfg.get("turbulence_strength"), 0.5)),
        "turbsim_luma_only": bool(_pick_value(args.turbsim_luma_only, dcfg.get("turbsim_luma_only"), False)),
        "turbsim_patch_grid_downsample": int(
            _pick_value(args.turbsim_patch_grid_downsample, dcfg.get("turbsim_patch_grid_downsample"), 1)
        ),
        "turbsim_psf_resolution": int(_pick_value(args.turbsim_psf_resolution, dcfg.get("turbsim_psf_resolution"), 32)),
        "turbsim_reuse_psf_per_frame": bool(
            _pick_value(args.turbsim_reuse_psf_per_frame, dcfg.get("turbsim_reuse_psf_per_frame"), False)
        ),
    }
    return resolved


def _save_sample_gif(
    sample_dir: Path,
    num_frames: int,
    duration_ms: int,
    loop: int,
    optimize: bool,
    name: str,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("Pillow is required for GIF export. Install with: pip install pillow") from exc

    frames = []
    for frame_idx in range(num_frames):
        frame_path = sample_dir / f"frame_{frame_idx:03d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame for GIF export: {frame_path}")
        frames.append(Image.open(frame_path).convert("RGB"))

    out_path = sample_dir / name
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=int(loop),
        optimize=bool(optimize),
    )


def build_turbulence_sequence_dataset(
    clean_root: Path,
    output_root: Path,
    sample_start_index: int,
    target_samples: int,
    num_frames: int,
    params: TurbulenceParams,
    seed: int,
    gif_enabled: bool,
    gif_duration_ms: int,
    gif_loop: int,
    gif_optimize: bool,
    gif_name: str,
) -> int:
    """Build offline sequence dataset: clean.png + frame_000.. for each sample."""
    if sample_start_index < 0:
        raise ValueError("sample_start_index must be >= 0")
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    patch_files = _iter_image_files(clean_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "clean_src", "frames", "backend", "cn2", "focal_length", "wind_speed"]
    manifest_path = output_root / "manifest_sequence.csv"
    _ensure_csv_header(manifest_path, fieldnames)

    created = 0

    start = int(sample_start_index)
    stop = int(sample_start_index + target_samples)
    for sample_idx in tqdm(range(start, stop), desc="Build sequence samples", unit="sample"):
        sample_dir = output_root / f"sample_{sample_idx:07d}"
        clean_out = sample_dir / "clean.png"
        frame_0 = sample_dir / "frame_000.png"

        if clean_out.exists() and frame_0.exists():
            continue

        sample_rng = np.random.default_rng(_sample_seed(seed, sample_idx))
        src_path = patch_files[int(sample_rng.integers(0, len(patch_files)))]
        clean_rgb = _read_rgb(src_path)
        clean_f = clean_rgb.astype(np.float32) / 255.0

        context = sample_turbulence_context(params=params, rng=sample_rng)
        seq, frame_metas = add_atmospheric_turbulence_sequence(
            image=clean_f,
            num_frames=num_frames,
            params=params,
            rng=sample_rng,
            context=context,
            return_meta=True,
        )

        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(clean_out), cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR))
        for frame_idx, frame in enumerate(seq):
            frame_uint8 = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
            frame_path = sample_dir / f"frame_{frame_idx:03d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

        if bool(gif_enabled):
            _save_sample_gif(
                sample_dir=sample_dir,
                num_frames=num_frames,
                duration_ms=gif_duration_ms,
                loop=gif_loop,
                optimize=gif_optimize,
                name=gif_name,
            )

        meta = {
            "sample_id": f"{sample_idx:07d}",
            "source": str(src_path.as_posix()),
            "num_frames": int(num_frames),
            "backend": str(params.backend),
            "device": str(frame_metas[0].get("device", "")) if frame_metas else "",
            "context": context,
            "frames": frame_metas,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

        _append_csv_row(
            manifest_path,
            fieldnames,
            {
                "sample_id": f"{sample_idx:07d}",
                "clean_src": str(src_path.as_posix()),
                "frames": str(num_frames),
                "backend": str(params.backend),
                "cn2": f"{float(context['cn2']):.6e}",
                "focal_length": f"{float(context['focal_length']):.6f}",
                "wind_speed": f"{float(context['wind_speed']):.6f}",
            },
        )
        created += 1

        if (sample_idx + 1) % 200 == 0:
            print(f"[INFO] Generated {sample_idx + 1}/{target_samples} sequence samples...")

    print(f"[INFO] Newly created sequence samples: {created}")
    return created


def build_turbulence_sequence_lmdb(
    clean_root: Path,
    lmdb_root: Path,
    sample_start_index: int,
    target_samples: int,
    num_frames: int,
    params: TurbulenceParams,
    seed: int,
    map_size_gb: int,
    image_codec: str,
    image_quality: int,
    prefetch_queue_size: int,
    allow_overwrite: bool,
) -> int:
    """Build sequence LMDB directly from clean patches without writing frame PNG folders."""
    if sample_start_index < 0:
        raise ValueError("sample_start_index must be >= 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    patch_files = _iter_image_files(clean_root)

    # target_samples <= 0 means using full clean pool once (no replacement).
    if target_samples <= 0:
        if sample_start_index != 0:
            raise ValueError("sample_start_index must be 0 when target_samples<=0 (full-pool mode).")
        total_samples = len(patch_files)
        sample_indices = list(range(total_samples))
        sample_sources = patch_files.copy()
        np.random.default_rng(seed).shuffle(sample_sources)
    else:
        total_samples = int(target_samples)
        sample_indices = list(range(int(sample_start_index), int(sample_start_index + target_samples)))
        sample_sources = []

    if total_samples <= 0:
        raise RuntimeError("No source clean images resolved for LMDB build.")

    codec_norm = str(image_codec).lower().strip()
    if codec_norm not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError("image_codec must be one of: png, jpg, jpeg, webp")

    if map_size_gb <= 0:
        map_size_bytes = _estimate_direct_lmdb_map_size_bytes(
            clean_files=patch_files,
            target_samples=target_samples,
            num_frames=num_frames,
            codec=codec_norm,
        )
        print(f"[INFO] map_size_gb<=0, auto map_size selected: {map_size_bytes / (1024**3):.2f} GB")
    else:
        map_size_bytes = int(map_size_gb) * (1024**3)

    lmdb_root.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(lmdb_root),
        map_size=map_size_bytes,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
    )

    existing_samples = 0
    with env.begin(write=False) as txn_read:
        existing_samples = _infer_existing_lmdb_num_samples(txn_read)

    if target_samples > 0 and existing_samples > 0 and sample_start_index < existing_samples and not allow_overwrite:
        raise ValueError(
            f"LMDB already has {existing_samples} samples; sample_start_index={sample_start_index} would overwrite. "
            "Set --sample-start-index >= existing count, or pass --lmdb-allow-overwrite to overwrite intentionally."
        )

    # Pipeline GPU simulation (producer) and CPU encode/write (consumer).
    write_queue: Queue[dict[str, object] | None] = Queue(maxsize=max(1, int(prefetch_queue_size)))
    writer_errors: list[Exception] = []
    writer_device = {"value": ""}

    def _writer_loop() -> None:
        try:
            with env.begin(write=True) as txn:
                processed = 0
                while True:
                    item = write_queue.get()
                    try:
                        if item is None:
                            break

                        sample_idx = int(item["sample_idx"])
                        src_path = Path(str(item["src_path"]))
                        clean_rgb = np.asarray(item["clean_rgb"], dtype=np.uint8)
                        seq = np.asarray(item["seq"], dtype=np.float32)
                        context = dict(item["context"])
                        frame_metas = list(item["frame_metas"])

                        sample_key = f"sample-{sample_idx:07d}"
                        txn.put(
                            f"{sample_key}-clean".encode("utf-8"),
                            _encode_rgb_image_bytes(clean_rgb, codec=codec_norm, quality=image_quality),
                        )
                        for frame_idx, frame in enumerate(seq):
                            frame_uint8 = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                            txn.put(
                                f"{sample_key}-frame-{frame_idx:03d}".encode("utf-8"),
                                _encode_rgb_image_bytes(frame_uint8, codec=codec_norm, quality=image_quality),
                            )

                        meta = {
                            "sample_id": f"{sample_idx:07d}",
                            "source": str(src_path.as_posix()),
                            "num_frames": int(num_frames),
                            "backend": str(params.backend),
                            "context": context,
                            "frames": frame_metas,
                        }
                        txn.put(f"{sample_key}-meta".encode("utf-8"), json.dumps(meta, ensure_ascii=True).encode("utf-8"))

                        if frame_metas:
                            writer_device["value"] = str(frame_metas[0].get("device", writer_device["value"]))

                        processed += 1
                        if processed % 200 == 0:
                            print(f"[INFO] LMDB generated {processed}/{total_samples} samples...")
                    finally:
                        write_queue.task_done()

                txn.put("__len__".encode("utf-8"), str(total_samples).encode("utf-8"))
                if target_samples > 0:
                    final_num_samples = max(int(existing_samples), int(sample_start_index + total_samples))
                else:
                    final_num_samples = max(int(existing_samples), int(total_samples))
                txn.put("__len__".encode("utf-8"), str(final_num_samples).encode("utf-8"))
                lmdb_meta = {
                    "num_samples": int(final_num_samples),
                    "num_frames": int(num_frames),
                    "source_root": str(clean_root.as_posix()),
                    "backend": str(params.backend),
                    "device": writer_device["value"],
                    "build_mode": "direct_lmdb",
                    "image_codec": codec_norm,
                    "image_quality": int(image_quality),
                }
                txn.put("__meta__".encode("utf-8"), json.dumps(lmdb_meta, ensure_ascii=True).encode("utf-8"))
        except Exception as exc:  # pragma: no cover - runtime path
            writer_errors.append(exc)

    try:
        writer_thread = Thread(target=_writer_loop, name="lmdb-writer", daemon=True)
        writer_thread.start()

        if sample_sources:
            source_iter = zip(sample_indices, sample_sources)
        else:
            source_iter = ((idx, None) for idx in sample_indices)

        for sample_idx, src_path in tqdm(source_iter, total=total_samples, desc="Build LMDB samples", unit="sample"):
            sample_rng = np.random.default_rng(_sample_seed(seed, sample_idx))
            if src_path is None:
                src_path = patch_files[int(sample_rng.integers(0, len(patch_files)))]
            clean_rgb = _read_rgb(src_path)
            clean_f = clean_rgb.astype(np.float32) / 255.0
            context = sample_turbulence_context(params=params, rng=sample_rng)
            seq, frame_metas = add_atmospheric_turbulence_sequence(
                image=clean_f,
                num_frames=num_frames,
                params=params,
                rng=sample_rng,
                context=context,
                return_meta=True,
            )

            payload = {
                "sample_idx": sample_idx,
                "src_path": str(src_path),
                "clean_rgb": clean_rgb,
                "seq": seq,
                "context": context,
                "frame_metas": frame_metas,
            }
            while True:
                if writer_errors:
                    raise writer_errors[0]
                try:
                    write_queue.put(payload, timeout=0.5)
                    break
                except Full:
                    continue

            if writer_errors:
                raise writer_errors[0]

        while True:
            if writer_errors:
                raise writer_errors[0]
            try:
                write_queue.put(None, timeout=0.5)
                break
            except Full:
                continue
        write_queue.join()
        writer_thread.join()

        if writer_errors:
            raise writer_errors[0]
    finally:
        env.sync()
        env.close()

    print(f"[INFO] Direct LMDB build completed: {total_samples} samples at {lmdb_root}")
    return total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset preparation and turbulence sequence generation.")
    parser.add_argument(
        "--action",
        type=str,
        default="prepare",
        choices=["prepare", "extract_clean", "build_sequence"],
        help="Operation to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. For build_sequence, reads build/degradation settings from config.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Project data root directory.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="uc_merced",
        choices=sorted(DATASET_SOURCES.keys()),
        help="Dataset key to prepare when action=prepare.",
    )
    parser.add_argument("--force-download", action="store_true", help="Force redownload and re-extract archive.")

    parser.add_argument("--input-root", type=Path, default=None, help="Input image root for extract/build actions.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root for extract/build actions.")
    parser.add_argument(
        "--sample-start-index",
        type=int,
        default=None,
        help="Start sample index for chunked build_sequence generation (e.g., 0 then 2000 then 4000).",
    )
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for clean extraction.")
    parser.add_argument("--stride", type=int, default=256, help="Stride for clean extraction.")
    parser.add_argument("--max-patches", type=int, default=50000, help="Maximum clean patches to extract.")
    parser.add_argument(
        "--cloud-threshold",
        type=float,
        default=0.22,
        help="Cloud score threshold; lower is stricter clear-image filtering.",
    )

    parser.add_argument("--target-samples", type=int, default=None, help="Target sequence sample count.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of turbulence frames per sample.")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["turbsim_v1", "turbsim_gpu_v1", "simple_parametric"],
    )
    parser.add_argument(
        "--lmdb-only",
        action="store_true",
        default=None,
        help="When set, build sequence data directly to LMDB and skip intermediate sample PNG folders.",
    )
    parser.add_argument("--lmdb-output-root", type=Path, default=None, help="Output root for direct LMDB build.")
    parser.add_argument(
        "--lmdb-map-size-gb",
        type=int,
        default=None,
        help="LMDB map size in GB. Use 0 for auto-estimation to avoid oversized pre-allocation.",
    )
    parser.add_argument(
        "--lmdb-image-codec",
        type=str,
        choices=["png", "jpg", "jpeg", "webp"],
        default=None,
        help="Image codec stored in LMDB values. png is lossless; jpg/webp are smaller.",
    )
    parser.add_argument(
        "--lmdb-image-quality",
        type=int,
        default=None,
        help="Quality for jpg/webp image codec (1-100). Ignored for png.",
    )
    parser.add_argument(
        "--lmdb-prefetch-queue-size",
        type=int,
        default=None,
        help="Producer/consumer queue size for direct LMDB build (higher overlaps GPU simulation with CPU encode/write).",
    )
    parser.add_argument(
        "--lmdb-allow-overwrite",
        action="store_true",
        default=None,
        help="Allow overwriting existing sample keys in LMDB when sample_start_index overlaps.",
    )
    parser.add_argument(
        "--gif-enabled",
        action="store_true",
        default=None,
        help="When set (folder mode), export per-sample GIF alongside frame_XXX.png files.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=None,
        help="GIF frame duration in milliseconds (folder mode only).",
    )
    parser.add_argument(
        "--gif-loop",
        type=int,
        default=None,
        help="GIF loop count, 0 means infinite (folder mode only).",
    )
    parser.add_argument(
        "--gif-optimize",
        action="store_true",
        default=None,
        help="Enable GIF optimization via Pillow (folder mode only).",
    )
    parser.add_argument(
        "--gif-name",
        type=str,
        default=None,
        help="GIF file name inside each sample directory (folder mode only).",
    )

    parser.add_argument("--zernike-order", type=int, default=None)
    parser.add_argument("--phase-strength", type=float, default=None)
    parser.add_argument("--psf-kernel-size", type=int, default=None)
    parser.add_argument("--cn2-min", type=float, default=None)
    parser.add_argument("--cn2-max", type=float, default=None)
    parser.add_argument("--focal-length-min", type=float, default=None)
    parser.add_argument("--focal-length-max", type=float, default=None)
    parser.add_argument("--wind-speed-min", type=float, default=None)
    parser.add_argument("--wind-speed-max", type=float, default=None)
    parser.add_argument("--sequence-time-step", type=float, default=None)
    parser.add_argument("--aperture-diameter", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--object-size", type=float, default=None)
    parser.add_argument("--turbulence-strength", type=float, default=None)
    parser.add_argument(
        "--turbsim-luma-only",
        action="store_true",
        default=None,
        help="Fast mode: run TurbSim on luminance channel only and reconstruct RGB with original chroma.",
    )
    parser.add_argument(
        "--turbsim-patch-grid-downsample",
        type=int,
        default=None,
        help="Speed-quality knob for TurbSim blur patch grid. >1 reduces number of blur patches.",
    )
    parser.add_argument(
        "--turbsim-psf-resolution",
        type=int,
        default=None,
        help="Speed-quality knob for TurbSim PSF resolution (default 32). Lower is faster.",
    )
    parser.add_argument(
        "--turbsim-reuse-psf-per-frame",
        action="store_true",
        default=None,
        help="When set, reuses one PSF for all blur patches within a frame to accelerate generation.",
    )
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "prepare":
        prepare_dataset(dataset_key=args.dataset, data_root=args.data_root, force_download=bool(args.force_download))
        return

    if args.action == "extract_clean":
        input_root = args.input_root or (args.data_root / "raw" / "NWPU-RESISC45")
        output_root = args.output_root or (args.data_root / "clean_patches" / "nwpu")
        prepare_nwpu_clean_patches(
            input_root=input_root,
            output_root=output_root,
            patch_size=int(args.patch_size),
            stride=int(args.stride),
            max_patches=int(args.max_patches),
            cloud_threshold=float(args.cloud_threshold),
            seed=int(args.seed if args.seed is not None else 42),
        )
        return

    if args.action == "build_sequence":
        resolved = _resolve_build_sequence_from_config(args)
        params = TurbulenceParams(
            backend=str(resolved["backend"]),
            zernike_order=int(resolved["zernike_order"]),
            phase_strength=float(resolved["phase_strength"]),
            psf_kernel_size=int(resolved["psf_kernel_size"]),
            cn2_range=(float(resolved["cn2_min"]), float(resolved["cn2_max"])),
            focal_length_range=(float(resolved["focal_length_min"]), float(resolved["focal_length_max"])),
            wind_speed_range=(float(resolved["wind_speed_min"]), float(resolved["wind_speed_max"])),
            sequence_time_step=float(resolved["sequence_time_step"]),
            aperture_diameter=float(resolved["aperture_diameter"]),
            wavelength=float(resolved["wavelength"]),
            object_size=float(resolved["object_size"]),
            turbulence_strength=float(resolved["turbulence_strength"]),
            turbsim_luma_only=bool(resolved["turbsim_luma_only"]),
            turbsim_patch_grid_downsample=int(resolved["turbsim_patch_grid_downsample"]),
            turbsim_psf_resolution=int(resolved["turbsim_psf_resolution"]),
            turbsim_reuse_psf_per_frame=bool(resolved["turbsim_reuse_psf_per_frame"]),
        )
        if bool(resolved["lmdb_only"]):
            build_turbulence_sequence_lmdb(
                clean_root=Path(resolved["clean_root"]),
                lmdb_root=Path(resolved["lmdb_output_root"]),
                sample_start_index=int(resolved["sample_start_index"]),
                target_samples=int(resolved["target_samples"]),
                num_frames=int(resolved["num_frames"]),
                params=params,
                seed=int(resolved["seed"]),
                map_size_gb=int(resolved["lmdb_map_size_gb"]),
                image_codec=str(resolved["lmdb_image_codec"]),
                image_quality=int(resolved["lmdb_image_quality"]),
                prefetch_queue_size=int(resolved["lmdb_prefetch_queue_size"]),
                allow_overwrite=bool(resolved["lmdb_allow_overwrite"]),
            )
        else:
            build_turbulence_sequence_dataset(
                clean_root=Path(resolved["clean_root"]),
                output_root=Path(resolved["output_root"]),
                sample_start_index=int(resolved["sample_start_index"]),
                target_samples=int(resolved["target_samples"]),
                num_frames=int(resolved["num_frames"]),
                params=params,
                seed=int(resolved["seed"]),
                gif_enabled=bool(resolved["gif_enabled"]),
                gif_duration_ms=int(resolved["gif_duration_ms"]),
                gif_loop=int(resolved["gif_loop"]),
                gif_optimize=bool(resolved["gif_optimize"]),
                gif_name=str(resolved["gif_name"]),
            )
        return

    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
