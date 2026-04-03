"""Convert sequence sample folders into an LMDB dataset.

The source layout is expected as:
  sample_0000000/
    clean.png
    frame_000.png ... frame_006.png
    meta.json

Stored keys (binary):
  __len__
  __meta__
  sample-0000000-clean
  sample-0000000-frame-000 ...
  sample-0000000-meta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import lmdb
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sequence folder dataset to LMDB.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing sample_* folders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output LMDB directory.")
    parser.add_argument("--num-frames", type=int, default=7, help="Frames per sample to export.")
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=0,
        help="LMDB map size in GB. Use 0 for auto-estimation based on source files.",
    )
    parser.add_argument(
        "--image-codec",
        type=str,
        choices=["raw", "png", "jpg", "jpeg", "webp"],
        default="raw",
        help="Storage codec in LMDB. raw keeps original bytes; jpg/webp usually reduce size.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="Quality for jpg/webp encoding (1-100). Ignored for raw/png.",
    )
    return parser.parse_args()


def _write_bytes(txn: lmdb.Transaction, key: str, value: bytes) -> None:
    txn.put(key.encode("utf-8"), value)


def _estimate_map_size_bytes(sample_dirs: list[Path], num_frames: int) -> int:
    total = 0
    for sample_dir in sample_dirs:
        total += (sample_dir / "clean.png").stat().st_size
        for frame_idx in range(num_frames):
            total += (sample_dir / f"frame_{frame_idx:03d}.png").stat().st_size
        meta_path = sample_dir / "meta.json"
        if meta_path.exists():
            total += meta_path.stat().st_size
    # Add 35% safety margin and a 256MB lower bound.
    return int(max(256.0 * (1024**2), float(total) * 1.35))


def _encode_image_bytes(raw_bytes: bytes, codec: str, quality: int) -> bytes:
    codec_norm = str(codec).lower().strip()
    if codec_norm == "raw":
        return raw_bytes

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode source image while converting LMDB codec.")

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


def convert_sequence_to_lmdb(
    input_root: Path,
    output_root: Path,
    num_frames: int,
    map_size_gb: int,
    image_codec: str,
    image_quality: int,
) -> int:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    sample_dirs = sorted([p for p in input_root.glob("sample_*") if p.is_dir()])
    if not sample_dirs:
        raise RuntimeError(f"No sample_* folders found under {input_root}")

    if map_size_gb <= 0:
        map_size_bytes = _estimate_map_size_bytes(sample_dirs=sample_dirs, num_frames=num_frames)
        print(f"[INFO] map_size_gb<=0, auto map_size selected: {map_size_bytes / (1024**3):.2f} GB")
    else:
        map_size_bytes = int(map_size_gb) * (1024**3)

    output_root.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(output_root),
        map_size=map_size_bytes,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
    )

    with env.begin(write=True) as txn:
        for sample_idx, sample_dir in enumerate(sample_dirs):
            sample_key = f"sample-{sample_idx:07d}"
            clean_path = sample_dir / "clean.png"
            if not clean_path.exists():
                raise FileNotFoundError(f"Missing clean image: {clean_path}")

            clean_raw = clean_path.read_bytes()
            _write_bytes(txn, f"{sample_key}-clean", _encode_image_bytes(clean_raw, codec=image_codec, quality=image_quality))

            for frame_idx in range(num_frames):
                frame_path = sample_dir / f"frame_{frame_idx:03d}.png"
                if not frame_path.exists():
                    raise FileNotFoundError(f"Missing frame image: {frame_path}")
                frame_raw = frame_path.read_bytes()
                _write_bytes(
                    txn,
                    f"{sample_key}-frame-{frame_idx:03d}",
                    _encode_image_bytes(frame_raw, codec=image_codec, quality=image_quality),
                )

            meta_path = sample_dir / "meta.json"
            meta_bytes = meta_path.read_bytes() if meta_path.exists() else b"{}"
            _write_bytes(txn, f"{sample_key}-meta", meta_bytes)

        _write_bytes(txn, "__len__", str(len(sample_dirs)).encode("utf-8"))
        meta = {
            "num_samples": len(sample_dirs),
            "num_frames": int(num_frames),
            "source_root": str(input_root.as_posix()),
            "image_codec": str(image_codec).lower().strip(),
            "image_quality": int(image_quality),
        }
        _write_bytes(txn, "__meta__", json.dumps(meta, ensure_ascii=True).encode("utf-8"))

    env.sync()
    env.close()
    return len(sample_dirs)


def main() -> None:
    args = parse_args()
    count = convert_sequence_to_lmdb(
        input_root=args.input_root,
        output_root=args.output_root,
        num_frames=int(args.num_frames),
        map_size_gb=int(args.map_size_gb),
        image_codec=str(args.image_codec),
        image_quality=int(args.image_quality),
    )
    print(f"lmdb_done {count}")


if __name__ == "__main__":
    main()
