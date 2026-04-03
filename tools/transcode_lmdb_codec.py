"""Transcode image payloads in a sequence LMDB to a target codec.

This keeps key layout unchanged and only rewrites image-bearing values:
- sample-xxxxxxx-clean
- sample-xxxxxxx-frame-xxx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import lmdb
import numpy as np
from tqdm.auto import tqdm


def _decode_rgb_image_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if dec is None:
        raise RuntimeError("Failed to decode LMDB image bytes.")
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _encode_rgb_image_bytes(image_rgb: np.ndarray, codec: str, quality: int) -> bytes:
    codec_norm = str(codec).lower().strip()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if codec_norm in {"jpg", "jpeg"}:
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    elif codec_norm == "webp":
        ext = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(np.clip(quality, 1, 100))]
    else:
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    ok, encoded = cv2.imencode(ext, image_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image with codec={codec_norm}.")
    return encoded.tobytes()


def _is_image_key(key: str) -> bool:
    if key.endswith("-clean"):
        return True
    if "-frame-" in key:
        return True
    return False


def _resolve_map_size_bytes(input_lmdb_root: Path, map_size_gb: int) -> int:
    if map_size_gb > 0:
        return int(map_size_gb) * (1024**3)

    input_data_mdb = input_lmdb_root / "data.mdb"
    input_size = input_data_mdb.stat().st_size if input_data_mdb.exists() else 0
    return int(max(256 * (1024**2), input_size * 2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcode sequence LMDB image payloads to a target codec.")
    parser.add_argument("--input-lmdb", type=Path, required=True, help="Input LMDB directory.")
    parser.add_argument("--output-lmdb", type=Path, required=True, help="Output LMDB directory.")
    parser.add_argument("--codec", type=str, default="webp", choices=["png", "jpg", "jpeg", "webp"])
    parser.add_argument("--quality", type=int, default=92, help="Quality for jpg/webp (1-100).")
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=0,
        help="Output LMDB map size in GB. Use 0 for auto-estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_lmdb = args.input_lmdb
    output_lmdb = args.output_lmdb

    if not input_lmdb.exists():
        raise FileNotFoundError(f"Input LMDB does not exist: {input_lmdb}")
    if output_lmdb.exists() and any(output_lmdb.iterdir()):
        raise RuntimeError(f"Output LMDB already exists and is non-empty: {output_lmdb}")

    output_lmdb.mkdir(parents=True, exist_ok=True)

    env_in = lmdb.open(
        str(input_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    map_size_bytes = _resolve_map_size_bytes(input_lmdb_root=input_lmdb, map_size_gb=int(args.map_size_gb))
    env_out = lmdb.open(
        str(output_lmdb),
        map_size=map_size_bytes,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
    )

    try:
        with env_in.begin(write=False) as txn_in, env_out.begin(write=True) as txn_out:
            total_entries = int(env_in.stat().get("entries", 0))
            cursor = txn_in.cursor()
            for key_b, value_b in tqdm(cursor, total=total_entries, desc="Transcode LMDB", unit="entry"):
                key = key_b.decode("utf-8")
                out_value = value_b

                if _is_image_key(key):
                    image_rgb = _decode_rgb_image_bytes(value_b)
                    out_value = _encode_rgb_image_bytes(image_rgb, codec=str(args.codec), quality=int(args.quality))
                elif key == "__meta__":
                    try:
                        meta = json.loads(value_b.decode("utf-8"))
                        if isinstance(meta, dict):
                            meta["image_codec"] = str(args.codec).lower().strip()
                            meta["image_quality"] = int(args.quality)
                            meta["build_mode"] = "transcoded_lmdb"
                            out_value = json.dumps(meta, ensure_ascii=True).encode("utf-8")
                    except Exception:
                        out_value = value_b

                txn_out.put(key_b, out_value)
    finally:
        env_out.sync()
        env_in.close()
        env_out.close()

    print(f"[INFO] Transcoded LMDB saved to: {output_lmdb}")


if __name__ == "__main__":
    main()
