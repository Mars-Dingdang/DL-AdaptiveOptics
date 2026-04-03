"""Export frame sequences to per-sample GIF files.

Usage:
    python tools/export_sequence_gifs.py --root data/turbulence_seq_nwpu_ultramild_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-sample turbulence GIFs from frame folders.")
    parser.add_argument("--root", type=Path, required=True, help="Sequence root containing sample_* folders.")
    parser.add_argument("--num-frames", type=int, default=7, help="Number of frame_XXX images to include.")
    parser.add_argument("--duration-ms", type=int, default=220, help="GIF frame duration in milliseconds.")
    parser.add_argument("--loop", type=int, default=0, help="GIF loop count (0 means infinite).")
    parser.add_argument("--optimize", action="store_true", help="Enable PIL GIF optimization.")
    parser.add_argument("--name", type=str, default="turbulence.gif", help="Output GIF file name in each sample dir.")
    return parser.parse_args()


def export_gifs(root: Path, num_frames: int, duration_ms: int, loop: int, optimize: bool, name: str) -> int:
    sample_dirs = sorted([p for p in root.glob("sample_*") if p.is_dir()])
    count = 0
    for sample_dir in sample_dirs:
        frames = []
        for frame_idx in range(num_frames):
            frame_path = sample_dir / f"frame_{frame_idx:03d}.png"
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing frame: {frame_path}")
            frames.append(Image.open(frame_path).convert("RGB"))

        out_path = sample_dir / name
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=loop,
            optimize=optimize,
        )
        count += 1

    return count


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Root not found: {args.root}")
    created = export_gifs(
        root=args.root,
        num_frames=int(args.num_frames),
        duration_ms=int(args.duration_ms),
        loop=int(args.loop),
        optimize=bool(args.optimize),
        name=str(args.name),
    )
    print(f"gif_done {created}")


if __name__ == "__main__":
    main()
