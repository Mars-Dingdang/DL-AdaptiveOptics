"""Dataset download and preparation utilities.

This script provides a lightweight downloader for public remote sensing datasets.
For datasets without stable direct links, it creates target folders and prints
manual download instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil
import tarfile
import urllib.request
import zipfile


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
    """Download file from URL to local path."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract a zip or tar archive to output directory."""
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and prepare remote sensing datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="uc_merced",
        choices=sorted(DATASET_SOURCES.keys()),
        help="Dataset key to prepare.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Project data root directory.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload and re-extract dataset archive.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    prepare_dataset(dataset_key=args.dataset, data_root=args.data_root, force_download=args.force_download)


if __name__ == "__main__":
    main()
