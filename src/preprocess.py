#!/usr/bin/env python3
"""Preprocessing script for the DCGAN training pipeline.

Walks through a directory tree of raw images, resizes each to the configured
spatial resolution, converts to RGB when necessary, and writes the results to
a flat output directory ready for ``data_loader.py``.

Usage
-----
.. code-block:: bash

    python src/preprocess.py --config config/default.yaml

Notes
-----
* Supports JPEG, PNG, BMP, TIFF, and WebP.
* Corrupt or unreadable files are skipped with a logged warning — the script
  never crashes on a single bad image.
* A progress bar (``tqdm``) and a final summary are printed to *stdout*.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Set

import yaml
from PIL import Image
from tqdm import tqdm

# ---- Constants --------------------------------------------------------------

SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
}

# ---- Logging ----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---- Helpers ----------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    """Load and return the YAML configuration as a dictionary.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)

    logger.info("Loaded configuration from %s", config_path)
    return config


def collect_image_paths(raw_dir: Path) -> list[Path]:
    """Recursively collect all image file paths under *raw_dir*.

    Parameters
    ----------
    raw_dir : Path
        Root directory to walk.

    Returns
    -------
    list[Path]
        Sorted list of absolute image paths.
    """
    paths: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
    return paths


def process_image(
    src: Path,
    dst: Path,
    image_size: int,
) -> bool:
    """Resize a single image to ``(image_size, image_size)`` and save as PNG.

    Parameters
    ----------
    src : Path
        Source image path.
    dst : Path
        Destination file path (will be written as PNG).
    image_size : int
        Target spatial resolution (square).

    Returns
    -------
    bool
        ``True`` if the image was processed successfully, ``False`` otherwise.
    """
    try:
        with Image.open(src) as img:
            # Ensure RGB (handles grayscale, RGBA, palette, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # High-quality downscale via Lanczos resampling
            img_resized = img.resize(
                (image_size, image_size),
                resample=Image.LANCZOS,
            )
            img_resized.save(dst, format="PNG")
        return True

    except Exception as exc:  # noqa: BLE001 — intentionally broad
        logger.warning("Skipping %s — %s: %s", src, type(exc).__name__, exc)
        return False


# ---- Main -------------------------------------------------------------------


def preprocess(config: dict) -> None:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration (must contain a ``data`` section).
    """
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"]).resolve()
    processed_dir = Path(data_cfg["processed_dir"]).resolve()
    image_size: int = int(data_cfg["image_size"])

    # Validate source directory
    if not raw_dir.is_dir():
        logger.error("Raw image directory does not exist: %s", raw_dir)
        sys.exit(1)

    # Create destination directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Raw directory      : %s", raw_dir)
    logger.info("Processed directory: %s", processed_dir)
    logger.info("Target size        : %d × %d", image_size, image_size)

    # Discover images
    image_paths = collect_image_paths(raw_dir)
    total_found = len(image_paths)
    if total_found == 0:
        logger.warning("No supported images found in %s", raw_dir)
        return

    logger.info("Found %d image(s) to process.", total_found)

    # Process with progress bar
    processed_count = 0
    skipped_count = 0
    start_time = time.perf_counter()

    for idx, src_path in enumerate(
        tqdm(image_paths, desc="Processing", unit="img", dynamic_ncols=True),
    ):
        # Build a unique flat filename: <index>_<original_stem>.png
        dst_name = f"{idx:06d}_{src_path.stem}.png"
        dst_path = processed_dir / dst_name

        success = process_image(src_path, dst_path, image_size)
        if success:
            processed_count += 1
        else:
            skipped_count += 1

    elapsed = time.perf_counter() - start_time

    # Summary
    logger.info("=" * 50)
    logger.info("Preprocessing complete.")
    logger.info("  Total found  : %d", total_found)
    logger.info("  Processed    : %d", processed_count)
    logger.info("  Skipped      : %d", skipped_count)
    logger.info("  Time elapsed : %.2f s", elapsed)
    logger.info("  Output dir   : %s", processed_dir)
    logger.info("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``config`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess raw images for DCGAN training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the YAML configuration file (default: config/default.yaml).",
    )
    return parser.parse_args()


# ---- Entry-point ------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    preprocess(cfg)
