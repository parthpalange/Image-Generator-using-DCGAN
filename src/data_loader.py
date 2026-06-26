#!/usr/bin/env python3
"""Data-loading utilities for the DCGAN training pipeline.

This module provides:

* :class:`GANDataset` — a :class:`~torch.utils.data.Dataset` that reads
  pre-processed PNG images from a flat directory.
* :func:`get_dataloader` — a convenience factory that wraps *GANDataset* in a
  :class:`~torch.utils.data.DataLoader` with sensible defaults for GAN
  training (``shuffle=True``, ``drop_last=True``, ``pin_memory=True``).
* :func:`show_batch` — quick visual sanity-check that renders a grid of images
  from a single batch via :mod:`matplotlib`.

Usage
-----
.. code-block:: python

    from src.data_loader import get_dataloader, show_batch

    loader = get_dataloader(config)
    show_batch(next(iter(loader)))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---- Logging ----------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---- Constants --------------------------------------------------------------

SUPPORTED_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
}


# ---- Dataset ----------------------------------------------------------------


class GANDataset(Dataset):
    """PyTorch Dataset for pre-processed GAN training images.

    All images are expected to live in a single flat directory (the output of
    ``preprocess.py``).  Each image is loaded, transformed, and normalised to
    the range **[-1, 1]** so that a ``tanh`` output layer in the Generator
    matches directly.

    Parameters
    ----------
    processed_dir : str | Path
        Directory containing the pre-processed images.
    image_size : int
        Spatial resolution to resize / centre-crop to.
    channels : int, optional
        Number of colour channels (default ``3`` for RGB).

    Attributes
    ----------
    image_paths : list[Path]
        Sorted list of discovered image file paths.
    transform : torchvision.transforms.Compose
        The composed transform pipeline applied to every sample.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        image_size: int,
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir).resolve()
        self.image_size = image_size
        self.channels = channels

        # Discover images
        self.image_paths: list[Path] = sorted(
            p
            for p in self.processed_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.processed_dir}.  "
                "Did you run preprocess.py first?"
            )

        logger.info(
            "GANDataset: loaded %d images from %s",
            len(self.image_paths),
            self.processed_dir,
        )

        # Build transform pipeline
        # Resize → CenterCrop ensures exact spatial dims even for edge cases,
        # then normalise each channel to [-1, 1].
        mean = [0.5] * channels
        std = [0.5] * channels

        self.transform: T.Compose = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(image_size),
            T.ToTensor(),                    # → [0, 1]
            T.Normalize(mean, std),          # → [-1, 1]
        ])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load, transform, and return the image at *index*.

        Parameters
        ----------
        index : int
            Index of the image to fetch.

        Returns
        -------
        torch.Tensor
            Image tensor of shape ``(C, H, W)`` normalised to ``[-1, 1]``.
        """
        img_path = self.image_paths[index]

        try:
            image = Image.open(img_path)
            # Ensure correct colour mode
            if self.channels == 3 and image.mode != "RGB":
                image = image.convert("RGB")
            elif self.channels == 1 and image.mode != "L":
                image = image.convert("L")

            return self.transform(image)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s (%s). Returning a black tensor.",
                img_path,
                exc,
            )
            # Return a zero tensor as a safe fallback so training doesn't crash
            return torch.zeros(self.channels, self.image_size, self.image_size)


# ---- DataLoader factory -----------------------------------------------------


def get_dataloader(config: dict) -> DataLoader:
    """Create and return a ready-to-use :class:`DataLoader` for GAN training.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary.  Expected keys::

            data.processed_dir  — path to pre-processed images
            data.image_size     — target spatial resolution
            data.channels       — number of colour channels
            data.num_workers    — DataLoader worker count
            training.batch_size — mini-batch size

    Returns
    -------
    DataLoader
        A DataLoader yielding batches of shape
        ``(batch_size, channels, image_size, image_size)`` in ``[-1, 1]``.
    """
    data_cfg: dict[str, Any] = config["data"]
    training_cfg: dict[str, Any] = config["training"]

    dataset = GANDataset(
        processed_dir=data_cfg["processed_dir"],
        image_size=int(data_cfg["image_size"]),
        channels=int(data_cfg["channels"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        drop_last=True,
        pin_memory=True,
        persistent_workers=int(data_cfg["num_workers"]) > 0,
    )

    logger.info(
        "DataLoader ready — %d images, batch_size=%d, num_workers=%d",
        len(dataset),
        int(training_cfg["batch_size"]),
        int(data_cfg["num_workers"]),
    )
    return loader


# ---- Visualisation -----------------------------------------------------------


def show_batch(
    batch: torch.Tensor,
    *,
    nrow: int = 8,
    title: str = "Training Batch",
    figsize: tuple[int, int] = (12, 12),
    save_path: str | Path | None = None,
) -> None:
    """Display (and optionally save) a grid of images from a single batch.

    The images are expected to be in the range ``[-1, 1]`` and will be
    de-normalised back to ``[0, 1]`` for display.

    Parameters
    ----------
    batch : torch.Tensor
        Batch tensor of shape ``(B, C, H, W)`` in ``[-1, 1]``.
    nrow : int, optional
        Number of images per row in the grid (default ``8``).
    title : str, optional
        Title shown above the plot.
    figsize : tuple[int, int], optional
        Matplotlib figure size.
    save_path : str | Path | None, optional
        If provided, the figure is saved to this path instead of displayed.
    """
    # De-normalise: [-1, 1] → [0, 1]
    batch_denorm = batch.detach().cpu() * 0.5 + 0.5
    batch_denorm = torch.clamp(batch_denorm, 0.0, 1.0)

    grid = vutils.make_grid(batch_denorm, nrow=nrow, padding=2, normalize=False)
    grid_np: np.ndarray = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid_np)
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Batch grid saved to %s", save_path)
        plt.close(fig)
    else:
        plt.show()


# ---- Convenience: load config + build loader --------------------------------


def load_config(config_path: str | Path) -> dict:
    """Load and return a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---- CLI entry-point (quick sanity check) ------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Quick sanity-check: load a batch and display a grid.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save the grid image to this path instead of displaying.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dl = get_dataloader(cfg)
    sample_batch = next(iter(dl))

    print(f"Batch shape : {sample_batch.shape}")
    print(f"Value range : [{sample_batch.min().item():.3f}, {sample_batch.max().item():.3f}]")

    show_batch(sample_batch, save_path=args.save)
