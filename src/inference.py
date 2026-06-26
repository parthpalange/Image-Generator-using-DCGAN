"""
DCGAN Inference / Generation Script
====================================
Generate images from a trained DCGAN generator.

Usage:
    python inference.py --checkpoint models/generator_final.pth --num_images 16
    python inference.py --checkpoint models/generator_final.pth --num_images 16 --seed 42
    python inference.py --checkpoint models/generator_final.pth --interpolate --steps 10
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import Generator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_generator(
    checkpoint_path: str, config: dict, device: torch.device
) -> Generator:
    """Instantiate a Generator and load trained weights."""
    latent_dim = config["model"]["latent_dim"]
    g_features = config["model"]["g_features"]
    nc = config["data"]["channels"]

    netG = Generator(
        latent_dim=latent_dim, num_channels=nc, g_features=g_features
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle both raw state_dict and full-checkpoint formats
    if "generator_state_dict" in state:
        netG.load_state_dict(state["generator_state_dict"])
    else:
        netG.load_state_dict(state)

    netG.eval()
    logger.info("Generator loaded from %s", checkpoint_path)
    return netG


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_random(
    netG: Generator,
    latent_dim: int,
    num_images: int,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate *num_images* random images. Returns tensor on CPU."""
    if seed is not None:
        torch.manual_seed(seed)
        logger.info("Using seed: %d", seed)

    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    fake = netG(noise).cpu()
    return fake


@torch.no_grad()
def interpolate_latent(
    netG: Generator,
    latent_dim: int,
    steps: int,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    """Spherical-linear interpolation between two random latent vectors.

    Returns a tensor of shape (steps, C, H, W) on CPU.
    """
    if seed is not None:
        torch.manual_seed(seed)
        logger.info("Using seed: %d", seed)

    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)

    # Slerp on the flattened vectors, then reshape back
    z1_flat = z1.view(1, -1)
    z2_flat = z2.view(1, -1)

    # Normalise for slerp
    z1_norm = z1_flat / z1_flat.norm(dim=1, keepdim=True)
    z2_norm = z2_flat / z2_flat.norm(dim=1, keepdim=True)

    omega = torch.acos(
        torch.clamp((z1_norm * z2_norm).sum(dim=1, keepdim=True), -1.0, 1.0)
    )

    alphas = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)

    # If omega ≈ 0 fall back to linear interpolation
    if omega.abs().item() < 1e-6:
        z_interp = (1.0 - alphas) * z1_flat + alphas * z2_flat
    else:
        sin_omega = torch.sin(omega)
        z_interp = (
            torch.sin((1.0 - alphas) * omega) / sin_omega * z1_flat
            + torch.sin(alphas * omega) / sin_omega * z2_flat
        )

    z_interp = z_interp.view(steps, latent_dim, 1, 1)
    fake = netG(z_interp).cpu()
    return fake


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_images(
    images: torch.Tensor,
    output_dir: Path,
    prefix: str = "generated",
    nrow: int = 8,
) -> None:
    """Save individual images and a combined grid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grid
    grid_path = output_dir / f"{prefix}_grid.png"
    vutils.save_image(images, str(grid_path), nrow=nrow, padding=2, normalize=True)
    logger.info("Grid saved → %s", grid_path)

    # Individual images
    for idx in range(images.size(0)):
        img_path = output_dir / f"{prefix}_{idx:04d}.png"
        vutils.save_image(images[idx], str(img_path), normalize=True)

    logger.info("Saved %d individual images → %s/", images.size(0), output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_inference(args: argparse.Namespace) -> None:
    """Execute inference based on parsed CLI arguments."""

    # ── device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load config ───────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    latent_dim = config["model"]["latent_dim"]

    # ── Load generator ────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    netG = load_generator(str(checkpoint_path), config, device)

    output_dir = Path(args.output)

    # ── Mode: interpolation ───────────────────────────────────────────────
    if args.interpolate:
        steps = args.steps
        logger.info("Generating latent-space interpolation (%d steps) …", steps)
        images = interpolate_latent(netG, latent_dim, steps, device, seed=args.seed)
        save_images(images, output_dir, prefix="interpolation", nrow=steps)
        _print_summary("Interpolation", steps, output_dir, args.seed)
        return

    # ── Mode: random generation ───────────────────────────────────────────
    num_images = args.num_images
    logger.info("Generating %d random images …", num_images)
    images = generate_random(netG, latent_dim, num_images, device, seed=args.seed)
    save_images(images, output_dir, prefix="generated", nrow=min(8, num_images))
    _print_summary("Random Generation", num_images, output_dir, args.seed)


def _print_summary(
    mode: str, count: int, output_dir: Path, seed: int | None
) -> None:
    """Print a formatted generation summary to stdout."""
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Generation Summary")
    print(f"{sep}")
    print(f"  Mode       : {mode}")
    print(f"  Count      : {count}")
    print(f"  Seed       : {seed if seed is not None else 'random'}")
    print(f"  Output dir : {output_dir.resolve()}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images with a trained DCGAN generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained generator checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the YAML configuration file (needed for model architecture).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=16,
        help="Number of random images to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Enable latent-space interpolation mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of interpolation steps (used with --interpolate).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
