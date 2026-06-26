"""
DCGAN Training Script
=====================
Train a Deep Convolutional GAN on image datasets using PyTorch.

Usage:
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --resume checkpoints/epoch_50.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path so we can import sibling modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import Generator
from src.discriminator import Discriminator
from src.data_loader import get_dataloader

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight initialisation (DCGAN paper – N(0, 0.02))
# ---------------------------------------------------------------------------
def weights_init(m: nn.Module) -> None:
    """Apply DCGAN-style weight initialisation.

    Conv and ConvTranspose layers are initialised from N(0, 0.02).
    BatchNorm layers are initialised with weight ~ N(1.0, 0.02), bias = 0.

    Note: The Generator and Discriminator classes already apply their own
    ``_init_weights`` in ``__init__``.  This function is available as an
    alternative entry-point when constructing models outside of those
    classes, or when re-initialising after loading a partial checkpoint.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        weight = getattr(m, "weight_orig", None) or getattr(m, "weight", None)
        if weight is not None:
            nn.init.normal_(weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    netG: nn.Module,
    netD: nn.Module,
    optimG: torch.optim.Optimizer,
    optimD: torch.optim.Optimizer,
    config: dict,
) -> None:
    """Persist training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator_state_dict": netG.state_dict(),
            "discriminator_state_dict": netD.state_dict(),
            "optimG_state_dict": optimG.state_dict(),
            "optimD_state_dict": optimD.state_dict(),
            "config": config,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    netG: nn.Module,
    netD: nn.Module,
    optimG: torch.optim.Optimizer,
    optimD: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    """Restore training state from a checkpoint. Returns (start_epoch, global_step)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netD.load_state_dict(ckpt["discriminator_state_dict"])
    optimG.load_state_dict(ckpt["optimG_state_dict"])
    optimD.load_state_dict(ckpt["optimD_state_dict"])
    start_epoch = ckpt["epoch"] + 1  # resume from *next* epoch
    global_step = ckpt.get("global_step", 0)
    logger.info("Resumed from checkpoint %s (epoch %d)", path, ckpt["epoch"])
    return start_epoch, global_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(config: dict, resume_path: str | None = None) -> None:
    """Run the full DCGAN training loop."""

    # ── unpack config ─────────────────────────────────────────────────────
    latent_dim = config["model"]["latent_dim"]
    g_features = config["model"]["g_features"]
    d_features = config["model"]["d_features"]
    nc = config["data"]["channels"]
    image_size = config["data"]["image_size"]

    num_epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr_g = config["training"]["lr_g"]
    lr_d = config["training"]["lr_d"]
    beta1 = config["training"]["beta1"]
    beta2 = config["training"]["beta2"]
    label_smoothing = config["training"].get("label_smoothing", 0.9)
    save_interval = config["training"]["save_interval"]
    sample_interval = config["training"]["sample_interval"]
    num_workers = config["data"].get("num_workers", 4)

    model_dir = Path(config["output"]["model_dir"])
    output_dir = Path(config["output"]["output_dir"])
    log_dir = Path(config["output"]["log_dir"])

    checkpoint_dir = model_dir / "checkpoints"
    sample_dir = output_dir / "samples"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────
    device_cfg = config["training"].get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("Using device: %s", device)

    # ── data ──────────────────────────────────────────────────────────────
    dataloader = get_dataloader(config)
    logger.info(
        "Dataset size: %d images  |  Batches per epoch: %d",
        len(dataloader.dataset),
        len(dataloader),
    )

    # ── models ────────────────────────────────────────────────────────────
    netG = Generator(
        latent_dim=latent_dim, num_channels=nc, g_features=g_features
    ).to(device)
    netD = Discriminator(
        num_channels=nc, d_features=d_features
    ).to(device)

    # Models already apply their own _init_weights, but we re-apply here
    # to ensure consistent initialisation even if constructor changes.
    netG.apply(weights_init)
    netD.apply(weights_init)

    logger.info("Generator:\n%s", netG)
    logger.info("Discriminator:\n%s", netD)

    # ── loss & optimisers ─────────────────────────────────────────────────
    criterion = nn.BCELoss()

    optimG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            Path(resume_path), netG, netD, optimG, optimD, device
        )

    # ── fixed noise for visualisation ─────────────────────────────────────
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── label constants (with smoothing) ──────────────────────────────────
    real_label_val = label_smoothing  # one-sided label smoothing (default 0.9)
    fake_label_val = 0.0

    # ── training ──────────────────────────────────────────────────────────
    logger.info(
        "Starting training  |  epochs %d→%d  |  batch_size %d  |  lr_g %.5f  |  lr_d %.5f",
        start_epoch,
        num_epochs - 1,
        batch_size,
        lr_g,
        lr_d,
    )

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch}/{num_epochs - 1}]",
            leave=True,
        )

        for i, real_images in pbar:
            # data_loader.GANDataset.__getitem__ returns a single tensor;
            # handle both (tensor,) and (tensor, label) formats gracefully.
            if isinstance(real_images, (list, tuple)):
                real_images = real_images[0]

            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # Labels
            real_labels = torch.full(
                (b_size,), real_label_val, dtype=torch.float, device=device
            )
            fake_labels = torch.full(
                (b_size,), fake_label_val, dtype=torch.float, device=device
            )

            # ━━━ Update Discriminator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Maximise  log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()

            # Real batch
            output_real = netD(real_images).view(-1)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Fake batch
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, fake_labels)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimD.step()

            # ━━━ Update Generator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Maximise  log(D(G(z)))
            netG.zero_grad()

            # We want D to classify fakes as real → use real_labels
            output_fake2 = netD(fake_images).view(-1)
            errG = criterion(output_fake2, real_labels)
            errG.backward()
            D_G_z2 = output_fake2.mean().item()
            optimG.step()

            # ── logging ───────────────────────────────────────────────────
            writer.add_scalar("Loss/Discriminator", errD.item(), global_step)
            writer.add_scalar("Loss/Generator", errG.item(), global_step)
            writer.add_scalar("D(x)", D_x, global_step)
            writer.add_scalar("D(G(z))/before_update", D_G_z1, global_step)
            writer.add_scalar("D(G(z))/after_update", D_G_z2, global_step)

            pbar.set_postfix(
                {
                    "D_loss": f"{errD.item():.4f}",
                    "G_loss": f"{errG.item():.4f}",
                    "D(x)": f"{D_x:.4f}",
                    "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}",
                }
            )

            # ── sample images ────────────────────────────────────────────
            if global_step % sample_interval == 0:
                with torch.no_grad():
                    fake_samples = netG(fixed_noise).detach().cpu()
                grid = vutils.make_grid(
                    fake_samples, nrow=8, padding=2, normalize=True
                )
                writer.add_image("Generated Samples", grid, global_step)
                save_path = sample_dir / f"step_{global_step:06d}.png"
                vutils.save_image(
                    fake_samples, str(save_path), nrow=8, padding=2, normalize=True
                )

            global_step += 1

        # ── epoch-level checkpoint ────────────────────────────────────────
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
            save_checkpoint(
                ckpt_path, epoch, global_step, netG, netD, optimG, optimD, config
            )

    # ── save final models ─────────────────────────────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(netG.state_dict(), model_dir / "generator_final.pth")
    torch.save(netD.state_dict(), model_dir / "discriminator_final.pth")
    logger.info("Final models saved to %s/", model_dir)

    writer.close()
    logger.info("Training complete ✓")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DCGAN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)
