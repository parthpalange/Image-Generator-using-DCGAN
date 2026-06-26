"""
DCGAN Evaluation Script
=======================
Evaluate a trained DCGAN generator by computing FID and Inception Score.

Usage:
    python evaluate.py --config config/default.yaml --checkpoint models/generator_final.pth
    python evaluate.py --config config/default.yaml --checkpoint models/generator_final.pth --num_images 5000
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import Generator
from src.data_loader import get_dataloader

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
# InceptionV3 feature extractor (for FID / IS)
# ---------------------------------------------------------------------------
class InceptionFeatureExtractor(nn.Module):
    """Wraps torchvision InceptionV3 to extract pool-3 features (2048-d)
    and class logits (1000-d)."""

    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        self.model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=False
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

        # Register a forward-hook on the avg-pool layer to grab features
        self._features: torch.Tensor | None = None
        self.model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    @torch.no_grad()
    def get_features_and_logits(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features [B, 2048], logits [B, 1000]) for *images* in [-1, 1]."""
        # Inception expects 299×299, values in [0, 1]
        images = nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        images = (images + 1.0) / 2.0  # [-1, 1] → [0, 1]
        images = images.to(self.device)

        logits = self.model(images)
        features = self._features.view(images.size(0), -1)
        return features.cpu(), logits.cpu()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_statistics(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and covariance of feature vectors."""
    mu = features.mean(dim=0)
    diff = features - mu.unsqueeze(0)
    sigma = (diff.T @ diff) / (features.size(0) - 1)
    return mu, sigma


def compute_fid(
    mu_real: torch.Tensor,
    sigma_real: torch.Tensor,
    mu_fake: torch.Tensor,
    sigma_fake: torch.Tensor,
) -> float:
    """Compute the Fréchet Inception Distance (FID).

    FID = ||mu_r - mu_f||^2 + Tr(sigma_r + sigma_f - 2 * sqrtm(sigma_r @ sigma_f))
    """
    import numpy as np
    from scipy.linalg import sqrtm

    mu_r = mu_real.numpy().astype(np.float64)
    mu_f = mu_fake.numpy().astype(np.float64)
    sigma_r = sigma_real.numpy().astype(np.float64)
    sigma_f = sigma_fake.numpy().astype(np.float64)

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)

    # Numerical stability: remove imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            logger.warning(
                "Imaginary component in sqrtm result (max imag = %.4e). "
                "Results may be inaccurate.",
                np.max(np.abs(covmean.imag)),
            )
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))
    return fid


def compute_inception_score(
    logits: torch.Tensor, splits: int = 10
) -> tuple[float, float]:
    """Compute the Inception Score (IS) with *splits* for mean ± std."""
    import numpy as np

    probs = torch.nn.functional.softmax(logits, dim=1).numpy()
    scores = []
    n = probs.shape[0]
    split_size = n // splits

    for k in range(splits):
        part = probs[k * split_size : (k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl_mean = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl_mean))

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_images(
    netG: nn.Module,
    latent_dim: int,
    num_images: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate *num_images* fake images and return them as a single tensor."""
    netG.eval()
    all_images = []
    remaining = num_images
    pbar = tqdm(total=num_images, desc="Generating images")

    while remaining > 0:
        bs = min(batch_size, remaining)
        noise = torch.randn(bs, latent_dim, 1, 1, device=device)
        fake = netG(noise).cpu()
        all_images.append(fake)
        remaining -= bs
        pbar.update(bs)

    pbar.close()
    return torch.cat(all_images, dim=0)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate(config: dict, checkpoint_path: str, num_images: int) -> dict:
    """Run full evaluation pipeline. Returns a metrics dict."""

    # ── device ────────────────────────────────────────────────────────────
    device_cfg = config["training"].get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("Using device: %s", device)

    latent_dim = config["model"]["latent_dim"]
    g_features = config["model"]["g_features"]
    nc = config["data"]["channels"]
    batch_size = config["training"]["batch_size"]

    # ── Load generator ────────────────────────────────────────────────────
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

    # ── Generate fake images ──────────────────────────────────────────────
    logger.info("Generating %d fake images …", num_images)
    fake_images = generate_images(netG, latent_dim, num_images, batch_size, device)

    # ── Save samples ──────────────────────────────────────────────────────
    eval_dir = Path(config["output"]["output_dir"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = eval_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Save first 64 as a grid
    grid = vutils.make_grid(fake_images[:64], nrow=8, padding=2, normalize=True)
    vutils.save_image(grid, str(eval_dir / "sample_grid.png"))
    logger.info("Sample grid saved → %s", eval_dir / "sample_grid.png")

    # Save individual images (up to 100)
    num_save = min(num_images, 100)
    for idx in range(num_save):
        vutils.save_image(
            fake_images[idx],
            str(samples_dir / f"sample_{idx:05d}.png"),
            normalize=True,
        )
    logger.info("Saved %d individual samples → %s/", num_save, samples_dir)

    # ── Inception feature extraction ──────────────────────────────────────
    logger.info("Extracting Inception features (this may take a while) …")
    inception = InceptionFeatureExtractor(device)

    # Fake features
    fake_features_list, fake_logits_list = [], []
    fake_loader = DataLoader(
        TensorDataset(fake_images), batch_size=batch_size, shuffle=False
    )
    for (batch,) in tqdm(fake_loader, desc="Inception (fake)"):
        feats, logits = inception.get_features_and_logits(batch)
        fake_features_list.append(feats)
        fake_logits_list.append(logits)

    fake_features = torch.cat(fake_features_list, dim=0)
    fake_logits = torch.cat(fake_logits_list, dim=0)

    # Real features (for FID)
    real_dataloader = get_dataloader(config)
    real_features_list = []
    collected = 0
    for real_batch in tqdm(real_dataloader, desc="Inception (real)"):
        # data_loader returns tensors directly (no labels)
        if isinstance(real_batch, (list, tuple)):
            real_batch = real_batch[0]
        feats, _ = inception.get_features_and_logits(real_batch)
        real_features_list.append(feats)
        collected += real_batch.size(0)
        if collected >= num_images:
            break

    real_features = torch.cat(real_features_list, dim=0)[:num_images]

    # ── Compute metrics ───────────────────────────────────────────────────
    logger.info("Computing FID …")
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    logger.info("Computing Inception Score …")
    is_mean, is_std = compute_inception_score(fake_logits)

    metrics = {
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "num_images_evaluated": num_images,
    }

    # ── Report ────────────────────────────────────────────────────────────
    report_path = eval_dir / "quality_report.txt"
    separator = "=" * 55
    report_lines = [
        separator,
        "  DCGAN Evaluation Report",
        separator,
        f"  Checkpoint       : {checkpoint_path}",
        f"  Images evaluated : {num_images}",
        f"  Device           : {device}",
        separator,
        f"  FID              : {fid:.4f}",
        f"  Inception Score  : {is_mean:.4f} ± {is_std:.4f}",
        separator,
        "",
        "  Lower FID is better (0 = identical distributions).",
        "  Higher IS is better (max ≈ 1000 for ImageNet classes).",
        separator,
    ]
    report_text = "\n".join(report_lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text + "\n")
    logger.info("Report saved → %s", report_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DCGAN generator (FID & IS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained generator checkpoint (.pth).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate for evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, args.num_images)
