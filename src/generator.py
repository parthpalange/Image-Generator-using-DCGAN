"""DCGAN Generator module.

Implements the Generator network based on the DCGAN paper (Radford et al., 2015),
extended to produce 256×256 RGB images via fractionally-strided convolutions
(ConvTranspose2d).

Architecture overview:
    z (latent_dim,) -> project & reshape -> (g_features*16, 4, 4)
    -> ConvTranspose2d blocks with BatchNorm + ReLU
    -> (num_channels, 256, 256) with Tanh activation

    Upsampling path: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256

Typical usage:
    >>> gen = Generator(latent_dim=100, num_channels=3, g_features=64)
    >>> fake_images = gen.generate(num_samples=16)  # (16, 3, 256, 256)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["Generator"]


class Generator(nn.Module):
    """DCGAN Generator network (256×256).

    Transforms a latent vector of shape ``(batch, latent_dim)`` into a
    synthetic image of shape ``(batch, num_channels, 256, 256)`` with pixel
    values in [-1, 1].

    Args:
        latent_dim: Dimensionality of the latent / noise vector *z*.
        num_channels: Number of channels in the output image (3 for RGB).
        g_features: Base number of feature maps.  Intermediate layers use
            multiples of this value (×16, ×8, ×4, ×2, ×1).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_channels: int = 3,
        g_features: int = 64,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.g_features = g_features

        self.main = nn.Sequential(
            # ---------------------------------------------------------
            # Layer 1: Project & reshape — (latent_dim, 1, 1) -> (g_f*16, 4, 4)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=g_features * 16,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(g_features * 16),
            nn.ReLU(inplace=True),
            # ---------------------------------------------------------
            # Layer 2: (g_f*16, 4, 4) -> (g_f*8, 8, 8)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=g_features * 16,
                out_channels=g_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(g_features * 8),
            nn.ReLU(inplace=True),
            # ---------------------------------------------------------
            # Layer 3: (g_f*8, 8, 8) -> (g_f*4, 16, 16)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=g_features * 8,
                out_channels=g_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(g_features * 4),
            nn.ReLU(inplace=True),
            # ---------------------------------------------------------
            # Layer 4: (g_f*4, 16, 16) -> (g_f*2, 32, 32)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=g_features * 4,
                out_channels=g_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(g_features * 2),
            nn.ReLU(inplace=True),
            # ---------------------------------------------------------
            # Layer 5: (g_f*2, 32, 32) -> (g_f, 64, 64)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=g_features * 2,
                out_channels=g_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(g_features),
            nn.ReLU(inplace=True),
            # ---------------------------------------------------------
            # Layer 6: (g_f, 64, 64) -> (g_f//2, 128, 128)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=g_features,
                out_channels=g_features // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(g_features // 2),
            nn.ReLU(inplace=True),
            # ---------------------------------------------------------
            # Layer 7 (output): (g_f//2, 128, 128) -> (num_channels, 256, 256)
            # ---------------------------------------------------------
            nn.ConvTranspose2d(
                in_channels=g_features // 2,
                out_channels=num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        # Apply DCGAN weight initialisation.
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise Conv and BatchNorm layers following DCGAN conventions.

        * Convolutional / transposed-convolutional weights are drawn from
          ``N(0, 0.02)``.
        * BatchNorm weights (gamma) are initialised to ``N(1, 0.02)`` and
          biases (beta) are set to ``0``.

        Args:
            module: A single ``nn.Module`` (called via ``self.apply``).
        """
        classname = module.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(module.bias.data, val=0.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Run the generator on a batch of latent vectors.

        Args:
            z: Latent vectors of shape ``(batch, latent_dim)`` **or**
                ``(batch, latent_dim, 1, 1)``.  If 2-D input is provided it
                is automatically reshaped to 4-D for the transposed
                convolution stack.

        Returns:
            Fake images of shape ``(batch, num_channels, 256, 256)`` with
            values in ``[-1, 1]``.
        """
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim) -> (B, latent_dim, 1, 1)
        return self.main(z)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate fake images from random latent vectors.

        This is a convenience wrapper around :meth:`forward` that handles
        noise sampling and sets the model to eval mode for inference.

        Args:
            num_samples: Number of images to generate.
            device: Device to place the noise tensor on.  Defaults to the
                device of the first model parameter.

        Returns:
            Generated images of shape ``(num_samples, num_channels, 256, 256)``
            with values in ``[-1, 1]``.
        """
        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()

        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        images = self.forward(z)

        if was_training:
            self.train()

        return images

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"latent_dim={self.latent_dim}, "
            f"num_channels={self.num_channels}, "
            f"g_features={self.g_features})"
        )

