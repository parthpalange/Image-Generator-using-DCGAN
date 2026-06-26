"""DCGAN Discriminator module.

Implements the Discriminator network based on the DCGAN paper (Radford et al., 2015),
extended to accept 256×256 images via strided convolutions.

Architecture overview:
    (num_channels, 256, 256) -> Conv2d blocks with BatchNorm + LeakyReLU(0.2)
    -> (1, 1, 1) -> Sigmoid -> scalar ∈ [0, 1]

    Downsampling path: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1

Spectral normalisation (Miyato et al., 2018) can be optionally applied to all
convolutional layers for improved training stability.

Typical usage:
    >>> disc = Discriminator(num_channels=3, d_features=64)
    >>> score = disc(images)  # images: (B, 3, 256, 256) -> score: (B, 1, 1, 1)

    >>> disc_sn = Discriminator(num_channels=3, d_features=64, use_spectral_norm=True)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

__all__ = ["Discriminator"]


def _get_norm_wrapper(use_spectral_norm: bool) -> Callable[[nn.Module], nn.Module]:
    """Return either ``spectral_norm`` or the identity function.

    Args:
        use_spectral_norm: Whether to wrap convolution layers with spectral
            normalisation.

    Returns:
        A callable that takes an ``nn.Module`` and returns it (possibly
        wrapped).
    """
    if use_spectral_norm:
        return spectral_norm
    return lambda m: m  # identity — no-op wrapper


class Discriminator(nn.Module):
    """DCGAN Discriminator network (256×256).

    Classifies a 256×256 image as real or fake, producing a scalar probability
    via a stack of strided Conv2d layers.

    Args:
        num_channels: Number of channels in the input image (3 for RGB).
        d_features: Base number of feature maps.  Intermediate layers use
            multiples of this value (×1, ×2, ×4, ×8, ×16).
        use_spectral_norm: If ``True``, applies spectral normalisation to
            **all** convolutional layers.  When spectral norm is enabled the
            ``BatchNorm2d`` layers are retained by default; set
            ``use_batchnorm=False`` explicitly in a subclass if you prefer
            to drop them (common in SN-GAN variants).
    """

    def __init__(
        self,
        num_channels: int = 3,
        d_features: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.d_features = d_features
        self.use_spectral_norm = use_spectral_norm

        sn: Callable[[nn.Module], nn.Module] = _get_norm_wrapper(use_spectral_norm)

        self.main = nn.Sequential(
            # ---------------------------------------------------------
            # Layer 1: (num_channels, 256, 256) -> (d_f, 128, 128)
            # No BatchNorm on first layer (DCGAN convention).
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=d_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # ---------------------------------------------------------
            # Layer 2: (d_f, 128, 128) -> (d_f*2, 64, 64)
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=d_features,
                    out_channels=d_features * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(d_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ---------------------------------------------------------
            # Layer 3: (d_f*2, 64, 64) -> (d_f*4, 32, 32)
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=d_features * 2,
                    out_channels=d_features * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(d_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ---------------------------------------------------------
            # Layer 4: (d_f*4, 32, 32) -> (d_f*8, 16, 16)
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=d_features * 4,
                    out_channels=d_features * 8,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(d_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ---------------------------------------------------------
            # Layer 5: (d_f*8, 16, 16) -> (d_f*16, 8, 8)
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=d_features * 8,
                    out_channels=d_features * 16,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(d_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ---------------------------------------------------------
            # Layer 6: (d_f*16, 8, 8) -> (d_f*16, 4, 4)
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=d_features * 16,
                    out_channels=d_features * 16,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(d_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # ---------------------------------------------------------
            # Layer 7 (output): (d_f*16, 4, 4) -> (1, 1, 1)
            # ---------------------------------------------------------
            sn(
                nn.Conv2d(
                    in_channels=d_features * 16,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.Sigmoid(),
        )

        # Apply DCGAN weight initialisation.
        # NOTE: When spectral_norm is active the wrapped weight is stored as
        # ``weight_orig``; ``_init_weights`` targets ``weight.data`` which
        # is the *computed* weight — ``nn.init`` on ``weight_orig`` is
        # handled correctly by ``apply`` because spectral_norm registers a
        # forward hook, and ``_init_weights`` still reaches the underlying
        # parameter via ``module.weight_orig`` when present.
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise Conv and BatchNorm layers following DCGAN conventions.

        * Convolutional weights are drawn from ``N(0, 0.02)``.
          If the layer is wrapped with spectral normalisation the underlying
          ``weight_orig`` parameter is initialised instead.
        * BatchNorm weights (gamma) are initialised to ``N(1, 0.02)`` and
          biases (beta) are set to ``0``.

        Args:
            module: A single ``nn.Module`` (called via ``self.apply``).
        """
        classname = module.__class__.__name__
        if "Conv" in classname:
            # Spectral norm stores the raw weight as ``weight_orig``.
            weight = getattr(module, "weight_orig", None)
            if weight is None:
                weight = getattr(module, "weight", None)
            if weight is not None:
                nn.init.normal_(weight.data, mean=0.0, std=0.02)
        elif "BatchNorm" in classname:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(module.bias.data, val=0.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images as real or fake.

        Args:
            images: Input images of shape ``(batch, num_channels, 256, 256)``
                with pixel values ideally in ``[-1, 1]``.

        Returns:
            Probability scores of shape ``(batch, 1, 1, 1)``.  Squeeze as
            needed, e.g. ``disc(x).view(-1)`` for a flat vector of scores.
        """
        return self.main(images)

    def __repr__(self) -> str:  # pragma: no cover
        sn_str = ", spectral_norm=True" if self.use_spectral_norm else ""
        return (
            f"{self.__class__.__name__}("
            f"num_channels={self.num_channels}, "
            f"d_features={self.d_features}"
            f"{sn_str})"
        )

