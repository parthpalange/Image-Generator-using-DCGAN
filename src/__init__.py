"""DCGAN model components.

Exports:
    Generator -- DCGAN generator network (latent_dim -> 64×64 image).
    Discriminator -- DCGAN discriminator network (64×64 image -> real/fake).
"""

from src.generator import Generator
from src.discriminator import Discriminator

__all__ = ["Generator", "Discriminator"]
