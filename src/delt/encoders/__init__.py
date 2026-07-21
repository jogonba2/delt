"""Package for encoders."""

from typing import Any

from .audio import ClapEncoder, GlapEncoder
from .base import Encoder
from .image import ClipEncoder, SiglipEncoder
from .text import SentenceTransformerEncoder
from .video import XclipEncoder

REGISTRY: dict[str, type[Encoder]] = {
    "glap": GlapEncoder,
    "clap": ClapEncoder,
    "clip": ClipEncoder,
    "siglip": SiglipEncoder,
    "sentence-transformer": SentenceTransformerEncoder,
    "xclip": XclipEncoder,
}


def get_encoder(class_name: str, *args: Any, **kwargs: Any) -> Encoder:
    """
    Instantiate an encoder by class name, given arbitrary arguments.

    Args:
        class_name (str): class name of the encoder. It should be available in `REGISTRY`.
        *args: positional arguments passed directly to the encoder's constructor.
        **kwargs: keyword arguments passed directly to the encoder's constructor.

    Returns:
        Encoder: an instantiated encoder.

    """
    if class_name not in REGISTRY:
        raise KeyError(
            f"Unknown encoder class: {class_name}. Available: {list(REGISTRY.keys())}"
        )

    encoder_cls = REGISTRY[class_name]
    return encoder_cls(*args, **kwargs)


__all__ = [
    "Encoder",
    "SentenceTransformerEncoder",
    "ClipEncoder",
    "GlapEncoder",
    "SiglipEncoder",
    "ClapEncoder",
    "XclipEncoder",
    "get_encoder",
]
