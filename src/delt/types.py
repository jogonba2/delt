"""Types used in every model."""

from dataclasses import dataclass
from typing import TypeAlias

import torch
from datasets.features._torchcodec import AudioDecoder
from PIL import Image as PILImage
from torchcodec.decoders._video_decoder import VideoDecoder


@dataclass
class Prediction:
    """A sample prediction."""

    probs: list[float]
    label: int


@dataclass
class SpanAnnotation:
    """A gold span annotation, with character offsets (`end` is exclusive)."""

    start: int
    end: int
    label: int | str


@dataclass
class Span:
    """A predicted span, with character offsets (`end` is exclusive)."""

    start: int
    end: int
    text: str
    label: int
    score: float
    probs: list[float]


@dataclass
class TrainingOutput:
    """Output from label tuning."""

    label_embeddings: torch.Tensor
    logit_scale: torch.Tensor
    training_time: float
    loss: float
    trained_param_count: int
    hyperparams: dict


Audio: TypeAlias = bytes | AudioDecoder
Video: TypeAlias = bytes | VideoDecoder
Image: TypeAlias = PILImage.Image
