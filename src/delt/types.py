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
