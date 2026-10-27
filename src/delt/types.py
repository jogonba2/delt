"""Types and defaults used in modeling."""

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch
from datasets.features._torchcodec import AudioDecoder
from PIL import Image as PILImage
from torchcodec.decoders._video_decoder import VideoDecoder

DEFAULT_LOGIT_SCALE = math.log(1 / 0.07)
MAX_LOGIT_SCALE = math.log(100)


@dataclass
class Prediction:
    """A sample prediction."""

    probs: list[float]
    label: int | str


@dataclass
class TrainingOutput:
    """Output from label tuning."""

    label_embeddings: torch.Tensor
    logit_scale: torch.Tensor
    training_time: float
    loss: float
    trained_param_count: int
    hyperparams: dict
    num_train_samples: int | None = None
    validation: dict | None = None


Audio: TypeAlias = bytes | AudioDecoder
Video: TypeAlias = bytes | VideoDecoder
Image: TypeAlias = PILImage.Image
