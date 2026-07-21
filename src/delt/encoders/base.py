"""Encoder base model for zero-shot text classification and label-tuning."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from delt.common.logging import get_logger
from delt.types import Audio, Image, Video

logger = get_logger(__name__)


class Encoder(ABC):
    """Base encoder model."""

    def __init__(
        self,
        encoder_name: str,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize a encoder model.

        Args:
            encoder_name (str): name of sentence-transformer-compatible model.
            normalize_embeddings (bool): whether to apply L2 normalization to all embeddings.

        """
        self.encoder_name = encoder_name
        self.normalize_embeddings = normalize_embeddings
        self._processor = None
        self._encoder: PreTrainedModel = None

    @property
    @abstractmethod
    def processor(self) -> PreTrainedTokenizerBase | ProcessorMixin | None:
        """
        Processor tied to an encoder.

        Returns:
            PreTrainedTokenizerBase | ProcessorMixin | None: tokenizers for text
                models, processor mixin for image/audio/video, and `None` for
                models outside the HuggingFace ecosystem.

        """
        ...

    @property
    @abstractmethod
    def encoder(self) -> PreTrainedModel:
        """
        A pretrained encoder model.

        Returns:
            PreTrainedModel: the pretrained encoder.

        """
        ...

    @abstractmethod
    def get_text_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """
        Embed a list of texts using the base encoder model.

        Args:
            texts (list[str]): a list of texts.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The text embeddings.

        """
        ...

    @abstractmethod
    def get_image_embeddings(
        self, images: list[Image], batch_size: int = 8
    ) -> torch.Tensor:
        """
        Embed a list of images using the base encoder model.

        Args:
            images (list[Image): list of images.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The image embeddings.

        """
        ...

    @abstractmethod
    def get_audio_embeddings(
        self, audios: list[Audio], batch_size: int = 16
    ) -> torch.Tensor:
        """
        Embed a list of audios using the base encoder model.

        Args:
            audios (list[Audio]): list of audios.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The audio embeddings.

        """
        ...

    @abstractmethod
    def get_video_embeddings(
        self, videos: list[Video], batch_size: int = 16
    ) -> torch.Tensor:
        """
        Embed a list of videos using the base encoder model.

        Args:
            videos (list[Video]): list of videos.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The video embeddings.

        """
        ...

    def freeze_params(self, param_names: Optional[set[str]] = None) -> None:
        """
        Freeze the params of the encoder model.

        Args:
            param_names (Optional[set[str]]): names of the params to be frozen. If `None` all params will be frozen.

        """
        for name, param in self._encoder.named_parameters():
            if param_names is None or name in param_names:
                param.requires_grad = False

    def unfreeze_params(self, param_names: Optional[set[str]] = None) -> None:
        """
        Unfreeze the params of the encoder model.

        Args:
            param_names (Optional[set[str]]): names of the params to be unfrozen. If `None` all params will be unfrozen.

        """
        for name, param in self._encoder.named_parameters():
            if param_names is None or name in param_names:
                param.requires_grad = True
