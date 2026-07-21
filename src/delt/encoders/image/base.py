"""Module for the base image encoder."""

from abc import abstractmethod

import torch
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from delt.encoders.base import Encoder
from delt.types import Audio, Image, Video


class ImageEncoder(Encoder):
    """Base image encoder model."""

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
        super().__init__(
            encoder_name=encoder_name,
            normalize_embeddings=normalize_embeddings,
        )

    @property
    def processor(self) -> PreTrainedTokenizerBase | ProcessorMixin | None:
        """
        Processor tied to an image encoder.

        Returns:
            PreTrainedTokenizerBase | ProcessorMixin | None: tokenizers for text
                models, processor mixin for image/audio/video, and `None` for
                models outside the HuggingFace ecosystem.

        """
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.encoder_name)
        return self._processor

    @property
    @abstractmethod
    def encoder(self) -> PreTrainedModel:
        """
        A pretrained image encoder model.

        Returns:
            PreTrainedModel: the pretrained image encoder.

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
