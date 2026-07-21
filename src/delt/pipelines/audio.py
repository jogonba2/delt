"""Module for audio pipelines."""

import torch

from delt.pipelines.base import Pipeline
from delt.types import Audio, Image, Video


class AudioPipeline(Pipeline):
    """Pipeline for audio training and inference."""

    def get_input_embeddings(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute embeddings from audio encoders.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): only `Audio` for audio encoders.
            batch_size (int): batch size to get embeddings from the encoder model.

        Returns:
            torch.Tensor: embeddings of shape (N, d).

        """
        assert all(isinstance(sample, Audio) for sample in data), (
            "Only `Audio` type is allowed for audio encoders."
        )
        return self.encoder.get_audio_embeddings(data, batch_size=batch_size)
