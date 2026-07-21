"""Module for video pipelines."""

import torch

from delt.pipelines.base import Pipeline
from delt.types import Audio, Image, Video


class VideoPipeline(Pipeline):
    """Pipeline for video training and inference."""

    def get_input_embeddings(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute embeddings from video encoders.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): only `Video` for video encoders.
            batch_size (int): batch size to get embeddings from the encoder model.

        Returns:
            torch.Tensor: embeddings of shape (N, d).

        """
        assert all(isinstance(sample, Video) for sample in data), (
            "Only `Video` type is allowed for video encoders."
        )
        return self.encoder.get_video_embeddings(data, batch_size=batch_size)
