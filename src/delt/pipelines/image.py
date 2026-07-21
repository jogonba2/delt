"""Pipeline for image pipelines."""

import torch

from delt.pipelines.base import Pipeline
from delt.types import Audio, Image, Video


class ImagePipeline(Pipeline):
    """Pipeline for image training and inference."""

    def get_input_embeddings(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute embeddings from image encoders.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): only `Image` for image encoders.
            batch_size (int): batch size to get embeddings from the encoder model.

        Returns:
            torch.Tensor: embeddings of shape (N, d).

        """
        assert all(isinstance(sample, Image) for sample in data), (
            "Only `PIL.Image.Image` type is allowed for image encoders."
        )
        return self.encoder.get_image_embeddings(data, batch_size=batch_size)
