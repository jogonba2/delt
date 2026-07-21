"""Module for text pipelines."""

import torch
from PIL import Image

from delt.pipelines.base import Pipeline


class TextPipeline(Pipeline):
    """Pipeline for text training and inference."""

    def get_input_embeddings(
        self,
        data: list[str] | list[Image.Image] | list[bytes],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute embeddings from text encoders.

        Args:
            data (list[str] | list[Image.Image] | list[bytes]): only `str` for text encoders.
            batch_size (int): batch size to get embeddings from the encoder model.

        Returns:
            torch.Tensor: embeddings of shape (N, d).

        """
        assert all(isinstance(sample, str) for sample in data), (
            "Only `str` type is allowed for text encoders."
        )
        return self.encoder.get_text_embeddings(data, batch_size=batch_size)
