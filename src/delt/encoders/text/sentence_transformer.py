"""Module for sentence-transformers encoders."""

import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel

from delt.common.logging import get_logger
from delt.encoders.text.base import TextEncoder

logger = get_logger(__name__)


class SentenceTransformerEncoder(TextEncoder):
    """Text encoder model that uses a pre-trained encoder from sentence-transformers."""

    def __init__(
        self,
        encoder_name: str,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize a SentenceTransformerEncoder model.

        Args:
            encoder_name (str): name of sentence-transformer-compatible model.
            normalize_embeddings (bool): whether to apply L2 normalization to all embeddings.

        """
        # Init encoder
        super().__init__(
            encoder_name=encoder_name,
            normalize_embeddings=normalize_embeddings,
        )

    @property
    def encoder(self) -> PreTrainedModel:
        """
        A pretrained text encoder model from sentence-transformers.

        Returns:
            PreTrainedModel: the pretrained text encoder.

        """
        if self._encoder is None:
            self._encoder = SentenceTransformer(
                self.encoder_name, trust_remote_code=True
            )
            self.freeze_params()
        return self._encoder

    def get_text_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """
        Embed a list of texts using the sentence-transformers' encoder model.

        Args:
            texts (list[str]): a list of texts.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The text embeddings.

        """
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        return torch.from_numpy(embeddings)
