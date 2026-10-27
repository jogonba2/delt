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

    def get_token_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> list[tuple[torch.Tensor, list[tuple[int, int]]]]:
        """
        Embed the tokens of each text in context using the sentence-transformers' encoder model.

        Args:
            texts (list[str]): a list of texts.
            batch_size (int): batch size for inference.

        Returns:
            list[tuple[torch.Tensor, list[tuple[int, int]]]]: for each text, the token embeddings with shape (T, d) and the (start, end) character offsets of each token, with `end` exclusive. Special tokens are dropped and texts longer than the encoder's max length are truncated.

        """
        token_embeddings = self.encoder.encode(
            texts,
            output_value="token_embeddings",
            batch_size=batch_size,
            show_progress_bar=True,
        )
        outputs = []
        for text, embeddings in zip(texts, token_embeddings, strict=True):
            offsets = self.encoder.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.encoder.max_seq_length,
            )["offset_mapping"]
            assert len(offsets) == len(embeddings), (
                "Token offsets do not match the token embeddings."
            )
            # Special tokens have empty offsets
            keep = [i for i, (start, end) in enumerate(offsets) if start < end]
            outputs.append(
                (
                    embeddings[keep].detach().float().cpu(),
                    [offsets[i] for i in keep],
                )
            )
        return outputs
