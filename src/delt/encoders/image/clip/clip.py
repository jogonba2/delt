"""Module for CLIP encoders."""

import torch
from tqdm import tqdm
from transformers import CLIPModel, PreTrainedModel

from delt.common import batchify
from delt.encoders.image.base import ImageEncoder
from delt.types import Image


class ClipEncoder(ImageEncoder):
    """Image encoder model that uses a pre-trained CLIP encoder."""

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
    def encoder(self) -> PreTrainedModel:
        """
        A pretrained CLIP encoder model.

        Returns:
            PreTrainedModel: the pretrained CLIP encoder.

        """
        if self._encoder is None:
            self._encoder = CLIPModel.from_pretrained(
                self.encoder_name,
            )
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._encoder.to(device)
            self._encoder.eval()
            self.freeze_params()
        return self._encoder

    def get_text_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """
        Embed a list of texts using the CLIP's text encoder model.

        Args:
            texts (list[str]): a list of texts.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The text embeddings.

        """
        output = []
        for batch in tqdm(
            batchify(texts, batch_size),
            desc="Encoding texts...",
            total=(len(texts) + batch_size - 1) // batch_size,
        ):
            features = self.processor(
                text=batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.encoder.device)

            with torch.no_grad():
                embeddings = self.encoder.get_text_features(**features)

            # Compat with Transformers 5
            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.pooler_output

            if self.normalize_embeddings:
                embeddings = embeddings / embeddings.norm(
                    p=2, dim=-1, keepdim=True
                )

            output.append(embeddings)
        return torch.vstack(output)

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
        output = []
        for batch in tqdm(
            batchify(images, batch_size),
            desc="Encoding images...",
            total=(len(images) + batch_size - 1) // batch_size,
        ):
            features = self.processor(batch, return_tensors="pt").to(
                self.encoder.device
            )

            with torch.no_grad():
                embeddings = self.encoder.get_image_features(**features)

            # Compat with Transformers 5
            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.pooler_output

            if self.normalize_embeddings:
                embeddings = embeddings / embeddings.norm(
                    p=2, dim=-1, keepdim=True
                )

            output.append(embeddings)
        return torch.vstack(output)
