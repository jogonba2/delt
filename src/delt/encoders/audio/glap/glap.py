"""Module for GLAP encoders."""

import torch
from glap_model import glap_inference
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from delt.common import batchify
from delt.encoders.audio.base import AudioEncoder
from delt.encoders.audio.utils import decode_audio
from delt.types import Audio


class GlapEncoder(AudioEncoder):
    """
    Audio encoder model that uses a pre-trained GLAP encoder.

    From https://github.com/xiaomi-research/dasheng-glap.
    """

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

        # In CPU until we find a way to move this properly to GPU w/o issues
        device_type = "cpu"
        self.device = torch.device(device_type)

    @property
    def processor(self) -> PreTrainedTokenizerBase | ProcessorMixin | None:
        """
        Processor tied to a GLAP model.

        This method is intentionally overriden for GLAP, since it is outside the HuggingFace
        ecosystem.

        Returns:
            PreTrainedTokenizerBase | ProcessorMixin | None: tokenizers for text
                models, processor mixin for image/audio/video, and `None` for
                models outside the HuggingFace ecosystem.

        """
        return None

    @property
    def encoder(self) -> PreTrainedModel:
        """
        A pretrained GLAP encoder model.

        Returns:
            PreTrainedModel: the pretrained GLAP encoder.

        """
        if self._encoder is None:
            self._encoder = glap_inference()
            self._encoder.to(self.device).eval()
            self._encoder.model_impl.text_encoder.to(self.device)
            self._encoder.eval()
            self.freeze_params()
        return self._encoder

    def get_text_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """
        Embed a list of texts using the GLAP's text encoder model.

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
            with torch.no_grad():
                embeddings = self.encoder.model_impl.text_encoder(
                    batch, source_lang="eng_Latn", device=self.device
                )
                embeddings = F.normalize(
                    self.encoder.model_impl.text_proj(embeddings), dim=-1
                )
            output.append(embeddings)

        return torch.vstack(output)

    def get_audio_embeddings(
        self, audios: list[Audio], batch_size: int = 16
    ) -> torch.Tensor:
        """
        Embed a list of audios using the GLAP audio encoder.

        Args:
            audios (list[Audio]): List of audios.
            batch_size (int): Batch size for inference.

        Returns:
            torch.Tensor: Audio embeddings.

        """
        output = []
        target_sampling_rate = 16000

        for batch in tqdm(
            batchify(audios, batch_size),
            desc="Encoding audios...",
            total=(len(audios) + batch_size - 1) // batch_size,
        ):
            # Still processed one-by-one because encode_audio doesn't support batching.
            for audio in batch:
                waveform = decode_audio(audio, target_sampling_rate)

                audio_tensor = (
                    torch.from_numpy(waveform)
                    .unsqueeze(0)  # (1, num_samples)
                    .to(self.device)
                )
                audio_length = torch.tensor(
                    [audio_tensor.shape[-1]], device=self.device
                )

                with torch.no_grad():
                    embeddings = self.encoder.encode_audio(
                        audio=audio_tensor,
                        audio_length=audio_length,
                    )

                output.append(embeddings)

        return torch.vstack(output)
