"""Module for CLAP encoders."""

import torch
import transformers
from tqdm import tqdm
from transformers import ClapModel, PreTrainedModel

from delt.common import batchify
from delt.encoders.audio.base import AudioEncoder
from delt.encoders.audio.utils import decode_audio
from delt.types import Audio


class ClapEncoder(AudioEncoder):
    """Audio encoder model that uses a pre-trained Clap encoder."""

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
        A pretrained CLAP encoder model.

        Returns:
            PreTrainedModel: the pretrained CLAP encoder.

        """
        if self._encoder is None:
            self._encoder = ClapModel.from_pretrained(
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
        Embed a list of texts using the CLAP's text encoder model.

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
            # Text embeddings are always normalized:
            # https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/models/clap/modeling_clap.py#L1658
            with torch.no_grad():
                embeddings = self.encoder.get_text_features(**features)

            # Compat with Transformers 5
            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.pooler_output
            output.append(embeddings)

        return torch.vstack(output)

    def get_audio_embeddings(
        self,
        audios: list[Audio],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Embed a list of audios using the CLAP's audio encoder model.

        Args:
            audios (list[Audio]): list of audios.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The audio embeddings.

        """
        output = []
        target_sr = self.processor.feature_extractor.sampling_rate
        for batch in tqdm(
            batchify(audios, batch_size),
            desc="Encoding audios...",
            total=(len(audios) + batch_size - 1) // batch_size,
        ):
            samples = [
                decode_audio(audio, target_sampling_rate=target_sr)
                for audio in batch
            ]

            processor_kwargs = {
                "sampling_rate": target_sr,
                "return_tensors": "pt",
                "padding": True,
            }

            if int(transformers.__version__.split(".")[0]) >= 5:
                processor_kwargs["audio"] = samples
            else:
                processor_kwargs["audios"] = samples

            features = self.processor(**processor_kwargs)
            features = features.to(self.encoder.device)

            with torch.no_grad():
                embeddings = self.encoder.get_audio_features(**features)

            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.pooler_output

            output.append(embeddings)

        return torch.vstack(output)
