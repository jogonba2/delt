"""Module for XCLIP encoders."""

import types
from typing import Any

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, XCLIPModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from delt.common import batchify
from delt.encoders.video.base import VideoEncoder
from delt.encoders.video.utils import extract_frames
from delt.types import Video


def _monkey_patch_get_video_features(
    self: PreTrainedModel, pixel_values: torch.Tensor, **kwargs: Any
) -> tuple | BaseModelOutputWithPooling:
    batch_size, num_frames, num_channels, height, width = pixel_values.shape
    pixel_values = pixel_values.reshape(-1, num_channels, height, width)

    video_outputs = self.vision_model(pixel_values=pixel_values, **kwargs)
    video_embeds = video_outputs.pooler_output
    video_embeds = self.visual_projection(video_embeds)

    cls_features = video_embeds.view(batch_size, num_frames, -1)
    mit_outputs = self.mit(cls_features, **kwargs)
    if isinstance(mit_outputs, BaseModelOutputWithPooling):
        video_outputs.pooler_output = mit_outputs.pooler_output
    else:
        video_outputs.pooler_output = mit_outputs[1]
    return video_outputs.pooler_output


class XclipEncoder(VideoEncoder):
    """Video encoder model that uses a pre-trained XCLIP encoder."""

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
        A pretrained XCLIP encoder model.

        Returns:
            PreTrainedModel: the pretrained XCLIP encoder.

        """
        if self._encoder is None:
            self._encoder = XCLIPModel.from_pretrained(self.encoder_name)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._encoder.to(device)
            self._encoder.eval()
            self.freeze_params()

            # Monkey patching for `get_video_features` issues
            # with output type in HuggingFace Transformers
            self._encoder.get_video_features = types.MethodType(
                _monkey_patch_get_video_features, self._encoder
            )
        return self._encoder

    def get_text_embeddings(
        self, texts: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """
        Embed a list of texts using the XCLIP's text encoder model.

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

    def get_video_embeddings(
        self, videos: list[Video], batch_size: int = 8
    ) -> torch.Tensor:
        """
        Embed a list of videos using the XCLIP's video encoder model.

        Args:
            videos (list[Video]): list of videos.
            batch_size (int): batch size for inference.

        Returns:
            torch.Tensor: The video embeddings.

        """
        output = []
        expected_frames = getattr(
            self.encoder.config.vision_config, "num_frames", 32
        )
        for batch in tqdm(
            batchify(videos, batch_size),
            desc="Encoding videos...",
            total=(len(videos) + batch_size - 1) // batch_size,
        ):
            frames = [
                extract_frames(video, clip_len=expected_frames)
                for video in batch
            ]
            features = self.processor(frames, return_tensors="pt").to(
                self.encoder.device
            )

            with torch.no_grad():
                embeddings = self.encoder.get_video_features(**features)

            # Compat with Transformers 5
            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.pooler_output

            if self.normalize_embeddings:
                embeddings = embeddings / embeddings.norm(
                    p=2, dim=-1, keepdim=True
                )

            output.append(embeddings)
        return torch.vstack(output)
