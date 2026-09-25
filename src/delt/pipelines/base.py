"""Module for the base pipeline."""

from abc import ABC, abstractmethod

import torch

from delt.common import format_prompt
from delt.encoders import get_encoder
from delt.predict import predict
from delt.train import label_tuning, label_tuning_cv
from delt.types import Audio, Image, Prediction, TrainingOutput, Video


class EmbeddingPipeline(ABC):
    """
    A base pipeline for training, and inference with text/image/audio/video modalities.

    Assumes `input_embeddings` are already precomputed.

    `label_embeddings` can be sent directly to the constructor if they are precomputed.
    Otherwise, an `EmbeddingPipeline` object can be instantiated with an encoder to build
    the `label_embeddings` for you.

    """

    def __init__(
        self,
        label_embeddings: torch.Tensor | list,
        logit_scale: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize an embedding-based pipeline.

        Args:
            label_embeddings (torch.Tensor | list): the label embeddings with shape (N, d).
            logit_scale (torch.Tensor | None): the scale factor for the logits with shape (1,).

        """
        self.label_embeddings = label_embeddings
        self.logit_scale = (
            logit_scale if logit_scale is not None else torch.tensor([1.0])
        )

    def fit(
        self,
        input_embeddings: torch.Tensor | list,
        truths: torch.Tensor | list,
        training_args: dict | None = None,
        do_cv: bool = False,
    ) -> TrainingOutput:
        """
        Train the label embeddings and logit scale using label tuning.

        The tuned label embeddings and logit scale will be kept in the pipeline,
        so they are reused when doing inference with the same pipeline.

        Args:
            input_embeddings (torch.Tensor | list): the embedded data with shape (N, d).
            truths (torch.Tensor | list): reference labels as integers with shape (N,) or expected probabilities with shape (N, classes).
            training_args (Optional[dict]): training args for label tuning in `delt.train.label_tuning`, e.g., `learning_rate`, `dropout`, and `drift_coefficient`.
            do_cv (bool): whether to do k-fold cross validation in training or not when training with label tuning.

        Returns:
            TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, and other outputs for inspecting training.

        """
        if training_args is None:
            training_args = {}

        if do_cv:
            output = label_tuning_cv(
                input_embeddings,
                self.label_embeddings,
                truths,
            )
        else:
            output = label_tuning(
                input_embeddings,
                self.label_embeddings,
                truths,
                **training_args,
            )

        self.label_embeddings = output.label_embeddings
        self.logit_scale = output.logit_scale
        return output

    def predict(
        self,
        input_embeddings: torch.Tensor | list,
        batch_size: int = 16,
    ) -> list[Prediction]:
        """
        Compute the max prob label and probs per label for each input.

        Args:
            input_embeddings (torch.Tensor | list): the embedded data with shape (N, d).
            batch_size (int): the batch size for inference.

        Returns:
            list[Prediction]: list with predictions for each sample, containing the label with max prob and probs per label.

        """
        return predict(
            input_embeddings,
            self.label_embeddings,
            self.logit_scale,
            batch_size,
        )

    @classmethod
    def from_prompts(
        cls,
        encoder_name: str,
        encoder_class: str,
        label_verbalizations: dict[str, str],
        prompt_template: str,
    ) -> "EmbeddingPipeline":
        """
        Instantiate an `EmbeddingPipeline` object and initialize `label_embeddings` with a given prompt, verbalizations, and encoder.

        Useful when you have precomputed `input_embeddings` but you need to initialize `label_embeddings`.

        Args:
            encoder_name (str): pretrained name or path of the encoder.
            encoder_class (str): encoder class to be instantiated with `delt.encoders.get_encoder`.
            label_verbalizations (dict[str, str]): verbalizations of the labels, e.g. {"positive": "very cool!", "negative": "horrible"}
            prompt_template (str): template to format label verbalizations, e.g., "This text is {}" being instantiated as "This text is very cool!".

        Returns:
            EmbeddingPipeline: an `EmbeddingPipeline` object with initialized `label_embeddings`.

        """
        encoder = get_encoder(encoder_class, encoder_name)
        prompts = format_prompt(label_verbalizations, prompt_template)
        label_embeddings = encoder.get_text_embeddings(prompts)
        return cls(label_embeddings)


class EncoderPipeline(EmbeddingPipeline):
    """
    A base pipeline for encoding, training, and inference with text/image/audio/video modalities.

    Does not assume `input_embeddings` nor `label_embeddings` are precomputed,
    so they are computed by using the encoder provided at instantiation time.
    """

    def __init__(
        self,
        encoder_name: str,
        encoder_class: str,
        label_verbalizations: dict[str, str],
        prompt_template: str,
    ) -> None:
        """
        Initialize an encoder-based pipeline.

        Args:
            encoder_name (str): pretrained name or path of the encoder.
            encoder_class (str): encoder class to be instantiated with `delt.encoders.get_encoder`.
            label_verbalizations (dict[str, str]): verbalizations of the labels, e.g. {"positive": "very cool!", "negative": "horrible"}
            prompt_template (str): template to format label verbalizations, e.g., "This text is {}" being instantiated as "This text is very cool!".

        """
        prompts = format_prompt(label_verbalizations, prompt_template)
        encoder = get_encoder(encoder_class, encoder_name)

        self.encoder = encoder

        super().__init__(label_embeddings=encoder.get_text_embeddings(prompts))

    @abstractmethod
    def get_input_embeddings(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute embeddings from an encoder model.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): input data.
            batch_size (int): batch size to get embeddings from the encoder model.

        Returns:
            torch.Tensor: embeddings of shape (N, d).

        """
        ...

    def fit(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        truths: list[int],
        training_args: dict | None = None,
        embeddings_batch_size: int = 16,
        do_cv: bool = False,
    ) -> TrainingOutput:
        """
        Train the label embeddings and logit scale using label tuning.

        The tuned label embeddings and logit scale will be kept in the pipeline,
        so they are reused when doing inference with the same pipeline.

        Args:
            data (list[str] | list[Image.Image] | list[bytes]): input data.
            truths (list[int]): reference labels as integers.
            training_args (Optional[dict]): training args for label tuning in `delt.train.label_tuning`, e.g., `learning_rate`, `dropout`, and `drift_coefficient`.
            embeddings_batch_size (int): batch size to get embeddings from the encoder models.
            do_cv (bool): whether to do k-fold cross validation in training or not when training with label tuning.

        Returns:
            TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, and other outputs for inspecting training.

        """
        embeddings = self.get_input_embeddings(data, embeddings_batch_size)
        return super().fit(embeddings, truths, training_args, do_cv)

    def predict(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        batch_size: int = 16,
        embeddings_batch_size: int = 16,
    ) -> list[Prediction]:
        """
        Compute the max prob label and probs per label for each input.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): input data.
            batch_size (int): the batch size for inference.
            embeddings_batch_size (int): batch size to get embeddings from the encoder models.

        Returns:
            list[Prediction]: list with predictions for each sample, containing the label with max prob and probs per label.

        """
        embeddings = self.get_input_embeddings(data, embeddings_batch_size)
        return super().predict(embeddings, batch_size)
