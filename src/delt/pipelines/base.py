"""Module for the base pipeline."""

from abc import ABC, abstractmethod

import torch

from delt.common import format_prompt
from delt.encoders import get_encoder
from delt.predict import predict
from delt.train import label_tuning, label_tuning_cv
from delt.types import Audio, Image, Prediction, TrainingOutput, Video


class Pipeline(ABC):
    """A base pipeline for computing embeddings, training, and inference with text/image/audio/video modalities."""

    def __init__(
        self,
        encoder_name: str,
        encoder_class: str,
        label_verbalizations: dict[str, str],
        prompt_template: str,
    ) -> None:
        """
        Initialize a pipeline.

        Args:
            encoder_name (str): pretrained name or path of the encoder.
            encoder_class (str): encoder class to be instantiated with `delt.encoders.get_encoder`.
            label_verbalizations (dict[str, str]): verbalizations of the labels, e.g. {"positive": "very cool!", "negative": "horrible"}
            prompt_template (str): template to format label verbalizations, e.g., "This text is {}" being instantiated as "This text is very cool!".

        """
        self.encoder_name = encoder_name
        self.encoder_class = encoder_class
        self.label_verbalizations = label_verbalizations
        self.prompt_template = prompt_template

        self.prompts = format_prompt(label_verbalizations, prompt_template)
        self.encoder = get_encoder(encoder_class, encoder_name=encoder_name)
        self.label_embeddings = self.encoder.get_text_embeddings(self.prompts)
        self.logit_scale = torch.Tensor([1.0])

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
        training_args: dict = {},
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
            training_args (dict): training args for label tuning in `delt.train.label_tuning`, e.g., `learning_rate`, `dropout`, and `drift_coefficient`.
            embeddings_batch_size (int): batch size to get embeddings from the encoder models.
            do_cv (bool): whether to do k-fold cross validation in training or not when training with label tuning.

        Returns:
            TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, and other outputs for inspecting training.

        """
        input_embeddings = self.get_input_embeddings(
            data, embeddings_batch_size
        )

        if do_cv:
            output = label_tuning_cv(
                input_embeddings, self.label_embeddings, truths
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
        data: list[str] | list[Image] | list[Audio] | list[Video],
        batch_size: int = 16,
        embeddings_batch_size: int = 16,
    ) -> list[Prediction]:
        """
        Compute the max prob label and probs per label for each input.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): input data.
            batch_size (int): the batch size.
            embeddings_batch_size (int): batch size to get embeddings from the encoder models.

        Returns:
            list[Prediction]: list with predictions for each sample, containing the label with max prob and probs per label.

        """
        input_embeddings = self.get_input_embeddings(
            data, embeddings_batch_size
        )
        return predict(
            input_embeddings,
            self.label_embeddings,
            self.logit_scale,
            batch_size,
        )
