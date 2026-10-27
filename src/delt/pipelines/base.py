"""Module for the base pipeline."""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from delt.common import format_prompt, labels_to_indices
from delt.encoders import get_encoder
from delt.evaluate import evaluate as compute_metrics
from delt.predict import DEFAULT_LOGIT_SCALE, predict
from delt.train import label_tuning, label_tuning_cv
from delt.types import Audio, Image, Prediction, TrainingOutput, Video

INFO_FILENAME = "model.json"
EMBEDDINGS_FILENAME = "label_embeddings.safetensors"


def _read_model(
    path: str | Path, device: str
) -> tuple[dict[str, torch.Tensor], dict]:
    """Read the tensors and the information of a model saved with `EmbeddingPipeline.save`."""
    path = Path(path)
    info = json.loads((path / INFO_FILENAME).read_text(encoding="utf-8"))
    tensors = load_file(path / EMBEDDINGS_FILENAME, device=device)
    return tensors, info


class EmbeddingPipeline(ABC):
    """
    A base pipeline for training, and inference with text/image/audio/video modalities.

    Assumes `input_embeddings` are already precomputed.

    `label_embeddings` can be sent directly to the constructor if they are precomputed.
    Otherwise, an `EmbeddingPipeline` object can be instantiated with an encoder to build
    the `label_embeddings` for you.

    """

    # Label embeddings that are not named in `label_names`, e.g., the "none" class in span pipelines
    extra_label_embeddings = 0

    def __init__(
        self,
        label_embeddings: torch.Tensor | list,
        logit_scale: torch.Tensor | None = None,
        label_names: list[str] | None = None,
    ) -> None:
        """
        Initialize an embedding-based pipeline.

        Args:
            label_embeddings (torch.Tensor | list): the label embeddings with shape (N, d).
            logit_scale (torch.Tensor | None): the scale factor for the logits in log space with shape (1,), i.e., the logits are multiplied by `logit_scale.exp()`. Defaults to `log(1 / 0.07)`, the initial value used in label tuning, so zero-shot and tuned probabilities are comparable.
            label_names (list[str] | None): the names of the labels, in the same order as `label_embeddings`.
                                            If given, label names can be used in `truths` and are returned
                                            as the predicted labels.

        """
        self.label_embeddings = label_embeddings
        self.label_names = label_names
        self.logit_scale = (
            logit_scale
            if logit_scale is not None
            else torch.tensor([DEFAULT_LOGIT_SCALE])
        )
        self.training_output: TrainingOutput | None = None
        self.validation: dict | None = None

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
            truths (torch.Tensor | list): reference labels as integers (or label names, if `label_names` is set) with shape (N,) or expected probabilities with shape (N, classes).
            training_args (Optional[dict]): training args for label tuning in `delt.train.label_tuning`, e.g., `learning_rate`, `dropout`, and `drift_coefficient`.
            do_cv (bool): whether to do k-fold cross validation in training or not when training with label tuning.

        Returns:
            TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, cross-validation performance (if `do_cv`), and other outputs for inspecting training.

        """
        if training_args is None:
            training_args = {}

        truths = labels_to_indices(truths, self.label_names)

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
        self.training_output = output
        self.validation = output.validation
        return output

    def evaluate(
        self,
        input_embeddings: torch.Tensor | list,
        truths: torch.Tensor | list,
        batch_size: int = 16,
    ) -> dict:
        """
        Compute the performance of the pipeline on held-out data.

        The metrics are kept in the pipeline, so they are saved with `save`. They are replaced by the next call to `evaluate` or `fit`.

        Args:
            input_embeddings (torch.Tensor | list): the embedded data with shape (N, d).
            truths (torch.Tensor | list): reference labels as integers (or label names, if `label_names` is set) with shape (N,) or expected probabilities with shape (N, classes).
            batch_size (int): the batch size for inference.

        Returns:
            dict: number of samples, accuracy, macro and weighted F1, and precision, recall, F1 and support per label.

        """
        metrics = compute_metrics(
            input_embeddings,
            self.label_embeddings,
            self.logit_scale,
            truths,
            self.label_names,
            batch_size,
        )
        self.validation = {"method": "holdout", **metrics}
        return metrics

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
            list[Prediction]: list with predictions for each sample, containing the label with max prob (name if `label_names` is set, index otherwise) and probs per label.

        """
        return predict(
            input_embeddings,
            self.label_embeddings,
            self.logit_scale,
            batch_size,
            self.label_names,
        )

    def save(self, path: str | Path) -> Path:
        """
        Save the label embeddings, logit scale, label names, and training and validation information to a directory.

        The directory contains:
        - `label_embeddings.safetensors`: the `label_embeddings` (L, d) and `logit_scale` tensors, loadable with `safetensors.torch.load_file`.
        - `model.json`: the label names, information about the training, the performance on validation data (holdout or cross-validation, see `evaluate` and `fit`), and the encoder used to build the embeddings.

        Information about training is only available if the pipeline was trained with `fit`, and validation performance if it was computed with `evaluate` or `fit(do_cv=True)`. Both correspond to the last call.

        Args:
            path (str | Path): directory to save the model into. It is created if it does not exist.

        Returns:
            Path: the directory where the model was saved.

        """
        label_embeddings = torch.as_tensor(self.label_embeddings)
        num_labels = len(label_embeddings) - self.extra_label_embeddings
        if (
            self.label_names is not None
            and len(self.label_names) != num_labels
        ):
            msg = (
                f"Got {len(self.label_names)} label names for "
                f"{num_labels} label embeddings."
            )
            raise ValueError(msg)

        training = None
        if self.training_output is not None:
            training = {
                "hyperparams": self.training_output.hyperparams,
                "num_train_samples": self.training_output.num_train_samples,
                "training_time": self.training_output.training_time,
                "loss": self.training_output.loss,
                "trained_param_count": self.training_output.trained_param_count,
            }

        try:
            delt_version = version("deltpy")
        except PackageNotFoundError:
            delt_version = None

        info = {
            "delt_version": delt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label_names": self.label_names,
            "embedding_dim": label_embeddings.shape[-1],
            "training": training,
            "validation": self.validation,
            "encoder": self._encoder_info(),
        }

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "label_embeddings": label_embeddings.detach()
                .cpu()
                .contiguous(),
                "logit_scale": torch.as_tensor(self.logit_scale)
                .detach()
                .cpu()
                .contiguous(),
            },
            path / EMBEDDINGS_FILENAME,
            metadata={"format": "pt"},
        )
        (path / INFO_FILENAME).write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def _encoder_info(self) -> dict | None:
        return None

    @classmethod
    def load(
        cls, path: str | Path, device: str = "cpu"
    ) -> "EmbeddingPipeline":
        """
        Load a pipeline saved with `save`.

        Args:
            path (str | Path): directory where the model was saved.
            device (str): device to load the label embeddings into.

        Returns:
            EmbeddingPipeline: a pipeline with the saved label embeddings, logit scale, and label names.

        """
        tensors, info = _read_model(path, device)
        return cls(
            tensors["label_embeddings"],
            tensors["logit_scale"],
            info["label_names"],
        )

    @classmethod
    def load_from_prompts(
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
        return cls(label_embeddings, label_names=list(label_verbalizations))


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
        self.encoder_name = encoder_name
        self.encoder_class = encoder_class
        self.label_verbalizations = label_verbalizations
        self.prompt_template = prompt_template

        super().__init__(
            label_embeddings=encoder.get_text_embeddings(prompts),
            label_names=list(label_verbalizations),
        )

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
        truths: list[int] | list[str],
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
            truths (list[int] | list[str]): reference labels as label names or integers (positions of the labels in `label_verbalizations`).
            training_args (Optional[dict]): training args for label tuning in `delt.train.label_tuning`, e.g., `learning_rate`, `dropout`, and `drift_coefficient`.
            embeddings_batch_size (int): batch size to get embeddings from the encoder models.
            do_cv (bool): whether to do k-fold cross validation in training or not when training with label tuning.

        Returns:
            TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, cross-validation performance (if `do_cv`), and other outputs for inspecting training.

        """
        embeddings = self.get_input_embeddings(data, embeddings_batch_size)
        return super().fit(embeddings, truths, training_args, do_cv)

    def evaluate(
        self,
        data: list[str] | list[Image] | list[Audio] | list[Video],
        truths: list[int] | list[str],
        batch_size: int = 16,
        embeddings_batch_size: int = 16,
    ) -> dict:
        """
        Compute the performance of the pipeline on held-out data.

        The metrics are kept in the pipeline, so they are saved with `save`. They are replaced by the next call to `evaluate` or `fit`.

        Args:
            data (list[str] | list[Image] | list[Audio] | list[Video]): input data.
            truths (list[int] | list[str]): reference labels as label names or integers (positions of the labels in `label_verbalizations`).
            batch_size (int): the batch size for inference.
            embeddings_batch_size (int): batch size to get embeddings from the encoder models.

        Returns:
            dict: number of samples, accuracy, macro and weighted F1, and precision, recall, F1 and support per label.

        """
        embeddings = self.get_input_embeddings(data, embeddings_batch_size)
        return super().evaluate(embeddings, truths, batch_size)

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
            list[Prediction]: list with predictions for each sample, containing the name of the label with max prob and probs per label.

        """
        embeddings = self.get_input_embeddings(data, embeddings_batch_size)
        return super().predict(embeddings, batch_size)

    def _encoder_info(self) -> dict:
        return {
            "encoder_name": self.encoder_name,
            "encoder_class": self.encoder_class,
            "label_verbalizations": self.label_verbalizations,
            "prompt_template": self.prompt_template,
        }

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "EncoderPipeline":
        """
        Load a pipeline saved with `save`.

        The encoder is rebuilt from the saved encoder information, and the saved label embeddings and logit scale replace those computed from the prompts.

        Args:
            path (str | Path): directory where the model was saved.
            device (str): device to load the label embeddings into.

        Returns:
            EncoderPipeline: a pipeline with the saved label embeddings, logit scale, and label names.

        """
        tensors, info = _read_model(path, device)
        pipeline = cls(**info["encoder"])
        pipeline.label_embeddings = tensors["label_embeddings"]
        pipeline.logit_scale = tensors["logit_scale"]
        pipeline.label_names = info["label_names"]
        return pipeline
