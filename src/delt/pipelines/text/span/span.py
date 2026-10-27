"""Module for span pipelines."""

import random
from pathlib import Path

import torch

from delt.pipelines.base import EmbeddingPipeline
from delt.pipelines.text.span.types import PredictedSpan, SpanAnnotation
from delt.pipelines.text.span.utils import (
    enumerate_candidates,
    mean_pool_spans,
    select_spans,
)
from delt.pipelines.text.text import TextPipeline
from delt.predict import predict_proba
from delt.types import TrainingOutput


class SpanPipeline(TextPipeline):
    """
    Pipeline for span detection and classification with label tuning.

    Instead of tagging tokens (e.g., BIO), every contiguous span of up to `max_span_width` words
    is a candidate that is classified against the label embeddings. Each text is encoded once and
    each candidate is embedded as the mean of its contextual token embeddings.
    An extra, trainable "none" class represents candidates that are not spans of interest.
    Labels are plain concepts, so label verbalizations are not polluted with tagging prefixes.
    """

    def __init__(
        self,
        encoder_name: str,
        encoder_class: str,
        label_verbalizations: dict[str, str],
        prompt_template: str,
        none_verbalization: str = "not a relevant phrase",
        max_span_width: int = 8,
        negative_ratio: float | None = None,
        seed: int = 13,
    ) -> None:
        """
        Initialize a span pipeline.

        Args:
            encoder_name (str): pretrained name or path of the encoder.
            encoder_class (str): encoder class to be instantiated with `delt.encoders.get_encoder`.
            label_verbalizations (dict[str, str]): verbalizations of the labels, e.g. {"person": "a person name"}. Do not include the "none" class.
            prompt_template (str): template to format label verbalizations, e.g., "This text is {}".
            none_verbalization (str): verbalization of the "none" class, which is appended after the labels.
            max_span_width (int): maximum number of words in a candidate span.
            negative_ratio (float | None): number of "none" candidates sampled per gold span when training. If `None`, all the candidates that are not gold spans are used.
            seed (int): seed to sample negative candidates.

        """
        self.none_index = len(list(label_verbalizations))
        self.max_span_width = max_span_width
        self.negative_ratio = negative_ratio
        self.seed = seed
        super().__init__(
            encoder_name,
            encoder_class,
            {**label_verbalizations, "__none__": none_verbalization},
            prompt_template,
        )
        self.label_names = list(label_verbalizations)

    def save(self, path: str | Path) -> Path:
        """Not supported yet, since the "none" class and the span settings are not saved."""
        msg = "Saving is not supported yet for `SpanPipeline`."
        raise NotImplementedError(msg)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SpanPipeline":
        """Not supported yet, since the "none" class and the span settings are not saved."""
        msg = "Loading is not supported yet for `SpanPipeline`."
        raise NotImplementedError(msg)

    def evaluate(self, *args: object, **kwargs: object) -> dict:
        """Not supported yet, since metrics would need to be computed on spans."""
        msg = "Evaluation is not supported yet for `SpanPipeline`."
        raise NotImplementedError(msg)

    def _label_index(self, label: int | str) -> int:
        if isinstance(label, str):
            if label not in self.label_names:
                msg = f"Unknown label {label!r}, expected one of {self.label_names}."
                raise ValueError(msg)
            return self.label_names.index(label)
        if not 0 <= label < self.none_index:
            msg = f"Label index {label} out of range for {self.none_index} labels."
            raise ValueError(msg)
        return label

    @staticmethod
    def _encoded_length(token_offsets: list[tuple[int, int]]) -> int:
        """Get the number of characters covered by the encoder, which truncates long texts."""
        return token_offsets[-1][1] if token_offsets else 0

    def fit(
        self,
        data: list[str],
        truths: list[list[SpanAnnotation]],
        training_args: dict | None = None,
        embeddings_batch_size: int = 16,
        do_cv: bool = False,
    ) -> TrainingOutput:
        """
        Train the label embeddings and logit scale using label tuning on candidate spans.

        Gold spans are positives and the remaining candidates (all of them, or a random subset given by `negative_ratio`) are "none" negatives.

        Args:
            data (list[str]): input texts.
            truths (list[list[SpanAnnotation]]): gold spans of each text. Offsets must align with word boundaries.
            training_args (Optional[dict]): training args for label tuning in `delt.train.label_tuning`.
            embeddings_batch_size (int): batch size to get embeddings from the encoder model.
            do_cv (bool): not supported yet, since folds would be split by span instead of by text.

        Returns:
            TrainingOutput: containing the tuned label embeddings (N, d), logit scale (1,), training time, and other outputs for inspecting training.

        """
        if do_cv:
            msg = "Cross-validation is not supported yet for `SpanPipeline`."
            raise NotImplementedError(msg)
        if len(data) != len(truths):
            msg = "`data` and `truths` must have the same length."
            raise ValueError(msg)

        encoded = self.encoder.get_token_embeddings(
            data, embeddings_batch_size
        )

        # (text index, offsets, label) of each training span
        positives: list[tuple[int, tuple[int, int], int]] = []
        negatives: list[tuple[int, tuple[int, int], int]] = []
        for index, (text, annotations, (_, token_offsets)) in enumerate(
            zip(data, truths, encoded, strict=True)
        ):
            candidates = enumerate_candidates(text, self.max_span_width)
            candidate_set = set(candidates)
            length = self._encoded_length(token_offsets)
            gold = set()
            for annotation in annotations:
                offsets = (annotation.start, annotation.end)
                if offsets not in candidate_set:
                    msg = (
                        f"Span {text[annotation.start : annotation.end]!r} {offsets} in {text!r} "
                        f"does not align with word boundaries or is wider than {self.max_span_width} words."
                    )
                    raise ValueError(msg)
                if annotation.end > length:
                    msg = (
                        f"Span {text[annotation.start : annotation.end]!r} {offsets} in {text!r} "
                        "is beyond the encoder's max length."
                    )
                    raise ValueError(msg)
                gold.add(offsets)
                positives.append(
                    (index, offsets, self._label_index(annotation.label))
                )
            negatives.extend(
                (index, offsets, self.none_index)
                for offsets in candidates
                if offsets not in gold and offsets[1] <= length
            )

        if self.negative_ratio is not None:
            num_negatives = min(
                len(negatives),
                round(self.negative_ratio * max(len(positives), 1)),
            )
            negatives = random.Random(self.seed).sample(
                negatives, num_negatives
            )

        # Group spans by text to pool each text's token embeddings at once
        samples = sorted(positives + negatives, key=lambda sample: sample[0])
        spans_per_text: list[list[tuple[int, int]]] = [[] for _ in data]
        for index, offsets, _ in samples:
            spans_per_text[index].append(offsets)
        labels = [label for _, _, label in samples]

        embeddings = torch.cat(
            [
                mean_pool_spans(token_embeddings, token_offsets, spans)
                for (token_embeddings, token_offsets), spans in zip(
                    encoded, spans_per_text, strict=True
                )
            ]
        )
        return EmbeddingPipeline.fit(self, embeddings, labels, training_args)

    def predict(
        self,
        data: list[str],
        batch_size: int = 16,
        embeddings_batch_size: int = 16,
        threshold: float = 0.5,
    ) -> list[list[PredictedSpan]]:
        """
        Detect and classify non-overlapping spans in each text.

        A candidate is scored as `1 - p(none)` and gets the highest-probability non-none label.
        Candidates above the threshold are kept from the highest score down, skipping those that overlap a kept one.
        Candidates beyond the encoder's max length are not considered.

        Args:
            data (list[str]): input texts.
            batch_size (int): the batch size for inference.
            embeddings_batch_size (int): batch size to get embeddings from the encoder model.
            threshold (float): minimum score to keep a span.

        Returns:
            list[list[PredictedSpan]]: spans of each text. `PredictedSpan.probs` has the probability of each label followed by the "none" class.

        """
        encoded = self.encoder.get_token_embeddings(
            data, embeddings_batch_size
        )
        all_candidates = [
            [
                offsets
                for offsets in enumerate_candidates(text, self.max_span_width)
                if offsets[1] <= self._encoded_length(token_offsets)
            ]
            for text, (_, token_offsets) in zip(data, encoded, strict=True)
        ]
        if not any(all_candidates):
            return [[] for _ in data]

        embeddings = torch.cat(
            [
                mean_pool_spans(token_embeddings, token_offsets, candidates)
                for (token_embeddings, token_offsets), candidates in zip(
                    encoded, all_candidates, strict=True
                )
            ]
        )
        all_probs = predict_proba(
            embeddings, self.label_embeddings, self.logit_scale, batch_size
        )

        results: list[list[PredictedSpan]] = []
        offset = 0
        for text, candidates in zip(data, all_candidates, strict=True):
            probs = [
                [float(prob) for prob in probs]
                for probs in all_probs[offset : offset + len(candidates)]
            ]
            offset += len(candidates)
            scores = [1.0 - p[self.none_index] for p in probs]
            keep = select_spans(candidates, scores, threshold)
            results.append(
                [
                    PredictedSpan(
                        start=candidates[i][0],
                        end=candidates[i][1],
                        text=text[candidates[i][0] : candidates[i][1]],
                        label=self.label_names[
                            max(
                                range(self.none_index),
                                key=lambda k: probs[i][k],
                            )
                        ],
                        score=scores[i],
                        probs=probs[i],
                    )
                    for i in keep
                ]
            )
        return results
