"""Module for span pipelines."""

import random
from typing import Literal

import torch

from delt.common.spans import enumerate_candidates, select_spans
from delt.pipelines.base import EmbeddingPipeline
from delt.pipelines.text import TextPipeline
from delt.types import Span, SpanAnnotation, TrainingOutput


class SpanPipeline(TextPipeline):
    """
    Pipeline for span detection and classification with label tuning.

    Instead of tagging tokens (e.g., BIO), every contiguous span of up to `max_span_width` tokens
    is a candidate that is embedded from its text and classified against the label embeddings.
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
        negative_ratio: float = 5.0,
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
            max_span_width (int): maximum number of tokens in a candidate span.
            negative_ratio (float): number of "none" candidates sampled per gold span when training.
            seed (int): seed to sample negative candidates.

        """
        self.label_names = list(label_verbalizations)
        self.none_index = len(self.label_names)
        self.max_span_width = max_span_width
        self.negative_ratio = negative_ratio
        self.seed = seed
        super().__init__(
            encoder_name,
            encoder_class,
            {**label_verbalizations, "__none__": none_verbalization},
            prompt_template,
        )

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

    def _embed_spans(
        self, strings: list[str], batch_size: int
    ) -> torch.Tensor:
        """Embed span strings, encoding identical strings only once."""
        unique = list(dict.fromkeys(strings))
        embeddings = self.get_input_embeddings(unique, batch_size)
        position = {string: i for i, string in enumerate(unique)}
        return embeddings[[position[string] for string in strings]]

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

        Gold spans are positives and a random subset of the remaining candidates are "none" negatives.

        Args:
            data (list[str]): input texts.
            truths (list[list[SpanAnnotation]]): gold spans of each text. Offsets must align with token boundaries.
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

        strings: list[str] = []
        labels: list[int] = []
        negatives: list[str] = []
        for text, annotations in zip(data, truths, strict=True):
            candidates = set(enumerate_candidates(text, self.max_span_width))
            all_candidates = set(
                enumerate_candidates(
                    text, self.max_span_width, filter_edges=False
                )
            )
            gold = set()
            for annotation in annotations:
                offsets = (annotation.start, annotation.end)
                if offsets not in all_candidates:
                    msg = (
                        f"Span {text[annotation.start : annotation.end]!r} {offsets} in {text!r} "
                        f"does not align with token boundaries or is wider than {self.max_span_width} tokens."
                    )
                    raise ValueError(msg)
                gold.add(offsets)
                strings.append(text[annotation.start : annotation.end])
                labels.append(self._label_index(annotation.label))
            negatives.extend(
                text[start:end] for start, end in sorted(candidates - gold)
            )

        num_negatives = min(
            len(negatives), round(self.negative_ratio * max(len(strings), 1))
        )
        strings.extend(
            random.Random(self.seed).sample(negatives, num_negatives)
        )
        labels.extend([self.none_index] * num_negatives)

        embeddings = self._embed_spans(strings, embeddings_batch_size)
        return EmbeddingPipeline.fit(self, embeddings, labels, training_args)

    def predict(
        self,
        data: list[str],
        batch_size: int = 16,
        embeddings_batch_size: int = 16,
        threshold: float = 0.5,
        overlap: Literal["flat", "nested"] = "flat",
    ) -> list[list[Span]]:
        """
        Detect and classify spans in each text.

        A candidate is scored as `1 - p(none)` and gets the highest-probability non-none label.

        Args:
            data (list[str]): input texts.
            batch_size (int): the batch size for inference.
            embeddings_batch_size (int): batch size to get embeddings from the encoder model.
            threshold (float): minimum score to keep a span.
            overlap (Literal["flat", "nested"]): "flat" keeps non-overlapping spans only (highest score first); "nested" keeps all spans above the threshold.

        Returns:
            list[list[Span]]: spans of each text. `Span.probs` has the probability of each label followed by the "none" class.

        """
        all_candidates = [
            enumerate_candidates(text, self.max_span_width) for text in data
        ]
        strings = [
            text[start:end]
            for text, candidates in zip(data, all_candidates, strict=True)
            for start, end in candidates
        ]
        if not strings:
            return [[] for _ in data]

        embeddings = self._embed_spans(strings, embeddings_batch_size)
        predictions = EmbeddingPipeline.predict(self, embeddings, batch_size)

        results: list[list[Span]] = []
        offset = 0
        for text, candidates in zip(data, all_candidates, strict=True):
            probs = [
                [float(x) for x in p.probs]
                for p in predictions[offset : offset + len(candidates)]
            ]
            offset += len(candidates)
            scores = [1.0 - p[self.none_index] for p in probs]
            keep = select_spans(candidates, scores, threshold, overlap)
            results.append(
                [
                    Span(
                        start=candidates[i][0],
                        end=candidates[i][1],
                        text=text[candidates[i][0] : candidates[i][1]],
                        label=max(
                            range(self.none_index), key=lambda k: probs[i][k]
                        ),
                        score=scores[i],
                        probs=probs[i],
                    )
                    for i in keep
                ]
            )
        return results
