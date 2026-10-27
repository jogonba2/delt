"""Tests for `delt.SpanPipeline`."""

import json
import random
from itertools import pairwise

import pytest
import torch

from delt import SpanPipeline
from delt.pipelines.text import SpanAnnotation


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


def _annotate(text: str, phrase: str, label: str) -> SpanAnnotation:
    start = text.index(phrase)
    return SpanAnnotation(start, start + len(phrase), label)


@pytest.fixture
def span_dataset():
    """Build a tiny NER-like dataset."""
    samples = [
        (
            "Alice visited Paris last summer.",
            [("Alice", "person"), ("Paris", "location")],
        ),
        (
            "Bob moved to Berlin in March.",
            [("Bob", "person"), ("Berlin", "location")],
        ),
        (
            "Carol flew from Madrid to Rome.",
            [
                ("Carol", "person"),
                ("Madrid", "location"),
                ("Rome", "location"),
            ],
        ),
        (
            "David lives in London with Eve.",
            [("David", "person"), ("London", "location"), ("Eve", "person")],
        ),
    ]
    texts = [text for text, _ in samples]
    truths = [
        [_annotate(text, phrase, label) for phrase, label in spans]
        for text, spans in samples
    ]
    return texts, truths


class TestSpanPipeline:
    """Class to test `SpanPipeline`."""

    @classmethod
    def setup_class(cls):
        """Create a `TestSpanPipeline` instance."""
        cls.pipeline = SpanPipeline(
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformer",
            {"person": "a person name", "location": "a place name"},
            "This text is {}",
            max_span_width=3,
        )

    @staticmethod
    def _f1(preds, truths):
        pred_set = {
            (i, s.start, s.end, s.label)
            for i, spans in enumerate(preds)
            for s in spans
        }
        gold_set = {
            (i, a.start, a.end, a.label)
            for i, spans in enumerate(truths)
            for a in spans
        }
        tp = len(pred_set & gold_set)
        if tp == 0:
            return 0.0
        precision, recall = tp / len(pred_set), tp / len(gold_set)
        return 2 * precision * recall / (precision + recall)

    def test_label_tuning_improves_zero_shot(self, span_dataset):
        """Check whether label tuning improves or matches zero-shot span F1."""
        texts, truths = span_dataset
        zs_f1 = self._f1(self.pipeline.predict(texts), truths)
        self.pipeline.fit(texts, truths)
        preds = self.pipeline.predict(texts)
        assert self._f1(preds, truths) >= zs_f1
        assert all(len(s.probs) == 3 for spans in preds for s in spans)

    def test_negative_sampling(self, span_dataset):
        """Training with sampled negatives still predicts the label probs."""
        texts, truths = span_dataset
        pipeline = SpanPipeline(
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformer",
            {"person": "a person name", "location": "a place name"},
            "This text is {}",
            max_span_width=3,
            negative_ratio=2.0,
        )
        pipeline.fit(texts, truths, training_args={"epochs": 10})
        preds = pipeline.predict(texts, threshold=0.0)
        assert all(len(s.probs) == 3 for spans in preds for s in spans)

    def test_predicted_spans_do_not_overlap(self, span_dataset):
        """Predicted spans of each text never overlap."""
        texts, _ = span_dataset
        for spans in self.pipeline.predict(texts, threshold=0.0):
            for previous, current in pairwise(spans):
                assert previous.end <= current.start

    def test_empty_text(self):
        """Texts without candidates get no spans."""
        assert self.pipeline.predict([""]) == [[]]
        preds = self.pipeline.predict(["", "Alice left."], threshold=0.0)
        assert preds[0] == []
        assert preds[1]

    def test_misaligned_span_raises(self):
        """Gold spans must align with token boundaries."""
        with pytest.raises(ValueError):
            self.pipeline.fit(
                ["Alice left."], [[SpanAnnotation(1, 4, "person")]]
            )

    def test_cv_not_supported(self, span_dataset):
        """Cross-validation is rejected."""
        texts, truths = span_dataset
        with pytest.raises(NotImplementedError):
            self.pipeline.fit(texts, truths, do_cv=True)

    def test_evaluate_on_training_data(self, span_dataset):
        """Check that `evaluate` returns span metrics and keeps them."""
        texts, truths = span_dataset
        pipeline = self._new_pipeline()
        pipeline.fit(texts, truths, training_args={"epochs": 200})
        assert pipeline.validation is None

        metrics = pipeline.evaluate(texts, truths, threshold=0.5)
        assert metrics["num_samples"] == len(texts)
        assert metrics["micro"]["f1"] > 0.5
        assert set(metrics["per_label"]) == {"person", "location"}
        assert metrics["per_label"]["person"]["support"] == 5
        assert pipeline.validation == {"method": "holdout", **metrics}

    def test_evaluate_length_mismatch(self, span_dataset):
        """Check that texts and gold spans must be aligned."""
        texts, truths = span_dataset
        with pytest.raises(ValueError, match="same length"):
            self.pipeline.evaluate(texts, truths[:-1])

    def test_save_load_roundtrip(self, span_dataset, tmp_path):
        """Check that a loaded span pipeline predicts like the saved one."""
        texts, truths = span_dataset
        pipeline = self._new_pipeline()
        pipeline.fit(texts, truths, training_args={"epochs": 50})
        pipeline.evaluate(texts, truths)
        pipeline.save(tmp_path)

        info = json.loads((tmp_path / "model.json").read_text())
        assert info["label_names"] == ["person", "location"]
        assert info["validation"]["micro"]
        assert info["encoder"]["label_verbalizations"] == {
            "person": "a person name",
            "location": "a place name",
        }
        assert info["encoder"]["none_verbalization"] == (
            "not a relevant phrase"
        )
        assert info["encoder"]["max_span_width"] == 3
        assert info["encoder"]["negative_ratio"] == 2.0

        loaded = SpanPipeline.load(tmp_path)
        assert loaded.label_names == ["person", "location"]
        assert loaded.none_index == 2
        assert loaded.max_span_width == 3
        assert len(loaded.label_embeddings) == 3
        before = pipeline.predict(texts, threshold=0.0)
        after = loaded.predict(texts, threshold=0.0)
        assert before == after

    @staticmethod
    def _new_pipeline():
        return SpanPipeline(
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformer",
            {"person": "a person name", "location": "a place name"},
            "This text is {}",
            max_span_width=3,
            negative_ratio=2.0,
        )
