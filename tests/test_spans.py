"""Tests for `delt.pipelines.text.span.utils`."""

import torch

from delt.pipelines.text.span.utils import (
    enumerate_candidates,
    mean_pool_spans,
    select_spans,
    span_metrics,
    tokenize_with_offsets,
)


def test_tokenize_with_offsets():
    """Check that words and punctuation are split with their offsets."""
    text = "Mr. O'Neil left."
    tokens = [text[s:e] for s, e in tokenize_with_offsets(text)]
    assert tokens == ["Mr", ".", "O'Neil", "left", "."]


def test_enumerate_candidates_respects_max_width():
    """Check that candidates are not wider than the maximum width."""
    text = "a b c"
    candidates = enumerate_candidates(text, max_span_width=2)
    assert [text[s:e] for s, e in candidates] == ["a", "a b", "b", "b c", "c"]


def test_mean_pool_spans():
    """Check that spans are the normalized mean of their tokens."""
    token_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
    token_offsets = [(0, 3), (3, 5), (6, 9)]
    pooled = mean_pool_spans(token_embeddings, token_offsets, [(0, 5), (6, 9)])
    expected = torch.tensor([[1.0, 1.0], [3.0, 4.0]])
    assert torch.allclose(
        pooled, expected / expected.norm(dim=1, keepdim=True)
    )
    assert mean_pool_spans(token_embeddings, token_offsets, []).shape == (0, 2)


def test_select_spans_keeps_non_overlapping():
    """Check that only non-overlapping spans above the threshold are kept."""
    candidates = [(0, 5), (0, 2), (3, 5), (6, 9)]
    scores = [0.9, 0.8, 0.7, 0.6]
    assert select_spans(candidates, scores, 0.5) == [0, 3]
    assert select_spans(candidates, scores, 0.95) == []


def test_span_metrics():
    """Check exact-match span metrics, micro and per label."""
    predicted = [{(0, 5, "person"), (6, 11, "person")}, {(0, 3, "location")}]
    gold = [{(0, 5, "person"), (12, 16, "location")}, {(0, 3, "location")}]
    metrics = span_metrics(predicted, gold, ["person", "location", "org"])

    # person: tp=1, fp=1, fn=0. location: tp=1, fp=0, fn=1. org: unused.
    assert metrics["num_samples"] == 2
    assert metrics["micro"] == {
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
    }
    assert metrics["per_label"]["person"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "support": 1,
    }
    assert metrics["per_label"]["location"]["recall"] == 0.5
    assert metrics["per_label"]["location"]["support"] == 2
    assert metrics["per_label"]["org"]["f1"] == 0.0
    # "org" is neither gold nor predicted, so it does not count for macro F1
    assert metrics["macro_f1"] == (2 / 3 + 2 / 3) / 2


def test_span_metrics_label_must_match():
    """Check that a span with the right offsets but a wrong label is wrong."""
    metrics = span_metrics(
        [{(0, 5, "person")}], [{(0, 5, "location")}], ["person", "location"]
    )
    assert metrics["micro"]["f1"] == 0.0
    assert metrics["per_label"]["person"]["precision"] == 0.0
    assert metrics["per_label"]["location"]["recall"] == 0.0
