"""Tests for `delt.common.spans`."""

import pytest

from delt.common.spans import (
    enumerate_candidates,
    select_spans,
    tokenize_with_offsets,
)


def test_tokenize_with_offsets():
    text = "Mr. O'Neil left."
    tokens = [text[s:e] for s, e in tokenize_with_offsets(text)]
    assert tokens == ["Mr", ".", "O'Neil", "left", "."]


def test_enumerate_candidates_respects_max_width():
    text = "a b c"
    candidates = enumerate_candidates(
        text, max_span_width=2, filter_edges=False
    )
    assert [text[s:e] for s, e in candidates] == ["a", "a b", "b", "b c", "c"]


def test_enumerate_candidates_filters_edges():
    text = "Alice visited the Louvre."
    candidates = enumerate_candidates(text, max_span_width=3)
    assert [text[s:e] for s, e in candidates] == [
        "Alice",
        "Alice visited",
        "visited",
        "visited the Louvre",
        "Louvre",
    ]


def test_select_spans_flat_and_nested():
    candidates = [(0, 5), (0, 2), (3, 5), (6, 9)]
    scores = [0.9, 0.8, 0.7, 0.3]
    assert select_spans(candidates, scores, 0.5, "flat") == [0]
    assert select_spans(candidates, scores, 0.5, "nested") == [1, 0, 2]


def test_select_spans_invalid_overlap():
    with pytest.raises(ValueError):
        select_spans([(0, 1)], [1.0], overlap="other")  # type: ignore[arg-type]
