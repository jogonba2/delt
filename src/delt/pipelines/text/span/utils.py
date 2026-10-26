"""Module for span utils."""

import re
from typing import Literal

_TOKEN_PATTERN = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]")


def tokenize_with_offsets(text: str) -> list[tuple[int, int]]:
    """
    Split a text into word and punctuation tokens, keeping character offsets.

    Args:
        text (str): the text to tokenize.

    Returns:
        list[tuple[int, int]]: (start, end) character offsets of each token, with `end` exclusive.

    """
    return [match.span() for match in _TOKEN_PATTERN.finditer(text)]


def enumerate_candidates(
    text: str, max_span_width: int = 8
) -> list[tuple[int, int]]:
    """
    Enumerate every contiguous span of up to `max_span_width` tokens.

    Args:
        text (str): the text to enumerate spans from.
        max_span_width (int): maximum number of tokens in a span.

    Returns:
        list[tuple[int, int]]: (start, end) character offsets of each candidate span, with `end` exclusive.

    """
    tokens = tokenize_with_offsets(text)
    return [
        (tokens[i][0], tokens[j][1])
        for i in range(len(tokens))
        for j in range(i, min(i + max_span_width, len(tokens)))
    ]


def select_spans(
    candidates: list[tuple[int, int]],
    scores: list[float],
    threshold: float = 0.5,
    overlap: Literal["flat", "nested"] = "flat",
) -> list[int]:
    """
    Select the candidate spans to keep, given their scores.

    Args:
        candidates (list[tuple[int, int]]): (start, end) character offsets of each candidate.
        scores (list[float]): score of each candidate.
        threshold (float): minimum score to keep a candidate.
        overlap (Literal["flat", "nested"]): "flat" greedily keeps the highest-scoring spans that do not overlap; "nested" keeps every span above the threshold.

    Returns:
        list[int]: indices of the selected candidates, sorted by position in the text.

    """
    if overlap not in ("flat", "nested"):
        msg = f"`overlap` must be 'flat' or 'nested', got {overlap!r}."
        raise ValueError(msg)

    order = sorted(
        (i for i, score in enumerate(scores) if score >= threshold),
        key=lambda i: (-scores[i], candidates[i][0], -candidates[i][1]),
    )
    if overlap == "nested":
        return sorted(order, key=lambda i: candidates[i])

    selected: list[int] = []
    for i in order:
        start, end = candidates[i]
        if all(
            end <= candidates[j][0] or candidates[j][1] <= start
            for j in selected
        ):
            selected.append(i)
    return sorted(selected, key=lambda i: candidates[i])
