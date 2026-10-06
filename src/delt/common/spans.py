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


_STOPWORDS = frozenset(
    "a an the this that these those and or but nor of in on at to from by with for as into onto "
    "over under about is are was were be been being am has have had do does did will would can could "
    "should may might must shall it its it's he she they we you i me him her them us my your his their "
    "our not no than then so if while when where which who whom whose what lot lots some any each every "
    "such very too also just only".split()
)


def _is_valid_edge(token: str) -> bool:
    """Whether a token can start or end a span: it must be a word and not a stopword."""
    return any(c.isalnum() for c in token) and token.lower() not in _STOPWORDS


def enumerate_candidates(
    text: str, max_span_width: int = 8, filter_edges: bool = True
) -> list[tuple[int, int]]:
    """
    Enumerate every contiguous span of up to `max_span_width` tokens.

    Args:
        text (str): the text to enumerate spans from.
        max_span_width (int): maximum number of tokens in a span.
        filter_edges (bool): drop spans that start or end with punctuation or a stopword, since they are hardly ever spans of interest.

    Returns:
        list[tuple[int, int]]: (start, end) character offsets of each candidate span, with `end` exclusive.

    """
    tokens = tokenize_with_offsets(text)
    valid = [_is_valid_edge(text[start:end]) for start, end in tokens]
    return [
        (tokens[i][0], tokens[j][1])
        for i in range(len(tokens))
        for j in range(i, min(i + max_span_width, len(tokens)))
        if not filter_edges or (valid[i] and valid[j])
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
