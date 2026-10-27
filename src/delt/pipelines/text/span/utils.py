"""Module for span utils."""

import re

import torch
from torch.nn import functional as F

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


def mean_pool_spans(
    token_embeddings: torch.Tensor,
    token_offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Embed spans as the L2-normalized mean of the contextual embeddings of their tokens.

    Args:
        token_embeddings (torch.Tensor): contextual token embeddings of a text with shape (T, d).
        token_offsets (list[tuple[int, int]]): (start, end) character offsets of each token, with `end` exclusive.
        spans (list[tuple[int, int]]): (start, end) character offsets of each span, with `end` exclusive. Every span must overlap at least one token.

    Returns:
        torch.Tensor: span embeddings with shape (len(spans), d).

    """
    if not spans:
        return token_embeddings.new_zeros((0, token_embeddings.shape[-1]))
    pooled = torch.stack(
        [
            token_embeddings[
                [
                    k
                    for k, (token_start, token_end) in enumerate(token_offsets)
                    if token_start < end and start < token_end
                ]
            ].mean(0)
            for start, end in spans
        ]
    )
    return F.normalize(pooled, p=2, dim=1)


def select_spans(
    candidates: list[tuple[int, int]],
    scores: list[float],
    threshold: float = 0.5,
) -> list[int]:
    """
    Greedily select the highest-scoring candidate spans above the threshold that do not overlap.

    Args:
        candidates (list[tuple[int, int]]): (start, end) character offsets of each candidate.
        scores (list[float]): score of each candidate.
        threshold (float): minimum score to keep a candidate.

    Returns:
        list[int]: indices of the selected candidates, sorted by position in the text.

    """
    order = sorted(
        (i for i, score in enumerate(scores) if score >= threshold),
        key=lambda i: (-scores[i], candidates[i][0], -candidates[i][1]),
    )
    selected: list[int] = []
    for i in order:
        start, end = candidates[i]
        if all(
            end <= candidates[j][0] or candidates[j][1] <= start
            for j in selected
        ):
            selected.append(i)
    return sorted(selected, key=lambda i: candidates[i])
