"""Module for span types."""

from dataclasses import dataclass


@dataclass
class SpanAnnotation:
    """A gold span annotation, with character offsets (`end` is exclusive)."""

    start: int
    end: int
    label: int | str


@dataclass
class PredictedSpan:
    """A predicted span, with character offsets (`end` is exclusive)."""

    start: int
    end: int
    text: str
    label: str
    score: float
    probs: list[float]
