"""Package for teacher models."""

from .audio import LMMAudioTeacher
from .image import LMMImageTeacher
from .text import LLMTextTeacher
from .video import LMMVideoTeacher

__all__ = [
    "LMMAudioTeacher",
    "LMMImageTeacher",
    "LLMTextTeacher",
    "LMMVideoTeacher",
]
