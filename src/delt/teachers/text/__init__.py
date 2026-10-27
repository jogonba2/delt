"""Text teacher models."""

from delt.teachers.text.span import LLMSpanTextTeacher
from delt.teachers.text.text import JevTextTeacher, LLMTextTeacher

__all__ = ["JevTextTeacher", "LLMSpanTextTeacher", "LLMTextTeacher"]
