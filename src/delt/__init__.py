"""Main package of delt."""

from delt.encoders import (
    ClapEncoder,
    ClipEncoder,
    SentenceTransformerEncoder,
    SiglipEncoder,
    XclipEncoder,
    get_encoder,
)
from delt.pipelines import (
    AudioPipeline,
    EmbeddingPipeline,
    EncoderPipeline,
    ImagePipeline,
    SpanPipeline,
    TextPipeline,
    VideoPipeline,
)
from delt.teachers import (
    JevTextTeacher,
    LLMTextTeacher,
    LMMAudioTeacher,
    LMMImageTeacher,
    LMMVideoTeacher,
)

ENCODERS = [
    "ClapEncoder",
    "ClipEncoder",
    "SentenceTransformerEncoder",
    "SiglipEncoder",
    "XclipEncoder",
    "get_encoder",
]
PIPELINES = [
    "AudioPipeline",
    "EmbeddingPipeline",
    "EncoderPipeline",
    "ImagePipeline",
    "SpanPipeline",
    "TextPipeline",
    "VideoPipeline",
]
TEACHERS = [
    "JevTextTeacher",
    "LLMTextTeacher",
    "LMMAudioTeacher",
    "LMMImageTeacher",
    "LMMVideoTeacher",
]

__all__ = ENCODERS + PIPELINES + TEACHERS
