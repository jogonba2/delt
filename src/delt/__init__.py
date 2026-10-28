"""Main package of delt."""

from delt.calibrators import BetaCalibrator, Calibrator, get_calibrator
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
    LLMSpanTextTeacher,
    LLMTextTeacher,
    LMMAudioTeacher,
    LMMImageTeacher,
    LMMVideoTeacher,
)

__all__ = [
    # Pipelines
    "AudioPipeline",
    # Calibrators
    "BetaCalibrator",
    "Calibrator",
    "get_calibrator",
    # Encoders
    "ClapEncoder",
    "ClipEncoder",
    "EmbeddingPipeline",
    "EncoderPipeline",
    "ImagePipeline",
    # Teachers
    "JevTextTeacher",
    "LLMSpanTextTeacher",
    "LLMTextTeacher",
    "LMMAudioTeacher",
    "LMMImageTeacher",
    "LMMVideoTeacher",
    "SentenceTransformerEncoder",
    "SiglipEncoder",
    "SpanPipeline",
    "TextPipeline",
    "VideoPipeline",
    "XclipEncoder",
    "get_encoder",
]
