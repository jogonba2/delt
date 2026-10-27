"""Example script for the text span pipeline with LLM distillation."""

from delt.pipelines import SpanPipeline
from delt.pipelines.text.span import PredictedSpan, SpanAnnotation
from delt.teachers import LLMSpanTextTeacher


# Define some helper functions
def show_outputs(texts: list[str], preds: list[list[PredictedSpan]]):
    """Print the predicted spans for each text."""
    for text, spans in zip(texts, preds, strict=True):
        print(
            text,
            [(s.text, s.label, round(s.score, 2)) for s in spans],
        )


def annotate(silver: list[list[tuple[str, str]]]):
    """Convert (phrase, label) pairs into span annotations."""

    def annotate_span(text: str, phrase: str, label: str) -> SpanAnnotation:
        start = text.index(phrase)
        return SpanAnnotation(start, start + len(phrase), label)

    return [
        [annotate_span(text, phrase, label) for phrase, label in spans]
        for text, spans in zip(texts, silver, strict=True)
    ]


# Define your input data and label set
texts = [
    "Apple is a worldwide company.",
    "The AIRD team of TransPerfect is comprised of members like Ian, Celia, Mahmoud, YanRu and Llorenç.",
]
label_set = ["person", "company", "team"]

# Instantiate the teacher model
teacher = LLMSpanTextTeacher(
    "gpt-5.4-nano",
    {"temperature": 0},
    "Extract the named entities from the following text.",
)

# Generate the span silver labels (only `predict` is supported)
silver = teacher.predict(texts, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "person": "a person's name",
    "company": "a company name",
    "team": "an internal team of a company",
}


# Annotate the input data
truths = annotate(silver)

# Set the encoder
prompt_template = "This is {}"
encoder_class = "sentence-transformer"
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

# Instantiate the pipeline
pipeline = SpanPipeline(
    encoder_name,
    encoder_class,
    label_verbalizations,
    prompt_template,
    max_span_width=8,
    negative_ratio=3,
)

# Zero-shot prediction
preds = pipeline.predict(texts, batch_size=8)
show_outputs(texts, preds)

# Label-tuning training
training_output = pipeline.fit(texts, truths)

# Prediction after training
preds = pipeline.predict(texts, batch_size=8, threshold=0.8)
show_outputs(texts, preds)
