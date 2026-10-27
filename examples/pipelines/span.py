"""Example of span detection and classification with label tuning."""

from delt import SpanPipeline
from delt.pipelines.text import PredictedSpan, SpanAnnotation


# Define some helper functions
def show_outputs(texts: list[str], preds: list[list[PredictedSpan]]):
    """Print the predicted spans for each text."""
    for text, spans in zip(texts, preds, strict=True):
        print(
            text,
            [
                (s.text, pipeline.label_names[s.label], round(s.score, 2))
                for s in spans
            ],
        )


def annotate(gold: list[list[tuple[str, str]]]):
    """Convert (phrase, label) pairs into span annotations."""

    def annotate_span(text: str, phrase: str, label: str) -> SpanAnnotation:
        start = text.index(phrase)
        return SpanAnnotation(start, start + len(phrase), label)

    return [
        [annotate_span(text, phrase, label) for phrase, label in spans]
        for text, spans in zip(texts, gold, strict=True)
    ]


# Define your annotated data and label set
texts = [
    "Alice visited Paris last summer.",
    "Bob moved to Berlin in March.",
    "Carol flew from Madrid to Rome.",
    "David lives in London with Eve.",
    "TransPerfect is located in Barcelona and it's aimed to translation and regionalization.",
    "Google hired Maria to work on search in Zurich.",
    "Microsoft released a new version of Windows last week.",
    "The engineers at Siemens met John in Munich.",
    "Amazon opened an office in Tokyo, and Sarah joined the AI team.",
    "The package was shipped by Tesla to Peter in Oslo.",
]
gold = [
    [("Alice", "person"), ("Paris", "location")],
    [("Bob", "person"), ("Berlin", "location")],
    [("Carol", "person"), ("Madrid", "location"), ("Rome", "location")],
    [("David", "person"), ("London", "location"), ("Eve", "person")],
    [("TransPerfect", "organization"), ("Barcelona", "location")],
    [("Google", "organization"), ("Maria", "person"), ("Zurich", "location")],
    [("Microsoft", "organization")],
    [("Siemens", "organization"), ("John", "person"), ("Munich", "location")],
    [("Amazon", "organization"), ("Tokyo", "location"), ("Sarah", "person")],
    [("Tesla", "organization"), ("Peter", "person"), ("Oslo", "location")],
]
label_verbalizations = {
    "person": "a person's name",
    "location": "a city, country or place",
    "organization": "a company or organization name",
}

# Annotate the input data
truths = annotate(gold)

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
