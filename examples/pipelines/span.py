"""Example of span detection and classification with label tuning."""

from delt import SpanPipeline
from delt.pipelines.text import PredictedSpan, SpanAnnotation


# Define some helper functions
def show_outputs(texts: list[str], preds: list[list[PredictedSpan]]):
    """Print the predicted spans for each text."""
    for text, spans in zip(texts, preds, strict=True):
        print(
            text,
            [(s.text, s.label, round(s.score, 2)) for s in spans],
        )


def annotate(texts: list[str], gold: list[list[tuple[str, str]]]):
    """Convert (phrase, label) pairs into span annotations."""

    def annotate_span(text: str, phrase: str, label: str) -> SpanAnnotation:
        start = text.index(phrase)
        return SpanAnnotation(start, start + len(phrase), label)

    return [
        [annotate_span(text, phrase, label) for phrase, label in spans]
        for text, spans in zip(texts, gold, strict=True)
    ]


# Instantiate the pipeline
label_verbalizations = {
    "person": "a person's name",
    "location": "a city, country or place",
    "organization": "a company or organization name",
}
prompt_template = "This is {}"
encoder_class = "sentence-transformer"
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

pipeline = SpanPipeline(
    encoder_name,
    encoder_class,
    label_verbalizations,
    prompt_template,
    max_span_width=8,
    negative_ratio=3,
)

# Zero-shot prediction
test_texts = [
    "Laura works for IBM in Dublin.",
    "Nokia opened a lab in Helsinki.",
]
preds = pipeline.predict(test_texts, batch_size=8, threshold=0.8)
show_outputs(test_texts, preds)

# Label-tuning training
train_texts = [
    "Alice visited Paris last summer.",
    "Bob moved to Berlin in March.",
    "Carol flew from Madrid to Rome.",
    "David lives in London with Eve.",
    "TransPerfect is located in Barcelona and it's aimed to translation and regionalization.",
    "Google hired Maria to work on search in Zurich.",
]
train_annotated_spans = [
    [("Alice", "person"), ("Paris", "location")],
    [("Bob", "person"), ("Berlin", "location")],
    [("Carol", "person"), ("Madrid", "location"), ("Rome", "location")],
    [("David", "person"), ("London", "location"), ("Eve", "person")],
    [("TransPerfect", "organization"), ("Barcelona", "location")],
    [("Google", "organization"), ("Maria", "person"), ("Zurich", "location")],
]
train_truths = annotate(train_texts, train_annotated_spans)
training_output = pipeline.fit(train_texts, train_truths)

# Predict after training
preds = pipeline.predict(test_texts, batch_size=8, threshold=0.8)
show_outputs(test_texts, preds)

# Evaluate
test_annotated_spans = [
    [("Laura", "person"), ("IBM", "organization"), ("Dublin", "location")],
    [("Nokia", "organization"), ("Helsinki", "location")],
]
test_truths = annotate(test_texts, test_annotated_spans)
results = pipeline.evaluate(
    test_texts, test_truths, batch_size=8, threshold=0.8
)

# Save the pipeline
pipeline.save("checkpoints/my-span-model")

# Load the pipeline and predict again
pipeline = SpanPipeline.load("checkpoints/my-span-model")
preds = pipeline.predict(test_texts, batch_size=8, threshold=0.8)
show_outputs(test_texts, preds)
