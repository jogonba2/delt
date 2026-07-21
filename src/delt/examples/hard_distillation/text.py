"""Example script for the text pipeline with LMM distillation."""

from delt.pipelines import TextPipeline
from delt.teachers import LLMTextTeacher

# Define your input data and label set
texts = ["I hate you", "I love you"]
label_set = ["positive", "negative"]

# Instante the teacher model
teacher = LLMTextTeacher(
    "gpt-5.4-nano",
    {"temperature": 0},
    "Classify the following texts into positive or negative for a sentiment analysis task.",
)

# Generate the labels (`predict`) or probs (`predict_proba`)
truths = teacher.predict(texts, label_set)

# Create the label verbalizations (same order as the label set)
label_verbalizations = {
    "positive": "really positive",
    "negative": "really negative",
}

# Set the encoder
prompt_template = "This text is {}"
encoder_class = "sentence-transformer"
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

# Instantiate the pipeline
pipeline = TextPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(texts, batch_size=8)

# Label-tuning to train the student model
training_output = pipeline.fit(texts, truths)

# Prediction after training
preds = pipeline.predict(texts, batch_size=8)
