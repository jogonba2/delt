"""Example script for the text pipeline."""

from delt import TextPipeline

# Set your data and configure the encoder
texts = ["I'm happy", "I'm sad", "You're strong", "Fuck you."]
label_verbalizations = {
    "positive": "really positive",
    "negative": "really negative",
}
truths = [0, 1, 0, 1]
prompt_template = "This text is {}"
encoder_class = "sentence-transformer"
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

# Instantiate the pipeline
pipeline = TextPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
preds = pipeline.predict(texts, batch_size=8)

# Label-tuning training
training_output = pipeline.fit(texts, truths)

# Prediction after training
preds = pipeline.predict(texts, batch_size=8)
