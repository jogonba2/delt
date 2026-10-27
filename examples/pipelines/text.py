"""Example script for the text pipeline."""

from delt import TextPipeline

# Instantiate the pipeline
label_verbalizations = {
    "positive": "really positive",
    "negative": "really negative",
}
prompt_template = "This text is {}"
encoder_class = "sentence-transformer"
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
pipeline = TextPipeline(
    encoder_name, encoder_class, label_verbalizations, prompt_template
)

# Zero-shot prediction
test_texts = ["I don't like you"]
preds = pipeline.predict(test_texts, batch_size=8)

# Label-tuning training
training_texts = ["I'm happy", "I'm sad", "You're strong", "I hate you."]
training_truths = ["positive", "negative", "positive", "negative"]
training_output = pipeline.fit(training_texts, training_truths)

# Predict after training
preds = pipeline.predict(test_texts, batch_size=8)

# Evaluate
test_truths = ["negative"]
results = pipeline.evaluate(test_texts, test_truths, batch_size=8)

# Save the pipeline
pipeline.save("checkpoints/my-text-model")

# Load the pipeline and predict again
pipeline = TextPipeline.load("checkpoints/my-text-model")
preds = pipeline.predict(test_texts, batch_size=8)
