"""Example script for soft distillation from a finetuned encoder model."""

from transformers import pipeline

from delt.pipelines import TextPipeline

# Define your input data
texts = ["I hate you", "I love you"]

# Instantiate the teacher model and prepare the predictions
# to be considered as truths when training with label tuning.
# `truths` here will have shape (N, L)
trf_pipeline = pipeline(
    "text-classification",
    model="finiteautomata/bertweet-base-sentiment-analysis",
)
preds = trf_pipeline(texts, top_k=None)
label_to_idx = {"POS": 0, "NEG": 1}
truths = []
for pred in preds:
    truth = [0.0, 0.0]
    for output in pred:
        if output["label"] in label_to_idx:
            truth[label_to_idx[output["label"]]] = output["score"]
    truths.append(truth)

# Create the label verbalizations (same order as the truths)
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

# Label-tuning to train the student model
training_output = pipeline.fit(texts, truths)

# Prediction after training
preds = pipeline.predict(texts, batch_size=8)
