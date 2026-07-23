
<div align="center">

# 🧲 DELT

### **Zero-shot and Few-shot Multimodal Classification with Label Tuning**

A Dual Encoder toolkit for zero-shot learning, label tuning, and distillation.

[![License](https://img.shields.io/badge/license-CC_BY_NC_ND_4.0-green)](LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0-green)](CODE_OF_CONDUCT.md)

</div>

---

## 👋 DELT
Building high-quality classifiers usually requires collecting thousands of labeled examples or relying on strong models with large resource conssumption. `delt` was built for the situations where that simply is not practical, adopting dual encoders as backbone.

With `delt` you can:

- 🚀 Perform **zero-shot classification** without training data.
- 🎯 Fine-tune classifiers using only a few labeled examples through **label tuning**.
- 🌍 Work across **text, images, audio, and video** using a unified API.
- ⚡ Train in seconds by optimizing only lightweight label embeddings.
- 💻 Run inference efficiently on CPU once embeddings are available.
- 🤖 Distill powerful LLMs and multimodal models into compact, production-ready classifiers.

Whether you have **zero**, **ten**, or **millions** annotated samples, `delt` provides a simple pipeline that scales with your data.


# ✨ How it works

`delt` is built around **dual-encoder embedding models**, supporting **zero-shot** classification and finetuning through **label tuning**.

If you have no annotated data, `delt` performs zero-shot classification directly using pretrained embedding models, that is: no optimization, no gradient updates, no training loop, simply define your labels and start predicting.

When a labeled dataset is available (few data or a large volume), `delt` optimizes the label embeddings through **label tuning** while keeping the encoder frozen. This offers embedding reusability, dramatically fewer trainable parameters, faster training, smaller deployment artifacts, and competitive accuracy, making it especially suitable for embedding-based applications where embeddings are precomputed—such as retrieval-augmented generation (RAG) systems, visualization tools, and semantic search pipelines. The idea is simple and was originally proposed in (Müller Thomas, et al., 2022) and has been here extended for multimodal dual encoders:

Given a sentence like

> *"I love this movie."*

and the label

> *"positive"*

`delt` finetunes label embeddings computed from pretrained embedding models such that

```
❄️embedding("I love this movie")❄️ ≈ 🔥embedding("This review is positive")🔥
```

where ❄️ and 🔥 refer to frozen and trainable parameters. Unlike traditional training methods, the encoder remains frozen while only the label representations are optimized. In practice, the trainable component contains only (number_of_labels × embedding_dimension) parameters, making label tuning remarkably efficient while preserving the knowledge of the original embedding model.

When human-annotated data is unavailable and zero-shot methods do not deliver sufficient performance, delt provides distillation as an alternative. It supports both **soft distillation**, where label embeddings are fine-tuned to match the probability distribution produced by more powerful but computationally expensive models, and **hard distillation**, where label embeddings are fine-tuned to reproduce the labels predicted by models that do not provide class probabilities (e.g., LLMs and LMMs).
This approach enables knowledge from highly capable models to be compressed into a relatively small set of parameters, making it practical to deploy their capabilities in large-scale applications where running an LLM or LMM on every example would be prohibitively expensive. You can just select a random set of your million documents, annotate it with an LLM, train a model with `delt`, and predict all the remaining data with your brand new model in a more efficient way.

# 🚀 Supported modalities

`delt` provides ready-to-use encoders for every modality. However, you're also free to use precomputed embeddings instead.

| Modality | Provided encoders |
|----------|------------------|
| 🈳 Text | Sentence Transformers |
| 🖼 Image | CLIP, SigLIP, SigLIP2 |
| 🔊 Audio | CLAP, GLAP |
| 🎥 Video | X-CLIP |

Every modality follows exactly the same workflow, so learning one modality means learning them all:

<p align="center">
  <img src="https://github.com/jogonba2/delt/blob/main/assets/pipeline.png" alt="Pipeline" width="700"><br>
  <strong>Figure 1.</strong> A pipeline in delt works in the same way for every modality.
</p>


# 📦 Installation

You can install `delt` using either `uv` or `pip`:

```bash
uv add deltpy
```

or

```bash
pip install deltpy
```

For some cases such as hard distillation with LLMs, you will need environment variables depending on the LLM you use (through LiteLLM):

```bash
OPENAI_API_KEY=<KEY>
GEMINI_API_KEY=<KEY>
```

# 🚀 Pipeline example

The API is intentionally minimal. For examples of how to use `delt` for every modality, take a look to the [pipeline examples folder](src/delt/examples/pipelines).

```python
from delt import TextPipeline

# Instantiate the pipeline
pipeline = TextPipeline(
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    encoder_class="sentence-transformer",
    label_verbalizations={
        "positive": "really positive",
        "negative": "really negative",
    },
    prompt_template="This text is {}",
)

# Define your data
texts = ["I'm happy", "I'm sad"]
truths = [0, 1, 0, 1]

# Zero-shot
predictions = pipeline.predict(texts)

# Few-shot label tuning
pipeline.train(texts, labels)

# Predict again
predictions = pipeline.predict(texts)
```

The same design applies to **images**, **audio**, and **video**.

# ⚡ Label tuning example
If you already have precomputed embeddings and annotated training data, you can use label_tuning instead of the pipelines to train a classifier's label embeddings.

```python
import torch
from delt.predict import predict
from delt.train import label_tuning

# Input embeddings
embeddings = torch.cat([torch.randn(100, 16), torch.randn(100, 16) + 2])
labels = torch.cat(
    [torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)]
)

# Initial label embeddings
label_embeddings = torch.stack([torch.randn(16), torch.randn(16) + 2])

# Fine-tune label embeddings
output = label_tuning(
    embeddings,
    label_embeddings,
    labels,
    epochs=500,
)

# Predict
predictions = predict(
    embeddings,
    output.label_embeddings,
    output.logit_scale,
)
```


# 🧪 Distillation example

Instead of manually annotating thousands of examples, a stronger model acts as a **teacher**, while `delt` learns a lightweight **student**.

Both **soft** and **hard** distillation are supported by `delt`. You can take a look to the [soft distillation](src/delt/examples/soft_distillation/) and [hard distillation](src/delt/examples/hard_distillation/) folders to see how it works. For the sake of this section's usefulness, here we show an example to distill `gpt-5.4-nano` into label embeddings for a sentiment analysis task:

```python
from delt.pipelines import TextPipeline
from delt.teachers import LLMTextTeacher

# Example data
texts = ["I hate you", "I love you"]
label_set = ["positive", "negative"]

# Generate training labels with an LLM
teacher = LLMTextTeacher(
    model="gpt-5.4-nano",
    model_kwargs={"temperature": 0},
    instruction="Classify each text as positive or negative.",
)

truths = teacher.predict(texts, label_set)

# Create and train the student model
pipeline = TextPipeline(
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    encoder_class="sentence-transformer",
    label_verbalizations={
        "positive": "really positive",
        "negative": "really negative",
    },
    prompt_template="This text is {}",
)

pipeline.fit(texts, truths)

# Predict
predictions = pipeline.predict(texts)
```


# 🖥 User Interface

`delt` ships with a lightweight Streamlit application for experimentation.

```bash
uv run streamlit run src/delt/ui/app.py
```

# 📚 How to cite

`delt` extends the label tuning strategy, introduced in (Müller Thomas, et al., 2022), for multimodal dual encoders. If `delt` contributes to your research, please consider citing the original paper.

```bibtex
@inproceedings{muller-etal-2022-shot,
    title = "Few-Shot Learning with {S}iamese Networks and Label Tuning",
    author = {M{\"u}ller, Thomas  and
      P{\'e}rez-Torr{\'o}, Guillermo  and
      Franco-Salvador, Marc},
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2022.acl-long.584",
    pages = "8532--8545",
}
```


# 🤝 Contributing

Contributions are always welcome. Please make sure to:

- Install the development dependencies
- Format your code before submitting
- Follow the project's coding standards
- Open discussions for larger feature proposals
