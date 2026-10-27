"""Tests for saving and loading label tuning pipelines."""

import json
import random

import pytest
import torch
from safetensors.torch import load_file

from delt import EmbeddingPipeline
from delt.evaluate import evaluate


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    yield


@pytest.fixture
def data():
    """Build a linearly separable dataset with three classes."""
    dim, n_per_class = 16, 32
    centers = torch.zeros(3, dim)
    centers[0, 0] = centers[1, 1] = centers[2, 2] = 1.0
    embeddings = torch.cat(
        [c + 1e-3 * torch.randn(n_per_class, dim) for c in centers]
    )
    labels = torch.arange(3).repeat_interleave(n_per_class)
    names = ["cat", "dog", "bird"]
    return embeddings, labels, centers, names


def test_save_creates_torch_and_json_files(tmp_path, data):
    """Check that embeddings are loadable with torch and info is a json."""
    embeddings, labels, centers, names = data
    pipeline = EmbeddingPipeline(centers, label_names=names)
    pipeline.fit(embeddings, labels)
    pipeline.evaluate(embeddings, labels)
    pipeline.save(tmp_path)

    tensors = load_file(tmp_path / "label_embeddings.safetensors")
    assert torch.equal(tensors["label_embeddings"], pipeline.label_embeddings)
    assert torch.equal(tensors["logit_scale"], pipeline.logit_scale)

    info = json.loads((tmp_path / "model.json").read_text())
    assert info["label_names"] == names
    assert info["embedding_dim"] == 16
    assert info["training"]["num_train_samples"] == len(labels)
    assert info["training"]["hyperparams"]["epochs"] == 2000
    assert info["validation"]["method"] == "holdout"
    assert info["validation"]["accuracy"] > 0.99
    assert set(info["validation"]["per_label"]) == set(names)


def test_roundtrip_gives_same_predictions(tmp_path, data):
    """Check that a loaded pipeline predicts like the saved one."""
    embeddings, labels, centers, names = data
    pipeline = EmbeddingPipeline(centers, label_names=names)
    pipeline.fit(embeddings, labels)
    pipeline.save(tmp_path / "nested" / "model")

    loaded = EmbeddingPipeline.load(tmp_path / "nested" / "model")
    assert loaded.label_names == names
    before = pipeline.predict(embeddings)
    after = loaded.predict(embeddings)
    assert [p.label for p in before] == [p.label for p in after]
    assert all(
        (a.probs == b.probs).all() for a, b in zip(before, after, strict=True)
    )


def test_default_logit_scale_matches_label_tuning(data):
    """Check that zero-shot uses the same scale that label tuning starts from."""
    embeddings, labels, centers, _ = data
    pipeline = EmbeddingPipeline(centers)
    assert pipeline.logit_scale.shape == (1,)
    assert pipeline.logit_scale.exp().item() == pytest.approx(1 / 0.07)

    # Orthogonal classes: the zero-shot probs of a perfect match are not flat
    probs = pipeline.predict(embeddings[:1])[0].probs
    assert probs.max() > 0.99


def test_predict_defaults_to_default_logit_scale(data):
    """Check that `predict` without a logit scale is the pipeline zero-shot."""
    from delt.predict import predict

    embeddings, _, centers, _ = data
    preds = predict(embeddings, centers)
    zero_shot = EmbeddingPipeline(centers).predict(embeddings)
    assert all(
        (a.probs == b.probs).all() and a.label == b.label
        for a, b in zip(preds, zero_shot, strict=True)
    )


def test_save_without_training_or_names(tmp_path, data):
    """Check that a zero-shot pipeline can be saved without training info."""
    _, _, centers, _ = data
    EmbeddingPipeline(centers).save(tmp_path)
    info = json.loads((tmp_path / "model.json").read_text())
    assert info["label_names"] is None
    assert info["training"] is None
    assert info["validation"] is None
    loaded = EmbeddingPipeline.load(tmp_path)
    assert loaded.label_names is None
    assert torch.equal(loaded.label_embeddings, centers)


def test_save_model_with_mismatched_label_names(tmp_path, data):
    """Check that the number of label names must match label embeddings."""
    _, _, centers, _ = data
    pipeline = EmbeddingPipeline(centers, label_names=["only_one"])
    with pytest.raises(ValueError, match="label names"):
        pipeline.save(tmp_path)


def test_evaluate_returns_and_keeps_metrics(data):
    """Check that `evaluate` returns metrics and `fit` resets them."""
    embeddings, labels, centers, names = data
    pipeline = EmbeddingPipeline(centers, label_names=names)
    assert pipeline.validation is None

    metrics = pipeline.evaluate(embeddings, [names[i] for i in labels])
    assert metrics["accuracy"] == 1.0
    assert pipeline.validation == {"method": "holdout", **metrics}

    pipeline.fit(embeddings, labels, training_args={"epochs": 5})
    assert pipeline.validation is None


def test_evaluate_with_label_names_and_soft_truths(data):
    """Check that evaluation accepts label names and probabilities."""
    embeddings, labels, centers, names = data
    scale = torch.tensor([1.0])
    by_name = evaluate(
        embeddings, centers, scale, [names[i] for i in labels], names
    )
    soft = evaluate(
        embeddings, centers, scale, torch.nn.functional.one_hot(labels, 3)
    )
    assert by_name["accuracy"] == soft["accuracy"] == 1.0
    assert by_name["per_label"]["dog"]["support"] == 32
    assert set(soft["per_label"]) == {"0", "1", "2"}


def test_cv_performance_is_saved(tmp_path, data, monkeypatch):
    """Check that the cross-validation performance ends up in `model.json`."""
    from delt import train

    monkeypatch.setattr(train, "GRID_SEARCH_PARAMS", {"epochs": [20]})
    embeddings, labels, centers, names = data
    pipeline = EmbeddingPipeline(centers, label_names=names)
    output = pipeline.fit(embeddings, labels, do_cv=True)
    assert output.validation["method"] == "cross_validation"

    pipeline.save(tmp_path)
    validation = json.loads((tmp_path / "model.json").read_text())[
        "validation"
    ]
    assert validation["num_splits"] == 4
    assert len(validation["macro_f1_per_fold"]) == 4


class _FakeEncoder:
    """Encoder that embeds text as a bag of characters."""

    def get_text_embeddings(self, texts, batch_size=16):
        out = torch.zeros(len(texts), 8)
        for i, text in enumerate(texts):
            for char in text:
                out[i, ord(char) % 8] += 1
        return torch.nn.functional.normalize(out, dim=1)


def test_encoder_pipeline_roundtrip(tmp_path, monkeypatch):
    """Check that encoder info is saved and used to rebuild the pipeline."""
    from delt import TextPipeline
    from delt.pipelines import base

    monkeypatch.setattr(base, "get_encoder", lambda *_: _FakeEncoder())
    verbalizations = {"positive": "great", "negative": "awful"}
    pipeline = TextPipeline("fake-model", "FakeEncoder", verbalizations, "{}")
    texts = ["great great", "awful awful", "great!", "awful?"]
    truths = ["positive", "negative", "positive", "negative"]
    pipeline.fit(texts, truths, training_args={"epochs": 20})
    pipeline.evaluate(texts, truths)
    pipeline.save(tmp_path)

    info = json.loads((tmp_path / "model.json").read_text())
    assert info["encoder"] == {
        "encoder_name": "fake-model",
        "encoder_class": "FakeEncoder",
        "label_verbalizations": verbalizations,
        "prompt_template": "{}",
    }
    assert info["validation"]["num_samples"] == 4

    loaded = TextPipeline.load(tmp_path)
    assert loaded.label_names == ["positive", "negative"]
    assert torch.equal(loaded.label_embeddings, pipeline.label_embeddings)
    assert [p.label for p in loaded.predict(texts)] == [
        p.label for p in pipeline.predict(texts)
    ]
