"""Tests for calibrators and `calibrate` in pipelines."""

import json
import random

import numpy as np
import pytest
import torch

from delt import BetaCalibrator, EmbeddingPipeline, get_calibrator
from delt.calibrators import load_calibrator
from delt.evaluate import expected_calibration_error


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(13)
    torch.manual_seed(13)
    np.random.seed(13)
    yield


def _nll(probs, truths):
    return -np.log(probs[np.arange(len(truths)), truths]).mean()


@pytest.fixture
def overconfident():
    """Build overconfident probs, i.e., labels are right less often than the probs say."""
    rng = np.random.default_rng(13)
    n, num_labels = 600, 3
    truths = rng.integers(0, num_labels, n)
    logits = rng.normal(size=(n, num_labels))
    # Only a bit of signal, but scaled up so softmax is very peaked
    logits[np.arange(n), truths] += 1.0
    logits *= 6
    probs = torch.tensor(logits).softmax(-1).numpy()
    return probs, truths


@pytest.fixture
def data():
    """Build noisy embeddings with three classes, so probs are not saturated."""
    dim, n_per_class = 16, 64
    centers = torch.zeros(3, dim)
    centers[0, 0] = centers[1, 1] = centers[2, 2] = 1.0
    embeddings = torch.cat(
        [c + 0.8 * torch.randn(n_per_class, dim) for c in centers]
    )
    labels = torch.arange(3).repeat_interleave(n_per_class)
    return embeddings, labels, centers, ["cat", "dog", "bird"]


class TestBetaCalibrator:
    """Class to test `BetaCalibrator`."""

    def test_reduces_nll_on_overconfident_probs(self, overconfident):
        """Check that calibration improves the log-loss of overconfident probs."""
        probs, truths = overconfident
        targets = np.eye(3)[truths]
        calibrator = BetaCalibrator().fit(probs, targets)
        calibrated = calibrator.calibrate(probs)
        assert _nll(calibrated, truths) < _nll(probs, truths)

    def test_output_is_a_distribution(self, overconfident):
        """Check that calibrated probs sum to one."""
        probs, truths = overconfident
        calibrator = BetaCalibrator().fit(probs, np.eye(3)[truths])
        calibrated = calibrator.calibrate(probs)
        assert calibrated.shape == probs.shape
        assert np.allclose(calibrated.sum(-1), 1.0)
        assert (calibrated > 0).all()

    def test_parameters_are_monotonic(self, overconfident):
        """Check that `a` and `b` are non-negative."""
        probs, truths = overconfident
        calibrator = BetaCalibrator().fit(probs, np.eye(3)[truths])
        assert calibrator.parameters.shape == (3, 3)
        assert (calibrator.parameters[:, :2] >= 0).all()

    def test_identity_parameters_keep_probs(self, overconfident):
        """Check that `(a, b, c) = (1, 1, 0)` is the identity."""
        probs, _ = overconfident
        calibrator = BetaCalibrator()
        calibrator.parameters = np.array([[1.0, 1.0, 0.0]] * 3)
        assert np.allclose(calibrator.calibrate(probs), probs, atol=1e-5)

    def test_well_calibrated_probs_stay_close(self):
        """Check that already calibrated probs are not changed much."""
        rng = np.random.default_rng(13)
        probs = rng.dirichlet(np.ones(2), size=5000)
        truths = (rng.random(5000) > probs[:, 0]).astype(int)
        calibrator = BetaCalibrator().fit(probs, np.eye(2)[truths])
        assert np.abs(calibrator.calibrate(probs) - probs).max() < 0.05

    def test_soft_targets(self, overconfident):
        """Check that expected probabilities are valid targets."""
        probs, truths = overconfident
        targets = 0.9 * np.eye(3)[truths] + 0.1 / 3
        calibrator = BetaCalibrator().fit(probs, targets)
        assert np.isfinite(calibrator.parameters).all()

    def test_label_without_positives_is_left_uncalibrated(self, overconfident):
        """Check that a label that never happens gets the identity map."""
        probs, truths = overconfident
        truths = np.where(truths == 2, 0, truths)
        calibrator = BetaCalibrator().fit(probs, np.eye(3)[truths])
        assert calibrator.parameters[2].tolist() == [1.0, 1.0, 0.0]
        assert np.isfinite(calibrator.calibrate(probs)).all()

    def test_state_roundtrip(self, overconfident):
        """Check that a restored calibrator calibrates like the original."""
        probs, truths = overconfident
        calibrator = BetaCalibrator().fit(probs, np.eye(3)[truths])
        state = json.loads(json.dumps(calibrator.get_state()))
        restored = load_calibrator("beta", state)
        assert np.array_equal(
            restored.calibrate(probs), calibrator.calibrate(probs)
        )

    def test_not_fitted(self, overconfident):
        """Check that an unfitted calibrator cannot be used."""
        probs, _ = overconfident
        with pytest.raises(RuntimeError, match="not fitted"):
            BetaCalibrator().calibrate(probs)
        with pytest.raises(RuntimeError, match="not fitted"):
            BetaCalibrator().get_state()

    def test_wrong_number_of_labels(self, overconfident):
        """Check that probs must match the fitted labels."""
        probs, truths = overconfident
        calibrator = BetaCalibrator().fit(probs, np.eye(3)[truths])
        with pytest.raises(ValueError, match="labels"):
            calibrator.calibrate(probs[:, :2])

    def test_registry(self):
        """Check that calibrators are available by name."""
        assert isinstance(get_calibrator("beta"), BetaCalibrator)
        with pytest.raises(KeyError, match="Unknown calibration method"):
            get_calibrator("unknown")


class TestPipelineCalibration:
    """Class to test `EmbeddingPipeline.calibrate`."""

    def test_calibrate_changes_predicted_probs(self, data):
        """Check that `predict` applies the calibrator and probs stay valid."""
        embeddings, labels, centers, names = data
        pipeline = EmbeddingPipeline(centers, label_names=names)
        before = pipeline.predict(embeddings)
        calibrator = pipeline.calibrate(embeddings, labels)
        after = pipeline.predict(embeddings)

        assert pipeline.calibrator is calibrator
        assert not np.allclose(
            [p.probs for p in before], [p.probs for p in after]
        )
        assert np.allclose([sum(p.probs) for p in after], 1.0)

    def test_calibrate_improves_nll(self, data):
        """Check that calibration improves the log-loss of the zero-shot pipeline."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        raw = np.array([p.probs for p in pipeline.predict(embeddings)])
        pipeline.calibrate(embeddings, labels)
        calibrated = np.array([p.probs for p in pipeline.predict(embeddings)])
        assert _nll(calibrated, labels.numpy()) < _nll(raw, labels.numpy())

    def test_calibrate_with_label_names_and_lists(self, data):
        """Check that label names and lists are valid inputs."""
        embeddings, labels, centers, names = data
        pipeline = EmbeddingPipeline(centers, label_names=names)
        truths = [names[i] for i in labels.tolist()]
        pipeline.calibrate(embeddings.tolist(), truths)
        reference = EmbeddingPipeline(centers, label_names=names)
        reference.calibrate(embeddings, labels)
        assert np.allclose(
            pipeline.calibrator.parameters, reference.calibrator.parameters
        )

    def test_calibrate_with_soft_truths(self, data):
        """Check that expected probabilities are valid truths."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        soft = torch.nn.functional.one_hot(labels, 3).float() * 0.8 + 0.2 / 3
        pipeline.calibrate(embeddings, soft)
        assert pipeline.calibrator is not None

    def test_calibrate_does_not_use_previous_calibrator(self, data):
        """Check that recalibrating gives the same result."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        parameters = pipeline.calibrate(embeddings, labels).parameters.copy()
        again = pipeline.calibrate(embeddings, labels).parameters
        assert np.allclose(parameters, again)

    def test_unknown_method(self, data):
        """Check that unknown methods are rejected."""
        embeddings, labels, centers, _ = data
        with pytest.raises(KeyError, match="Unknown calibration method"):
            EmbeddingPipeline(centers).calibrate(
                embeddings, labels, method="unknown"
            )

    def test_evaluate_uses_calibrator(self, data):
        """Check that `evaluate` is computed with calibrated probs."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        pipeline.calibrator = BetaCalibrator()
        # A map that always favors the first label
        pipeline.calibrator.parameters = np.array(
            [[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        metrics = pipeline.evaluate(embeddings, labels)
        assert metrics["accuracy"] == pytest.approx(1 / 3)

    def test_fit_discards_calibrator(self, data):
        """Check that retraining invalidates the calibrator."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        pipeline.calibrate(embeddings, labels)
        pipeline.fit(embeddings, labels)
        assert pipeline.calibrator is None

    def test_save_load_roundtrip(self, tmp_path, data):
        """Check that a loaded pipeline predicts like the saved calibrated one."""
        embeddings, labels, centers, names = data
        pipeline = EmbeddingPipeline(centers, label_names=names)
        pipeline.fit(embeddings, labels)
        pipeline.calibrate(embeddings, labels)
        pipeline.save(tmp_path)

        info = json.loads((tmp_path / "model.json").read_text())
        assert info["calibrator"]["method"] == "beta"
        assert len(info["calibrator"]["state"]["parameters"]) == 3

        loaded = EmbeddingPipeline.load(tmp_path)
        assert isinstance(loaded.calibrator, BetaCalibrator)
        before = pipeline.predict(embeddings)
        after = loaded.predict(embeddings)
        assert [p.label for p in before] == [p.label for p in after]
        assert np.allclose([p.probs for p in before], [p.probs for p in after])

    def test_save_load_without_calibrator(self, tmp_path, data):
        """Check that models saved without calibrator, or before calibration existed, load."""
        _, _, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        pipeline.save(tmp_path)
        info = json.loads((tmp_path / "model.json").read_text())
        assert info["calibrator"] is None
        assert EmbeddingPipeline.load(tmp_path).calibrator is None

        del info["calibrator"]
        (tmp_path / "model.json").write_text(json.dumps(info))
        assert EmbeddingPipeline.load(tmp_path).calibrator is None


class TestExpectedCalibrationError:
    """Class to test `ece` in `evaluate`."""

    def test_hand_computed_value(self):
        """Check the ECE against a value computed by hand."""
        # Bin (0.9, 1.0]: conf 0.95, acc 1 -> 2/4 * 0.05
        # Bin (0.5, 0.6]: conf 0.6, acc 0.5 -> 2/4 * 0.1
        probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.6, 0.4], [0.6, 0.4]])
        y_true = np.array([0, 0, 0, 1])
        assert expected_calibration_error(probs, y_true) == pytest.approx(
            0.075
        )

    def test_perfectly_calibrated_bins(self):
        """Check that the ECE is zero when confidence matches accuracy."""
        probs = np.array([[0.75, 0.25]] * 4)
        y_true = np.array([0, 0, 0, 1])
        assert expected_calibration_error(probs, y_true) == pytest.approx(0)

    def test_confidence_one_and_uniform_probs(self):
        """Check the edges of the bins."""
        assert expected_calibration_error(
            np.array([[1.0, 0.0]]), np.array([1])
        ) == pytest.approx(1.0)
        # Max prob 0.5 is in the (0.466, 0.533] bin
        assert expected_calibration_error(
            np.array([[0.5, 0.5]] * 2), np.array([0, 1])
        ) == pytest.approx(0)

    def test_evaluate_reports_ece(self, data):
        """Check that `evaluate` includes the ECE, also for soft truths."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        metrics = pipeline.evaluate(embeddings, labels)
        assert 0 <= metrics["ece"] <= 1
        soft = torch.nn.functional.one_hot(labels, 3).float()
        assert pipeline.evaluate(embeddings, soft)["ece"] == pytest.approx(
            metrics["ece"]
        )

    def test_calibration_reduces_ece(self, data):
        """Check that calibrating lowers the ECE of the zero-shot pipeline."""
        embeddings, labels, centers, _ = data
        pipeline = EmbeddingPipeline(centers)
        before = pipeline.evaluate(embeddings, labels)["ece"]
        pipeline.calibrate(embeddings, labels)
        after = pipeline.evaluate(embeddings, labels)["ece"]
        assert after < before
