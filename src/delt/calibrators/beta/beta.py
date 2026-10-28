"""Beta calibration (Kull et al., 2017)."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from delt.calibrators.base import Calibrator
from delt.common.logging import get_logger

logger = get_logger(__name__)


class BetaCalibrator(Calibrator):
    """
    Beta calibration of the probability of each label.

    The probability `s` of a label is mapped to `p = sigmoid(a * ln(s) - b * ln(1 - s) + c)`, with `a, b >= 0`
    so the map is monotonic. It generalizes Platt scaling, and unlike it, it can fix both under- and
    over-confidence and has the identity as a particular case.

    For more than two labels, a map is fitted for each label against the rest, and the calibrated
    probabilities are normalized to sum to one. Hence, calibration can change the label with max prob.
    A label without positive (or without negative) samples is left uncalibrated, since the map is not identifiable.
    """

    def __init__(self) -> None:
        """Initialize a beta calibrator."""
        super().__init__(name="beta")
        self.parameters: np.ndarray | None = None

    @staticmethod
    def _features(
        probs: np.ndarray, eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        probs = np.clip(probs, eps, 1 - eps)
        return np.log(probs), -np.log1p(-probs)

    @staticmethod
    def _fit_label(
        probs: np.ndarray, targets: np.ndarray
    ) -> tuple[float, float, float]:
        """Fit `(a, b, c)` for the probabilities and targets of a label."""
        ln_s, minus_ln_1_s = BetaCalibrator._features(probs)

        def loss(params: np.ndarray) -> tuple[float, np.ndarray]:
            a, b, c = params
            logits = a * ln_s + b * minus_ln_1_s + c
            # Cross-entropy, which also works for soft targets
            value = np.mean(np.logaddexp(0, logits) - targets * logits)
            errors = expit(logits) - targets
            grad = np.array(
                [
                    np.mean(errors * ln_s),
                    np.mean(errors * minus_ln_1_s),
                    np.mean(errors),
                ]
            )
            return value, grad

        result = minimize(
            loss,
            x0=np.array((1.0, 1.0, 0.0)),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None), (0, None), (None, None)],
        )
        a, b, c = result.x
        return float(a), float(b), float(c)

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "BetaCalibrator":
        """
        Fit the calibrator on predicted probabilities.

        Args:
            probs (np.ndarray): predicted probabilities with shape (N, L).
            targets (np.ndarray): expected probabilities with shape (N, L), e.g., one-hot for hard labels.

        Returns:
            BetaCalibrator: the fitted calibrator.

        """
        probs = np.asarray(probs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if probs.ndim != 2 or probs.shape != targets.shape:
            msg = (
                "`probs` and `targets` must have the same shape (N, L), "
                f"got {probs.shape} and {targets.shape}."
            )
            raise ValueError(msg)

        parameters = []
        for label in range(probs.shape[1]):
            positives = targets[:, label].sum()
            if not 0 < positives < len(targets):
                logger.warning(
                    "Label %d has no positive or no negative samples, so it is left uncalibrated.",
                    label,
                )
                parameters.append((1.0, 1.0, 0.0))
                continue
            parameters.append(
                self._fit_label(probs[:, label], targets[:, label])
            )
        self.parameters = np.array(parameters)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate predicted probabilities.

        Args:
            probs (np.ndarray): predicted probabilities with shape (N, L).

        Returns:
            np.ndarray: calibrated probabilities with shape (N, L), where each row sums to one.

        """
        if self.parameters is None:
            msg = "The calibrator is not fitted, call `fit` first."
            raise RuntimeError(msg)
        if probs.shape[-1] != len(self.parameters):
            msg = (
                f"Got probabilities for {probs.shape[-1]} labels, "
                f"but the calibrator was fitted for {len(self.parameters)}."
            )
            raise ValueError(msg)

        ln_s, minus_ln_1_s = self._features(np.asarray(probs, np.float64))
        a, b, c = self.parameters.T
        calibrated = expit(a * ln_s + b * minus_ln_1_s + c)
        return calibrated / calibrated.sum(-1, keepdims=True)

    def get_state(self) -> dict:
        """
        Get the fitted parameters of the calibrator.

        Returns:
            dict: the `(a, b, c)` parameters of each label.

        """
        if self.parameters is None:
            msg = "The calibrator is not fitted, call `fit` first."
            raise RuntimeError(msg)
        return {"parameters": self.parameters.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "BetaCalibrator":
        """
        Restore a calibrator from the output of `get_state`.

        Args:
            state (dict): the `(a, b, c)` parameters of each label.

        Returns:
            BetaCalibrator: a fitted calibrator.

        """
        calibrator = cls()
        calibrator.parameters = np.array(state["parameters"])
        return calibrator
