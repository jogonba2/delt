"""Base calibrator for post-hoc calibration of predicted probabilities."""

from abc import ABC, abstractmethod

import numpy as np


class Calibrator(ABC):
    """Base calibrator, mapping predicted probabilities to calibrated ones."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "Calibrator":
        """
        Fit the calibrator on predicted probabilities.

        Args:
            probs (np.ndarray): predicted probabilities with shape (N, L).
            targets (np.ndarray): expected probabilities with shape (N, L), e.g., one-hot for hard labels.

        Returns:
            Calibrator: the fitted calibrator.

        """
        ...

    @abstractmethod
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate predicted probabilities.

        Args:
            probs (np.ndarray): predicted probabilities with shape (N, L).

        Returns:
            np.ndarray: calibrated probabilities with shape (N, L), where each row sums to one.

        """
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """
        Get the fitted parameters of the calibrator.

        Returns:
            dict: fitted parameters, which are serializable as JSON.

        """
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "Calibrator":
        """
        Restore a calibrator from the output of `get_state`.

        Args:
            state (dict): fitted parameters.

        Returns:
            Calibrator: a fitted calibrator.

        """
        ...
