"""Base teacher model for classification."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class Teacher(ABC, Generic[T]):
    """Base teacher model."""

    def __init__(self, model_name: str) -> None:
        """
        Initialize a teacher model for an abstract modality.

        Args:
            model_name (str): name of the teacher model.

        """
        self.model_name = model_name

    @abstractmethod
    def predict_proba(
        self, input_data: list[T], label_set: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        """
        Compute probabilities from a teacher model for the `input_data` over the `label_set`.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            torch.Tensor: output probabilities of shape (N, L).

        """
        ...

    def predict(
        self, input_data: list[T], label_set: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        """
        Assign labels from `label_set` to the `input_data` by using a teacher model.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            torch.Tensor: output labels of shape (N,) with each label being the one with max prob for each input.

        """
        probs = self.predict_proba(input_data, label_set, batch_size)
        return probs.argmax(-1)
