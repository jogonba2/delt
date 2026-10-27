"""Base teacher model for span classification."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class SpanTeacher(ABC, Generic[T]):
    """Base teacher model."""

    def __init__(self, model_name: str) -> None:
        """
        Initialize a teacher model for an abstract modality.

        Args:
            model_name (str): name of the teacher model.

        """
        self.model_name = model_name

    @abstractmethod
    def predict(
        self, input_data: list[T], label_set: list[str], batch_size: int = 8
    ) -> list[list[tuple[str, str]]]:
        """
        Assign labels from `label_set` to the spans in `input_data` by using a teacher model.

        Args:
            input_data (list[T]): list of input data.
            label_set (list[str]): set of labels in your task.
            batch_size (int): batch size.

        Returns:
            list[list[tuple[str, str]]]: a list of spans for each input, e.g., [[("person", "Jose)]]

        """
        ...
