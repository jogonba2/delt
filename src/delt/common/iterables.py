"""Module for iterables."""

from collections.abc import Iterable, Iterator
from itertools import islice, product
from typing import TypeVar

import torch

T = TypeVar("T")


def batchify(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """
    Create batches of `batch_size` from a generic `iterable`.

    Args:
        iterable (Iterable[T]): generic iterable to get batches from.
        batch_size (int): the batch size.

    Yields:
        list[T]: a batch of elements as a list.

    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    iterator = iter(iterable)

    while batch := list(islice(iterator, batch_size)):
        yield batch


def batchify_tensor(
    x: torch.Tensor, batch_size: int
) -> Iterator[torch.Tensor]:
    """
    Create batches of `batch_size` from a torch tensor with shape (N, ...).

    Args:
        x (torch.Tensor): the torch tensor.
        batch_size (int): the batch size.

    Yields:
        torch.Tensor: a batch of elements as a torch tensor with shape (`batch_size`, ...)

    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


def dict_cartesian_product(d: dict) -> list[dict]:
    """
    Get the cartesian product over the fields in a dictionary.

    Args:
        d (dict): a dictionary.

    Returns:
        list[dict]: list of dictionaries representing the cartesian product.

    """
    keys = d.keys()
    values = d.values()
    return [dict(zip(keys, combination)) for combination in product(*values)]
