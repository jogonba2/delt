"""Package for calibrators."""

from typing import Any

from delt.calibrators.base import Calibrator
from delt.calibrators.beta import BetaCalibrator

REGISTRY: dict[str, type[Calibrator]] = {
    "beta": BetaCalibrator,
}


def get_calibrator(method: str, *args: Any, **kwargs: Any) -> Calibrator:
    """
    Instantiate a calibrator by name, given arbitrary arguments.

    Args:
        method (str): name of the calibration method. It should be available in `REGISTRY`.
        *args: positional arguments passed directly to the calibrator's constructor.
        **kwargs: keyword arguments passed directly to the calibrator's constructor.

    Returns:
        Calibrator: an instantiated calibrator, which is not fitted.

    """
    if method not in REGISTRY:
        raise KeyError(
            f"Unknown calibration method: {method}. Available: {list(REGISTRY.keys())}"
        )

    return REGISTRY[method](*args, **kwargs)


def load_calibrator(method: str, state: dict) -> Calibrator:
    """
    Restore a fitted calibrator by name.

    Args:
        method (str): name of the calibration method. It should be available in `REGISTRY`.
        state (dict): fitted parameters, as returned by `Calibrator.get_state`.

    Returns:
        Calibrator: a fitted calibrator.

    """
    if method not in REGISTRY:
        raise KeyError(
            f"Unknown calibration method: {method}. Available: {list(REGISTRY.keys())}"
        )

    return REGISTRY[method].from_state(state)


__all__ = [
    "BetaCalibrator",
    "Calibrator",
    "get_calibrator",
    "load_calibrator",
]
