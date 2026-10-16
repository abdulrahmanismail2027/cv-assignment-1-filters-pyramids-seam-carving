"""Finite difference operations and derivative utilities."""

from typing import Tuple
import numpy as np

from .convolve import convolve2d_separable, PaddingMode


def rescale_midgray(image: np.ndarray) -> np.ndarray:
    """Scale a signed array with negative values to [0, 1] with zero at mid-gray (0.5).

    Values >= 0 map to [0.5, 1.0], values <= 0 map to [0.0, 0.5].
    """
    max_abs = np.max(np.abs(image))
    if max_abs < 1e-12:
        return np.full_like(image, 0.5, dtype=np.float32)
    rescaled = 0.5 + 0.5 * (image / max_abs)
    return np.clip(rescaled, 0.0, 1.0).astype(np.float32)


def get_finite_difference_filters() -> Tuple[np.ndarray, np.ndarray]:
    """Return finite difference operators Dx = [1, 0, -1] and Dy = Dx^T."""
    Dx = np.array([[1.0, 0.0, -1.0]], dtype=np.float64)
    Dy = Dx.T
    return Dx, Dy


def compute_finite_differences(
    image: np.ndarray,
    padding_mode: PaddingMode = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute finite differences along x and y, and the gradient magnitude.

    Uses separable convolution convolve2d_separable.
    Dx = [1 0 -1] (shape 1x3)
    Dy = Dx.T     (shape 3x1)

    Args:
        image: Grayscale float array.
        padding_mode: Padding mode (default 'constant' zero-padding).

    Returns:
        (Ix, Iy, magnitude): Derivatives along x, y, and gradient magnitude.
    """
    Dx, Dy = get_finite_difference_filters()
    Ix = convolve2d_separable(image, Dx, padding_mode=padding_mode)
    Iy = convolve2d_separable(image, Dy, padding_mode=padding_mode)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, magnitude
