"""Gaussian kernel generation and Derivative of Gaussian (DoG) filtering."""

from typing import Tuple
import numpy as np

from .convolve import convolve2d_separable, PaddingMode
from .diff import get_finite_difference_filters


def make_gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel of size (ksize, ksize).

    Args:
        ksize: Odd integer kernel size.
        sigma: Standard deviation of Gaussian.

    Returns:
        2D float64 kernel summing to 1.
    """
    if ksize % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {ksize}")
    k1d = make_gaussian_kernel_1d(ksize, sigma)
    return np.outer(k1d, k1d)


def make_gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """Create a normalized 1D Gaussian kernel of length ksize.

    Args:
        ksize: Odd integer kernel size.
        sigma: Standard deviation of Gaussian.

    Returns:
        1D float64 kernel summing to 1.
    """
    if ksize % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {ksize}")
    x = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def build_dog_filters(
    ksize: int = 13,
    sigma: float = 2.0,
    padding_mode: PaddingMode = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a 2D Gaussian kernel, and filter it with Dx and Dy to create DoG filters.

    Args:
        ksize: Size of Gaussian kernel (must be odd).
        sigma: Standard deviation of Gaussian (default 2.0).
        padding_mode: Padding mode when filtering Gaussian kernel.

    Returns:
        (G, DoG_x, DoG_y): 2D Gaussian kernel and the two DoG filters.
    """
    G = make_gaussian_kernel_2d(ksize, sigma)
    Dx, Dy = get_finite_difference_filters()
    DoG_x = convolve2d_separable(G, Dx, padding_mode=padding_mode)
    DoG_y = convolve2d_separable(G, Dy, padding_mode=padding_mode)
    return DoG_x, DoG_y


def compute_dog_derivatives(
    image: np.ndarray,
    ksize: int = 13,
    sigma: float = 2.0,
    padding_mode: PaddingMode = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute DoG filtered derivatives and gradient magnitude in a single convolution.

    Args:
        image: Grayscale float array.
        ksize: Gaussian kernel size.
        sigma: Gaussian standard deviation.
        padding_mode: Padding mode (default 'constant').

    Returns:
        (DoG_x, DoG_y, Ix_dog, Iy_dog, magnitude_dog)
    """
    DoG_x, DoG_y = build_dog_filters(ksize, sigma, padding_mode=padding_mode)
    Ix_dog = convolve2d_separable(image, DoG_x, padding_mode=padding_mode)
    Iy_dog = convolve2d_separable(image, DoG_y, padding_mode=padding_mode)
    magnitude_dog = np.sqrt(Ix_dog**2 + Iy_dog**2)
    return DoG_x, DoG_y, Ix_dog, Iy_dog, magnitude_dog
