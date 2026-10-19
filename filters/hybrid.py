"""Gaussian low/high-frequency decomposition and hybrid image utilities."""

import numpy as np

from .convolve import PaddingMode, convolve2d_separable
from .gauss import make_gaussian_kernel_2d


def gaussian_blur(
    image: np.ndarray,
    sigma: float,
    padding_mode: PaddingMode = "reflect",
) -> np.ndarray:
    """Blur a grayscale or color image with a truncated Gaussian kernel."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be positive and finite, got {sigma}")

    radius = int(np.ceil(3.0 * sigma))
    kernel = make_gaussian_kernel_2d(2 * radius + 1, sigma)
    return convolve2d_separable(
        np.asarray(image, dtype=np.float64), kernel, padding_mode=padding_mode
    )


def build_hybrid_image(
    low_frequency_image: np.ndarray,
    high_frequency_image: np.ndarray,
    sigma_low: float,
    sigma_high: float,
    padding_mode: PaddingMode = "reflect",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the low-pass image, high-pass image, and their hybrid sum."""
    low_input = np.asarray(low_frequency_image, dtype=np.float64)
    high_input = np.asarray(high_frequency_image, dtype=np.float64)
    if low_input.shape != high_input.shape:
        raise ValueError(
            "Images must have matching shapes after alignment, "
            f"got {low_input.shape} and {high_input.shape}"
        )

    low_pass = gaussian_blur(low_input, sigma_low, padding_mode)
    high_pass = high_input - gaussian_blur(high_input, sigma_high, padding_mode)
    return low_pass, high_pass, low_pass + high_pass


def log_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute the centered log-magnitude Fourier spectrum of a grayscale image."""
    gray = np.asarray(image, dtype=np.float64)
    if gray.ndim == 3:
        if gray.shape[2] != 3:
            raise ValueError(f"Color image must have 3 channels, got {gray.shape}")
        gray = gray @ np.array([0.299, 0.587, 0.114])
    elif gray.ndim != 2:
        raise ValueError(f"Image must be grayscale or RGB, got shape {gray.shape}")

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.log(np.abs(spectrum) + 1e-8)