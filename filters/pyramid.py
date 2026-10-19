"""Gaussian and Laplacian image pyramids."""

import numpy as np

from .convolve import PaddingMode, convolve2d_separable

_PYRAMID_TAPS = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
_PYRAMID_KERNEL = np.outer(_PYRAMID_TAPS, _PYRAMID_TAPS)


def _as_image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float64)
    if result.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape {result.shape}")
    return result


def reduce(
    image: np.ndarray,
    padding_mode: PaddingMode = "reflect",
) -> np.ndarray:
    """Blur with the separable 5-tap binomial kernel, then keep every other pixel."""
    image = _as_image(image)
    blurred = convolve2d_separable(image, _PYRAMID_KERNEL, padding_mode=padding_mode)
    return blurred[::2, ::2, ...]


def expand(
    image: np.ndarray,
    padding_mode: PaddingMode = "reflect",
) -> np.ndarray:
    """Zero-insert at each 2x2 block's top-left, blur, and scale by four."""
    image = _as_image(image)
    height, width = image.shape[:2]
    shape = (2 * height, 2 * width) + image.shape[2:]
    upsampled = np.zeros(shape, dtype=np.float64)
    upsampled[::2, ::2, ...] = image
    blurred = convolve2d_separable(
        upsampled, _PYRAMID_KERNEL, padding_mode=padding_mode
    )
    return 4.0 * blurred


def gaussian_pyramid(
    image: np.ndarray,
    levels: int = 5,
    padding_mode: PaddingMode = "reflect",
) -> list[np.ndarray]:
    """Build up to ``levels`` Gaussian images, stopping at 1x1."""
    if levels < 1:
        raise ValueError(f"levels must be at least 1, got {levels}")

    pyramid = [_as_image(image)]
    for _ in range(levels - 1):
        if pyramid[-1].shape[0] == 1 and pyramid[-1].shape[1] == 1:
            break
        pyramid.append(reduce(pyramid[-1], padding_mode=padding_mode))
    return pyramid


def laplacian_pyramid(
    image: np.ndarray,
    levels: int = 5,
    padding_mode: PaddingMode = "reflect",
) -> list[np.ndarray]:
    """Build Laplacian residuals and append the smallest Gaussian image."""
    gaussian = gaussian_pyramid(image, levels=levels, padding_mode=padding_mode)
    laplacian = []
    for current, smaller in zip(gaussian[:-1], gaussian[1:]):
        expanded = expand(smaller, padding_mode=padding_mode)
        expanded = expanded[: current.shape[0], : current.shape[1], ...]
        laplacian.append(current - expanded)
    laplacian.append(gaussian[-1].copy())
    return laplacian


def reconstruct_laplacian(
    pyramid: list[np.ndarray],
    padding_mode: PaddingMode = "reflect",
) -> np.ndarray:
    """Reconstruct an image from its Laplacian pyramid."""
    if not pyramid:
        raise ValueError("Laplacian pyramid must contain at least one level")

    image = _as_image(pyramid[-1]).copy()
    for residual in reversed(pyramid[:-1]):
        residual = _as_image(residual)
        expanded = expand(image, padding_mode=padding_mode)
        expanded = expanded[: residual.shape[0], : residual.shape[1], ...]
        if expanded.shape != residual.shape:
            raise ValueError(
                "Pyramid levels have incompatible shapes: "
                f"expanded {expanded.shape}, residual {residual.shape}"
            )
        image = expanded + residual
    return image