"""Multiresolution blending with Laplacian image pyramids."""

import numpy as np

from .pyramid import gaussian_pyramid, laplacian_pyramid, reconstruct_laplacian


def multiresolution_blend(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    levels: int = 5,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Blend two same-sized images across a mask at each Laplacian level.

    The mask is kept sharp as input; its Gaussian pyramid supplies a different
    transition width at each image frequency band.
    """
    image1 = np.asarray(image1, dtype=np.float64)
    image2 = np.asarray(image2, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if image1.shape != image2.shape:
        raise ValueError(f"Images must have matching shapes, got {image1.shape} and {image2.shape}")
    if image1.ndim not in (2, 3):
        raise ValueError(f"Images must be 2D or 3D, got shape {image1.shape}")
    if mask.shape != image1.shape[:2]:
        raise ValueError(f"Mask must match image height and width, got {mask.shape}")
    if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("Mask values must be finite and within [0, 1]")

    laplacian1 = laplacian_pyramid(image1, levels=levels)
    laplacian2 = laplacian_pyramid(image2, levels=levels)
    mask_gaussian = gaussian_pyramid(mask, levels=levels)
    weighted1 = []
    weighted2 = []
    blended_pyramid = []

    for level1, level2, level_mask in zip(laplacian1, laplacian2, mask_gaussian):
        if level1.ndim == 3:
            level_mask = level_mask[..., None]
        first_part = level_mask * level1
        second_part = (1.0 - level_mask) * level2
        weighted1.append(first_part)
        weighted2.append(second_part)
        blended_pyramid.append(first_part + second_part)

    blended = reconstruct_laplacian(blended_pyramid)
    return blended, weighted1, weighted2, blended_pyramid