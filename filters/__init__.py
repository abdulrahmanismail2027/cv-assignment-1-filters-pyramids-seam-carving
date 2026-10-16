"""Filters package containing convolution, finite differences, and Gaussian/DoG filters."""

from .convolve import (
    PaddingMode,
    convolve2d_loop,
    convolve2d_vectorized,
    convolve2d_separable,
)
from .diff import (
    get_finite_difference_filters,
    compute_finite_differences,
    rescale_midgray,
)
from .gauss import (
    make_gaussian_kernel_2d,
    make_gaussian_kernel_1d,
    build_dog_filters,
    compute_dog_derivatives,
)

__all__ = [
    "PaddingMode",
    "convolve2d_loop",
    "convolve2d_vectorized",
    "convolve2d_separable",
    "get_finite_difference_filters",
    "compute_finite_differences",
    "rescale_midgray",
    "make_gaussian_kernel_2d",
    "make_gaussian_kernel_1d",
    "build_dog_filters",
    "compute_dog_derivatives",
]
