"""Unit tests and numerical sanity checks for Filters and Edges."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path so filters package can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from scipy.signal import convolve2d
from filters import (
    convolve2d_loop,
    convolve2d_vectorized,
    convolve2d_separable,
    make_gaussian_kernel_2d,
    get_finite_difference_filters,
    build_dog_filters,
    compute_finite_differences,
    compute_dog_derivatives,
    build_hybrid_image,
    log_magnitude_spectrum,
)


def test_convolution_sanity_random_filters():
    """Sanity check: All three versions must match scipy.signal.convolve2d

    in mode='same' to rounding error for random odd-sized filters,
    both separable and not.
    """
    np.random.seed(42)
    test_cases = [
        (7, 9, 3, 3),
        (8, 8, 3, 5),
        (10, 12, 5, 3),
        (11, 11, 5, 5),
        (12, 10, 1, 3),
        (9, 13, 3, 1),
        (15, 15, 7, 7),
    ]

    for H, W, kh, kw in test_cases:
        im = np.random.randn(H, W)

        # 1. Non-separable random filter
        k_nonsep = np.random.randn(kh, kw)
        target_nonsep = convolve2d(im, k_nonsep, mode="same", boundary="fill", fillvalue=0)

        out_loop = convolve2d_loop(im, k_nonsep)
        out_vec = convolve2d_vectorized(im, k_nonsep)
        out_sep = convolve2d_separable(im, k_nonsep)

        assert np.allclose(out_loop, target_nonsep, atol=1e-12), (
            f"Loop failed on non-separable {kh}x{kw}, diff={np.max(np.abs(out_loop - target_nonsep))}"
        )
        assert np.allclose(out_vec, target_nonsep, atol=1e-12), (
            f"Vectorized failed on non-separable {kh}x{kw}, diff={np.max(np.abs(out_vec - target_nonsep))}"
        )
        assert np.allclose(out_sep, target_nonsep, atol=1e-12), (
            f"Separable fallback failed on non-separable {kh}x{kw}, diff={np.max(np.abs(out_sep - target_nonsep))}"
        )

        # 2. Separable random filter (outer product of col and row)
        col = np.random.randn(kh, 1)
        row = np.random.randn(1, kw)
        k_sep = col @ row
        target_sep = convolve2d(im, k_sep, mode="same", boundary="fill", fillvalue=0)

        out_loop_s = convolve2d_loop(im, k_sep)
        out_vec_s = convolve2d_vectorized(im, k_sep)
        out_sep_s = convolve2d_separable(im, k_sep)

        assert np.allclose(out_loop_s, target_sep, atol=1e-12), (
            f"Loop failed on separable {kh}x{kw}, diff={np.max(np.abs(out_loop_s - target_sep))}"
        )
        assert np.allclose(out_vec_s, target_sep, atol=1e-12), (
            f"Vectorized failed on separable {kh}x{kw}, diff={np.max(np.abs(out_vec_s - target_sep))}"
        )
        assert np.allclose(out_sep_s, target_sep, atol=1e-12), (
            f"Separable failed on separable {kh}x{kw}, diff={np.max(np.abs(out_sep_s - target_sep))}"
        )


def test_gaussian_and_box_filters_are_separable():
    """Verify that Gaussian and Box filters are detected as rank-1 and convolved accurately."""
    im = np.random.randn(16, 16)

    # Gaussian
    G = make_gaussian_kernel_2d(ksize=7, sigma=1.5)
    target_g = convolve2d(im, G, mode="same", boundary="fill", fillvalue=0)
    out_g = convolve2d_separable(im, G)
    assert np.allclose(out_g, target_g, atol=1e-12)

    # Box
    box = np.ones((5, 5)) / 25.0
    target_b = convolve2d(im, box, mode="same", boundary="fill", fillvalue=0)
    out_b = convolve2d_separable(im, box)
    assert np.allclose(out_b, target_b, atol=1e-12)


def test_color_image_support():
    """Verify that convolution works across RGB channels."""
    im_color = np.random.randn(10, 10, 3)
    k = make_gaussian_kernel_2d(ksize=3, sigma=1.0)

    out = convolve2d_vectorized(im_color, k)
    assert out.shape == (10, 10, 3)
    for c in range(3):
        target = convolve2d(im_color[:, :, c], k, mode="same", boundary="fill", fillvalue=0)
        assert np.allclose(out[:, :, c], target, atol=1e-12)


def test_finite_differences():
    """Verify finite difference operators Dx and Dy."""
    Dx, Dy = get_finite_difference_filters()
    assert Dx.shape == (1, 3)
    assert Dy.shape == (3, 1)

    ramp = np.arange(10, dtype=float).reshape(1, 10).repeat(10, axis=0)
    Ix, Iy, mag = compute_finite_differences(ramp)

    assert np.allclose(Ix[2:-2, 2:-2], 2.0)
    assert np.allclose(Iy[2:-2, 2:-2], 0.0)
    assert np.allclose(mag[2:-2, 2:-2], 2.0)


def test_hybrid_image_components_and_spectrum():
    """A constant image has no high-frequency component and a finite spectrum."""
    image = np.full((32, 32, 3), 0.4, dtype=np.float32)
    low_pass, high_pass, hybrid = build_hybrid_image(image, image, 2.0, 2.0)

    assert np.allclose(low_pass, image, atol=1e-7)
    assert np.allclose(high_pass, 0.0, atol=1e-7)
    assert np.allclose(hybrid, image, atol=1e-7)
    assert np.isfinite(log_magnitude_spectrum(hybrid)).all()


def run_all_tests():
    test_convolution_sanity_random_filters()
    test_gaussian_and_box_filters_are_separable()
    test_color_image_support()
    test_finite_differences()
    test_hybrid_image_components_and_spectrum()
    print("All unit tests for filters passed successfully!")


if __name__ == "__main__":
    run_all_tests()
