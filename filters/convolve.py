"""2D convolution implementations: loop, vectorized, and separable."""

from typing import Literal, Tuple
import numpy as np

PaddingMode = Literal["constant", "reflect"]


def _get_padding(kh: int, kw: int) -> Tuple[int, int, int, int]:
    """Compute padding sizes (top, bottom, left, right) to match scipy mode='same'.

    For odd filter sizes, padding is symmetric: kh // 2 and kw // 2.
    For even sizes, it matches scipy's centering convention.
    """
    pad_bottom = (kh - 1) // 2
    pad_top = kh - 1 - pad_bottom
    pad_right = (kw - 1) // 2
    pad_left = kw - 1 - pad_right
    return pad_top, pad_bottom, pad_left, pad_right


def _prepare_kernel(kernel: np.ndarray) -> np.ndarray:
    """Validate and format kernel as a 2D float64 array."""
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim == 1:
        k = k.reshape(1, -1)
    if k.ndim != 2:
        raise ValueError(f"Kernel must be 1D or 2D, got shape {kernel.shape}")
    return k


def _prepare_image_and_padding(
    image: np.ndarray,
    kh: int,
    kw: int,
    padding_mode: PaddingMode = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image to float64 and pad boundaries to match scipy mode='same'.

    Returns:
        (im, im_padded): float64 input image and its padded array.
    """
    im = np.asarray(image, dtype=np.float64)
    if im.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got ndim={im.ndim}")

    pad_top, pad_bottom, pad_left, pad_right = _get_padding(kh, kw)
    pad_width: list[Tuple[int, int]] = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if im.ndim == 3:
        pad_width.append((0, 0))

    im_padded = np.pad(im, pad_width, mode=padding_mode)
    return im, im_padded


def convolve2d_loop(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: PaddingMode = "constant",
) -> np.ndarray:
    """2D convolution using explicit loops over pixels and filter entries.

    Convolving with a filter is correlating with the filter flipped along both axes.
    Pads the input with zeros (or specified mode) so output size matches input size.

    Args:
        image: Grayscale (H, W) or color (H, W, C) float array.
        kernel: 2D filter kernel (kh, kw), or 1D array.
        padding_mode: Padding mode (default 'constant' for zero-padding).

    Returns:
        Filtered image array of the same shape and dtype as input.
    """
    k = _prepare_kernel(kernel)
    kh, kw = k.shape
    k_flipped = k[::-1, ::-1]
    im, im_padded = _prepare_image_and_padding(image, kh, kw, padding_mode)

    if im.ndim == 2:
        H, W = im.shape
        output = np.zeros((H, W), dtype=np.float64)
        for r in range(H):
            for c in range(W):
                val = 0.0
                for i in range(kh):
                    for j in range(kw):
                        val += im_padded[r + i, c + j] * k_flipped[i, j]
                output[r, c] = val
        return output.astype(image.dtype)
    else:
        H, W, C = im.shape
        out = np.zeros((H, W, C), dtype=image.dtype)
        for ch in range(C):
            out[:, :, ch] = convolve2d_loop(
                image[:, :, ch], kernel, padding_mode=padding_mode
            )
        return out


def convolve2d_vectorized(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: PaddingMode = "constant",
) -> np.ndarray:
    """2D convolution vectorized with no loop over pixels.

    Performs only whole-image NumPy operations by shifting the padded image
    according to filter entry offsets and accumulating weighted image slices.

    Args:
        image: Grayscale (H, W) or color (H, W, C) float array.
        kernel: 2D filter kernel (kh, kw), or 1D array.
        padding_mode: Padding mode (default 'constant' for zero-padding).

    Returns:
        Filtered image array of the same shape and dtype as input.
    """
    k = _prepare_kernel(kernel)
    kh, kw = k.shape
    k_flipped = k[::-1, ::-1]
    im, im_padded = _prepare_image_and_padding(image, kh, kw, padding_mode)

    H, W = im.shape[:2]
    output = np.zeros_like(im)

    # Whole-image slice accumulation: zero pixel loops
    for i in range(kh):
        for j in range(kw):
            coeff = k_flipped[i, j]
            if coeff != 0.0:
                output += coeff * im_padded[i : i + H, j : j + W]

    return output.astype(image.dtype)


def convolve2d_separable(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: PaddingMode = "constant",
    tol: float = 1e-7,
) -> np.ndarray:
    """2D convolution exploiting filter separability via SVD rank-1 check.

    Tests whether the filter has rank 1 up to rounding error using SVD:
      K ~ S[0] * (u @ v.T)
    If rank 1, convolves with the column first, then with the row.
    Otherwise, falls back to convolve2d_vectorized.

    Args:
        image: Grayscale (H, W) or color (H, W, C) float array.
        kernel: 2D filter kernel (kh, kw), or 1D array.
        padding_mode: Padding mode (default 'constant').
        tol: Relative tolerance for singular value ratio S[1] / S[0].

    Returns:
        Filtered image array of the same shape and dtype as input.
    """
    k = _prepare_kernel(kernel)
    kh, kw = k.shape

    # Fast-path for 1D slices
    if kh == 1 or kw == 1:
        return convolve2d_vectorized(image, k, padding_mode=padding_mode)

    U, S, Vt = np.linalg.svd(k, full_matrices=False)

    is_separable = False
    if len(S) == 1:
        is_separable = True
    elif S[0] < 1e-12:
        return np.zeros_like(image)
    elif S[1] <= tol * S[0]:
        is_separable = True

    if is_separable:
        col = U[:, 0:1] * np.sqrt(S[0])
        row = Vt[0:1, :] * np.sqrt(S[0])

        col_filtered = convolve2d_vectorized(image, col, padding_mode=padding_mode)
        output = convolve2d_vectorized(col_filtered, row, padding_mode=padding_mode)
        return output
    else:
        return convolve2d_vectorized(image, k, padding_mode=padding_mode)
