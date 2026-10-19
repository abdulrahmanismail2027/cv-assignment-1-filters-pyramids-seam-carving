"""Build, visualize, and verify Gaussian and Laplacian pyramids."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from filters import gaussian_pyramid, laplacian_pyramid, reconstruct_laplacian
from utils.utils import load_image, save_image

IMAGE_PATH = ROOT_DIR / "images" / "filters" / "facade.png"
OUTPUT_DIR = ROOT_DIR / "outputs" / "frequencies" / "pyramids"
LEVELS = 5


def _boost_laplacian(image: np.ndarray) -> np.ndarray:
    maximum = float(np.max(np.abs(image)))
    if maximum < 1e-15:
        return np.full_like(image, 0.5)
    return np.clip(0.5 + 0.5 * image / maximum, 0.0, 1.0)


def _save_pyramid_figure(
    gaussian: list[np.ndarray], laplacian: list[np.ndarray]
) -> None:
    figure, axes = plt.subplots(2, len(gaussian), figsize=(3.2 * len(gaussian), 7))
    for index, (gaussian_level, laplacian_level) in enumerate(zip(gaussian, laplacian)):
        axes[0, index].imshow(np.clip(gaussian_level, 0.0, 1.0))
        axes[0, index].set_title(
            f"G{index}: {gaussian_level.shape[1]}x{gaussian_level.shape[0]}"
        )
        axes[0, index].axis("off")
        axes[1, index].imshow(_boost_laplacian(laplacian_level))
        axes[1, index].set_title(
            f"L{index}: {laplacian_level.shape[1]}x{laplacian_level.shape[0]}"
        )
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("Gaussian", fontsize=11)
    axes[1, 0].set_ylabel("Laplacian\n(contrast boosted)", fontsize=11)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "pyramids.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_pyramid_layout(
    pyramid: list[np.ndarray], filename: str, contrast_boost: bool = False
) -> None:
    gap = 8
    canvas_shape = (
        sum(level.shape[0] for level in pyramid) + gap * (len(pyramid) - 1),
        pyramid[0].shape[1],
    ) + pyramid[0].shape[2:]
    canvas = np.ones(canvas_shape, dtype=np.float64)
    top = 0
    for level in reversed(pyramid):
        height, width = level.shape[:2]
        left = (canvas.shape[1] - width) // 2
        display_level = _boost_laplacian(level) if contrast_boost else np.clip(level, 0.0, 1.0)
        canvas[top : top + height, left : left + width, ...] = display_level
        top += height + gap
    save_image(OUTPUT_DIR / filename, canvas)


def _save_reconstruction_figure(
    original: np.ndarray,
    reconstructed: np.ndarray,
    absolute_error: np.ndarray,
) -> None:
    error_scale = float(np.max(absolute_error))
    error_display = (
        absolute_error / error_scale if error_scale > 0 else np.zeros_like(absolute_error)
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(
        axes,
        (original, reconstructed, error_display),
        ("Original", "Rebuilt from Laplacian pyramid", "Absolute error (normalized)"),
    ):
        axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "reconstruction.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    gaussian: list[np.ndarray],
    original: np.ndarray,
    max_error: float,
    rmse: float,
) -> float:
    level_pixels = [level.shape[0] * level.shape[1] for level in gaussian]
    ratio = sum(level_pixels) / level_pixels[0]
    ideal_finite_ratio = (4.0 / 3.0) * (1.0 - 4.0 ** (-len(gaussian)))
    sizes = ", ".join(f"{level.shape[1]}x{level.shape[0]}" for level in gaussian)
    report = f"""# Gaussian and Laplacian pyramids

Input: `images/filters/facade.png`, shape {original.shape[1]}x{original.shape[0]} (not a power-of-two image).

Gaussian level dimensions: {sizes}. Each reduction applies the separable kernel `([1, 4, 6, 4, 1] / 16)` horizontally and vertically, then samples every second row and column. Odd-sized expanded images are cropped on the bottom/right to match the corresponding Gaussian level.

The Laplacian levels are `L_i = G_i - expand(G_(i+1))`; the coarsest level is the last Gaussian image. Reconstruction uses the same expansion and adds each residual, so each step cancels algebraically: `expand(G_(i+1)) + (G_i - expand(G_(i+1))) = G_i`. The observed maximum absolute error is {max_error:.3e}; RMSE is {rmse:.3e}, from floating-point rounding.

Lecture 3's residual `G_i - blur(G_i)` is generally not an exact reconstruction residual: `blur(G_i)` is not necessarily the same image as the expanded next pyramid level after filtering and subsampling. Adding it back therefore does not restore `G_i` exactly.

This {len(gaussian)}-level Gaussian pyramid contains {sum(level_pixels):,} pixels, {ratio:.4f} times the input's {level_pixels[0]:,} pixels. Each level has approximately one quarter as many pixels as the previous one, giving `1 + 1/4 + 1/16 + ...`, whose infinite sum is `1 / (1 - 1/4) = 4/3`. For {len(gaussian)} ideal levels the ratio would be {ideal_finite_ratio:.4f}; ceil rounding at odd dimensions explains the small difference in the actual ratio.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    return ratio


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = load_image(IMAGE_PATH)
    gaussian = gaussian_pyramid(original, levels=LEVELS)
    laplacian = laplacian_pyramid(original, levels=LEVELS)
    reconstructed = reconstruct_laplacian(laplacian)
    absolute_error = np.abs(original.astype(np.float64) - reconstructed)
    max_error = float(np.max(absolute_error))
    rmse = float(np.sqrt(np.mean(absolute_error**2)))

    _save_pyramid_figure(gaussian, laplacian)
    _save_pyramid_layout(gaussian, "gaussian_pyramid_layout.png")
    _save_pyramid_layout(
        laplacian, "laplacian_pyramid_layout.png", contrast_boost=True
    )
    _save_reconstruction_figure(original, reconstructed, absolute_error)
    save_image(OUTPUT_DIR / "reconstructed.png", reconstructed)
    ratio = _write_report(gaussian, original, max_error, rmse)

    print(f"Gaussian dimensions: {[level.shape[:2] for level in gaussian]}")
    print(f"Gaussian size ratio: {ratio:.4f}x the input")
    print(f"Reconstruction max error: {max_error:.3e}; RMSE: {rmse:.3e}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()