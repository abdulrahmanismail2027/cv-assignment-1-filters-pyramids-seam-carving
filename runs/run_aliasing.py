"""Compare direct subsampling with Gaussian-prefiltered subsampling."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from filters import gaussian_blur
from utils.utils import load_image, save_image

IMAGE_PATH = ROOT_DIR / "images" / "frequencies" / "rings.png"
OUTPUT_DIR = ROOT_DIR / "outputs" / "frequencies" / "aliasing"
SHRINK_FACTOR = 4
SIGMA_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
SELECTED_SIGMA = 2.5


def subsample(image: np.ndarray, factor: int = SHRINK_FACTOR) -> np.ndarray:
    """Keep one pixel from each factor-by-factor block, starting at the origin."""
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    return image[::factor, ::factor]


def _save_comparison(direct: np.ndarray, filtered: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(direct, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Direct subsampling (every 4th pixel)")
    axes[1].imshow(filtered, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Gaussian prefilter, sigma={SELECTED_SIGMA:g}, then subsample")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "aliasing_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_sigma_sweep(image: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, sigma in zip(axes.flat, SIGMA_CANDIDATES):
        reduced = subsample(gaussian_blur(image, sigma=sigma))
        axis.imshow(reduced, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"sigma={sigma:g}")
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.suptitle("Rings after Gaussian prefiltering and quarter-size subsampling")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "sigma_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_report() -> None:
    report = f"""# Aliasing experiment

Input: `images/frequencies/rings.png` ({SHRINK_FACTOR}x reduction per dimension).

The direct result keeps every fourth row and column, with no prefilter. Its fine rings exceed the sampling grid's Nyquist limit, so different source patterns collapse to the same samples and appear as broad, false circles. The filtered result first applies a Gaussian with sigma {SELECTED_SIGMA:g} pixels, then takes the same samples; this attenuates those high spatial frequencies and prevents the visible false-circle pattern, at the cost of some edge sharpness.

Sigma {SELECTED_SIGMA:g} is the smallest setting in the tested sweep ({', '.join(f'{sigma:g}' for sigma in SIGMA_CANDIDATES)}) where false circles were no longer visually apparent. See `sigma_sweep.png` for the comparison and `aliasing_comparison.png` for the requested two-result figure.

A Gaussian pyramid should therefore low-pass each level before decimation. Blurring after reducing the image cannot recover information already aliased into lower frequencies.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_image(IMAGE_PATH, gray=True)
    direct = subsample(image)
    filtered = subsample(gaussian_blur(image, sigma=SELECTED_SIGMA))

    save_image(OUTPUT_DIR / "direct_subsampling.png", direct)
    save_image(OUTPUT_DIR / "gaussian_prefiltered.png", filtered)
    _save_comparison(direct, filtered)
    _save_sigma_sweep(image)
    _write_report()

    print(f"Input shape: {image.shape}; reduced shape: {direct.shape}")
    print(f"Selected smallest visually clean sigma: {SELECTED_SIGMA:g}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()