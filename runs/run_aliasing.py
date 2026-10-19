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

IMAGE_PATHS = {
    name: ROOT_DIR / "images" / "frequencies" / f"{name}.png"
    for name in ("brick", "rings")
}
OUTPUT_ROOT = ROOT_DIR / "outputs" / "frequencies" / "aliasing"
SHRINK_FACTOR = 4
SIGMA_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
SELECTED_SIGMA = 2.5


def subsample(image: np.ndarray, factor: int = SHRINK_FACTOR) -> np.ndarray:
    """Keep one pixel from each factor-by-factor block, starting at the origin."""
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    return image[::factor, ::factor]


def _save_comparison(
    direct: np.ndarray, filtered: np.ndarray, output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(direct, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Direct subsampling (every {SHRINK_FACTOR}th pixel)")
    axes[1].imshow(filtered, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Gaussian prefilter, sigma={SELECTED_SIGMA:g}, then subsample")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "aliasing_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_sigma_sweep(image: np.ndarray, name: str, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, sigma in zip(axes.flat, SIGMA_CANDIDATES):
        reduced = subsample(gaussian_blur(image, sigma=sigma))
        axis.imshow(reduced, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"sigma={sigma:g}")
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.suptitle(f"{name.title()} image after Gaussian prefiltering and subsampling")
    figure.tight_layout()
    figure.savefig(output_dir / "sigma_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for name, image_path in IMAGE_PATHS.items():
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=True)
        image = load_image(image_path, gray=True)
        direct = subsample(image)
        filtered = subsample(gaussian_blur(image, sigma=SELECTED_SIGMA))

        save_image(output_dir / "direct_subsampling.png", direct)
        save_image(output_dir / "gaussian_prefiltered.png", filtered)
        _save_comparison(direct, filtered, output_dir)
        _save_sigma_sweep(image, name, output_dir)

        print(f"{name.title()} input shape: {image.shape}; reduced shape: {direct.shape}")
        print(f"Gaussian prefilter sigma: {SELECTED_SIGMA:g}")
        print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()