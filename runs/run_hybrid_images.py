"""Generate hybrid-image deliverables for the frequency-domain assignment."""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from filters import build_hybrid_image, log_magnitude_spectrum, rescale_midgray
from utils.align_images import align_images
from utils.utils import load_image, save_image

IMAGE_DIR = ROOT_DIR / "images" / "frequencies"
OUTPUT_DIR = ROOT_DIR / "outputs" / "frequencies"


@dataclass(frozen=True)
class HybridPair:
    name: str
    low_image: str
    high_image: str
    sigma_low: float
    sigma_high: float
    low_points: tuple[tuple[float, float], tuple[float, float]]
    high_points: tuple[tuple[float, float], tuple[float, float]]
    note: str


PAIRS = (
    HybridPair(
        name="man_cat",
        low_image="hybrid_man.jpg",
        high_image="hybrid_cat.jpg",
        sigma_low=8.0,
        sigma_high=6.0,
        low_points=((299, 341), (443, 332)),
        high_points=((605, 290), (760, 372)),
        note="The aligned eyes support a recognizable face at both viewing scales.",
    ),
    HybridPair(
        name="apple_orange",
        low_image="apple.jpg",
        high_image="orange.jpg",
        sigma_low=10.0,
        sigma_high=7.0,
        low_points=((120, 256), (390, 256)),
        high_points=((120, 245), (390, 245)),
        note="Similar size and silhouette make the fruit pair a reasonable hybrid.",
    ),
    HybridPair(
        name="hand_eye",
        low_image="hand.jpg",
        high_image="eye.jpg",
        sigma_low=14.0,
        sigma_high=8.0,
        low_points=((250, 380), (610, 380)),
        high_points=((185, 230), (450, 230)),
        note=(
            "This is the intentionally weak pair: the eye's contours do not match "
            "the hand's palm and fingers, so its high-frequency detail reads as "
            "texture on a hand rather than a second coherent object."
        ),
    ),
)


def _show_image(axis, image: np.ndarray, title: str) -> None:
    axis.imshow(np.clip(image, 0.0, 1.0))
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def _save_spatial_process(
    pair: HybridPair,
    low_input: np.ndarray,
    high_input: np.ndarray,
    high_aligned: np.ndarray,
    low_pass: np.ndarray,
    high_pass: np.ndarray,
    hybrid: np.ndarray,
    pair_output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 6, figsize=(18, 4))
    items = (
        (low_input, f"Low-frequency input\n{pair.low_image}"),
        (high_input, f"High-frequency input\n{pair.high_image}"),
        (high_aligned, "Aligned high input"),
        (low_pass, f"Low-pass, sigma={pair.sigma_low:g}"),
        (rescale_midgray(high_pass), f"High-pass, sigma={pair.sigma_high:g}"),
        (hybrid, "Hybrid"),
    )
    for axis, (image, title) in zip(axes, items):
        _show_image(axis, image, title)
    figure.tight_layout()
    figure.savefig(pair_output_dir / "spatial_process.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_fourier_process(
    pair_output_dir: Path,
    low_input: np.ndarray,
    high_aligned: np.ndarray,
    low_pass: np.ndarray,
    high_pass: np.ndarray,
    hybrid: np.ndarray,
) -> None:
    images = (low_input, high_aligned, low_pass, high_pass, hybrid)
    titles = (
        "Low-frequency input",
        "Aligned high-frequency input",
        "Low-pass image",
        "High-pass image",
        "Hybrid image",
    )
    figure, axes = plt.subplots(1, len(images), figsize=(17, 4))
    for axis, image, title in zip(axes, images, titles):
        spectrum = log_magnitude_spectrum(image)
        axis.imshow(spectrum, cmap="magma")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle("Centered log magnitude of grayscale Fourier transforms")
    figure.tight_layout()
    figure.savefig(pair_output_dir / "fourier_process.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_pair(pair: HybridPair, include_fourier: bool = False) -> None:
    low_input = load_image(IMAGE_DIR / pair.low_image)
    high_input = load_image(IMAGE_DIR / pair.high_image)
    high_aligned = align_images(
        low_input,
        high_input,
        pair.low_points,
        pair.high_points,
    )
    low_pass, high_pass, hybrid = build_hybrid_image(
        low_input,
        high_aligned,
        pair.sigma_low,
        pair.sigma_high,
    )

    pair_output_dir = OUTPUT_DIR / pair.name
    pair_output_dir.mkdir(parents=True, exist_ok=True)
    save_image(pair_output_dir / "aligned_high_input.png", high_aligned)
    save_image(pair_output_dir / "low_pass.png", low_pass)
    save_image(pair_output_dir / "high_pass.png", rescale_midgray(high_pass))
    save_image(pair_output_dir / "hybrid.png", hybrid)
    _save_spatial_process(
        pair,
        low_input,
        high_input,
        high_aligned,
        low_pass,
        high_pass,
        hybrid,
        pair_output_dir,
    )

    if include_fourier:
        _save_fourier_process(
            pair_output_dir, low_input, high_aligned, low_pass, high_pass, hybrid
        )

    print(
        f"{pair.name}: sigma_low={pair.sigma_low:g}, "
        f"sigma_high={pair.sigma_high:g} -> {pair_output_dir}"
    )


def _write_report() -> None:
    lines = [
        "# Hybrid images",
        "",
        "Each hybrid is the low-pass of the first image plus the high-pass of the aligned second image.",
        "Alignment landmarks use (x, y) pixel coordinates in the original images.",
        "",
        "| Pair | Low-pass image | High-pass image | Sigma low | Sigma high | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for pair in PAIRS:
        lines.append(
            f"| {pair.name} | {pair.low_image} | {pair.high_image} | "
            f"{pair.sigma_low:g} | {pair.sigma_high:g} | {pair.note} |"
        )
    lines.extend(
        [
            "",
            "The man/cat pair also includes `fourier_process.png`, showing grayscale "
            "log-magnitude spectra for both aligned inputs, both filtered components, "
            "and their hybrid.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pair in PAIRS:
        run_pair(pair, include_fourier=pair.name == "man_cat")
    _write_report()
    print(f"Hybrid-image report saved to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()