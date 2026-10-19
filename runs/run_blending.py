"""Generate aligned multiresolution image blends and pyramid visualizations."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from filters import multiresolution_blend
from utils.align_images import align_images
from utils.utils import load_image, save_image

IMAGE_DIR = ROOT_DIR / "images" / "frequencies"
OUTPUT_DIR = ROOT_DIR / "outputs" / "frequencies" / "blending"


def _align_pair(
    base: np.ndarray,
    insert: np.ndarray,
    base_points: tuple[tuple[int, int], tuple[int, int]],
    insert_points: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    return align_images(base, insert, base_points, insert_points)


def _oraple_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, : width // 2] = 1.0
    return mask


def _cat_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.ones((height, width), dtype=np.float32)
    cv2.ellipse(mask, (width // 2, height // 2 + 12), (154, 190), -8, 0, 360, 0.0, -1)
    return mask


def _man_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.ones((height, width), dtype=np.float32)
    face_outline = np.array(
        [
            (width // 2, 66),
            (width // 2 + 112, 86),
            (width // 2 + 150, 164),
            (width // 2 + 132, 305),
            (width // 2 + 78, 402),
            (width // 2 - 12, 438),
            (width // 2 - 104, 392),
            (width // 2 - 146, 278),
            (width // 2 - 132, 150),
            (width // 2 - 70, 82),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [face_outline], 0.0)
    return mask


def _hard_cut(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask[..., None] * image1 + (1.0 - mask[..., None]) * image2


def _save_overview(
    name: str,
    image1: np.ndarray,
    aligned_image2: np.ndarray,
    mask: np.ndarray,
    hard_cut: np.ndarray,
    blend: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(16, 4))
    images = (image1, aligned_image2, mask, hard_cut, blend)
    titles = ("Image 1", "Aligned image 2", "Sharp mask", "Hard cut", "Multiresolution blend")
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(name.replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / name / "overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _boost_detail(image: np.ndarray) -> np.ndarray:
    maximum = float(np.max(np.abs(image)))
    if maximum < 1e-15:
        return np.full_like(image, 0.5)
    return np.clip(0.5 + 0.5 * image / maximum, 0.0, 1.0)


def _save_level_breakdown(
    weighted1: list[np.ndarray],
    weighted2: list[np.ndarray],
    combined: list[np.ndarray],
) -> None:
    levels = len(combined)
    figure, axes = plt.subplots(3, levels, figsize=(3.2 * levels, 8))
    row_data = (weighted1, weighted2, combined)
    row_titles = ("M × L1", "(1 − M) × L2", "Sum")
    for row, (images, label) in enumerate(zip(row_data, row_titles)):
        for level, image in enumerate(images):
            axes[row, level].imshow(_boost_detail(image))
            axes[row, level].set_title(f"Level {level}\n{image.shape[1]}x{image.shape[0]}")
            axes[row, level].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=11)
    figure.suptitle("Oraple: blended Laplacian bands (contrast boosted)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "oraple" / "level_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _process_pair(
    name: str,
    base_name: str,
    insert_name: str,
    base_points: tuple[tuple[int, int], tuple[int, int]],
    insert_points: tuple[tuple[int, int], tuple[int, int]],
    make_mask,
    levels: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    image1 = load_image(IMAGE_DIR / base_name)
    insert = load_image(IMAGE_DIR / insert_name)
    aligned_image2 = _align_pair(image1, insert, base_points, insert_points)
    mask = make_mask(image1.shape[:2])
    blend, weighted1, weighted2, combined = multiresolution_blend(
        image1, aligned_image2, mask, levels=levels
    )
    hard_cut = _hard_cut(image1, aligned_image2, mask)

    pair_output = OUTPUT_DIR / name
    pair_output.mkdir(parents=True, exist_ok=True)
    save_image(pair_output / "image1.png", image1)
    save_image(pair_output / "aligned_image2.png", aligned_image2)
    save_image(pair_output / "mask.png", mask)
    save_image(pair_output / "hard_cut.png", hard_cut)
    save_image(pair_output / "blend.png", blend)
    _save_overview(name, image1, aligned_image2, mask, hard_cut, blend)
    print(f"{name}: {base_name} + {insert_name}, levels={levels}")

    return weighted1, weighted2, combined


def _write_report() -> None:
    report = """# Multiresolution blending

All masks are drawn in code, kept binary/sharp at full resolution, and softened only by their Gaussian pyramids. Each blend uses `M*L1 + (1-M)*L2` at each Laplacian level, then reconstructs the combined pyramid.

| Blend | Base (white mask) | Insert (black mask) | Mask construction | Levels |
| --- | --- | --- | --- | ---: |
| Oraple | apple.jpg | orange.jpg | Left half white, right half black | 5 |
| Cat on apple | apple.jpg | hybrid_cat.jpg | Black ellipse over the aligned cat face | 5 |
| Man on orange | orange.jpg | hybrid_man.jpg | Black polygon around the aligned face | 4 |

The fruit pair uses top/bottom landmarks on each fruit; face overlays align the two eye centers to positions on the fruit. Each overview shows the base, aligned insert, mask, sharp hard cut, and multiresolution result. Oraple's `level_breakdown.png` shows both weighted Laplacian contributions and their sum at every level.

The hard cut applies the full-resolution binary mask identically to every frequency, so the discontinuity is visible across the boundary. The mask's Gaussian pyramid has a narrow transition at fine scales and progressively wider transitions at coarse scales. Consequently the blend keeps fine details local while changing broad color and illumination gradually across the seam.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oraple_parts = _process_pair(
        "oraple",
        "apple.jpg",
        "orange.jpg",
        ((255, 70), (255, 490)),
        ((255, 60), (255, 490)),
        _oraple_mask,
        levels=5,
    )
    _save_level_breakdown(*oraple_parts)

    _process_pair(
        "cat_on_apple",
        "apple.jpg",
        "hybrid_cat.jpg",
        ((180, 190), (305, 190)),
        ((605, 290), (760, 372)),
        _cat_mask,
        levels=5,
    )
    _process_pair(
        "man_on_orange",
        "orange.jpg",
        "hybrid_man.jpg",
        ((175, 190), (325, 180)),
        ((299, 341), (443, 332)),
        _man_mask,
        levels=4,
    )

    _write_report()
    print(f"Blending outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()