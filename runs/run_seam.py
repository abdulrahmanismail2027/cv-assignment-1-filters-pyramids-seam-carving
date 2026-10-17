"""Execution script for Seam Carving (vertical + horizontal).

Supports two energy functions:
  - finite  : gradient magnitude from finite differences (1.2)
  - dog     : gradient magnitude from Derivative of Gaussian (1.3)

For each direction (vertical / horizontal) and each energy it saves:
  - energy map
  - original with all removed seams in red
  - final carved image
  - labeled comparison: Original | Resized | Cropped | Seam-carved
"""
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_ROOT = ROOT_DIR / "outputs"
REDUCED_WIDTH = 0.70   # keep 70% → remove ~30%

from filters import (
    compute_finite_differences,
    rescale_midgray,
    build_dog_filters,
    convolve2d_separable,
)
from utils.utils import load_image, save_image


# ----------------------------------------------------------------------
# Energy functions
# ----------------------------------------------------------------------
def energy_finite(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    _, _, magnitude = compute_finite_differences(gray)
    return magnitude.astype(np.float64)


def energy_dog(image: np.ndarray, ksize: int = 13, sigma: float = 2.0) -> np.ndarray:
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    DoG_x, DoG_y = build_dog_filters(ksize=ksize, sigma=sigma, padding_mode="constant")
    Ix = convolve2d_separable(gray, DoG_x, padding_mode="constant")
    Iy = convolve2d_separable(gray, DoG_y, padding_mode="constant")
    return np.sqrt(Ix**2 + Iy**2).astype(np.float64)


ENERGY_FUNCTIONS = {
    "finite": energy_finite,
    "dog": energy_dog,
}


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------
def _get_seam_pixels(energy: np.ndarray) -> list[tuple[int, int]]:
    """Vertical seam (one pixel per row)."""
    H, W = energy.shape
    M = energy.astype(np.float64).copy()
    backtrack = np.zeros((H, W), dtype=np.int32)

    for r in range(1, H):
        for c in range(W):
            candidates = [(M[r-1, c], c)]
            if c > 0:
                candidates.append((M[r-1, c-1], c-1))
            if c < W-1:
                candidates.append((M[r-1, c+1], c+1))
            min_val, prev_c = min(candidates, key=lambda x: x[0])
            M[r, c] += min_val
            backtrack[r, c] = prev_c

    c = int(np.argmin(M[-1]))
    seam = []
    for r in range(H-1, -1, -1):
        seam.append((r, c))
        if r > 0:
            c = backtrack[r, c]
    seam.reverse()
    return seam


def _remove_seam(image: np.ndarray, seam: list[tuple[int, int]]) -> np.ndarray:
    H, W = image.shape[:2]
    if image.ndim == 2:
        out = np.zeros((H, W-1), dtype=image.dtype)
        for r, c in seam:
            out[r, :c] = image[r, :c]
            out[r, c:] = image[r, c+1:]
    else:
        out = np.zeros((H, W-1, image.shape[2]), dtype=image.dtype)
        for r, c in seam:
            out[r, :c, :] = image[r, :c, :]
            out[r, c:, :] = image[r, c+1:, :]
    return out


def _resize(image: np.ndarray, target_width: int) -> np.ndarray:
    H, W = image.shape[:2]
    target_height = int(H * (target_width / W))
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(pil).astype(np.float32) / 255.0


def _center_crop(image: np.ndarray, target_width: int) -> np.ndarray:
    W = image.shape[1]
    start = (W - target_width) // 2
    return image[:, start:start+target_width]


def _build_red_seams_visualization(image: np.ndarray, all_seams: list) -> np.ndarray:
    if image.ndim == 2:
        vis = np.stack([image]*3, axis=-1).astype(np.float32)
    else:
        vis = image.astype(np.float32).copy()

    H = image.shape[0]
    deleted = [[] for _ in range(H)]

    for seam in all_seams:
        for r, c in seam:
            orig_c = c
            for d in deleted[r]:
                if d <= orig_c:
                    orig_c += 1
                else:
                    break
            vis[r, orig_c] = [1.0, 0.0, 0.0]
            deleted[r].append(orig_c)
            deleted[r].sort()
    return np.clip(vis, 0, 1)


def _save_labeled_comparison(original, resized, cropped, carved, out_path: Path, title_suffix: str):
    def ensure_color(img):
        if img.ndim == 2:
            return np.stack([img]*3, axis=-1)
        return img

    imgs = [ensure_color(im) for im in (original, resized, cropped, carved)]
    titles = ["Original", "Resized", "Cropped", f"Seam carved\n({title_suffix})"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _transpose_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.T
    else:
        return np.transpose(image, (1, 0, 2))


# ----------------------------------------------------------------------
# Main worker
# ----------------------------------------------------------------------
def run_one(
    image: np.ndarray,
    image_name: str,
    direction: str,          # "vertical" or "horizontal"
    energy_name: str,
    energy_fn,
    new_width_ratio: float = REDUCED_WIDTH,
):
    # prepare folders
    base = OUTPUT_ROOT / direction / energy_name
    paths = {
        "energy":  base / "energy_maps",
        "seams":   base / "seams",
        "carved":  base / "carved",
        "compare": base / "compare",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # for horizontal we work on the transposed image
    working = image if direction == "vertical" else _transpose_image(image)

    original_width = working.shape[1]
    target_width = int(original_width * new_width_ratio)
    num_seams = original_width - target_width

    print(f"  [{direction:9s} | {energy_name:6s}] {image_name}: "
          f"{original_width} → {target_width} ({num_seams} seams)")

    # energy map (of the working orientation)
    energy = energy_fn(working)
    energy_vis = rescale_midgray(energy)
    save_image(paths["energy"] / f"{image_name}_energy.png", energy_vis)

    # remove seams
    all_seams = []
    current = working.copy()
    t0 = time.perf_counter()

    for _ in tqdm(range(num_seams), desc=f"    {direction[:1]}-{energy_name}", leave=False):
        e = energy_fn(current)
        seam = _get_seam_pixels(e)
        all_seams.append(seam)
        current = _remove_seam(current, seam)

    elapsed = time.perf_counter() - t0

    # red seams on the working image
    vis = _build_red_seams_visualization(working, all_seams)

    # baselines
    resized = _resize(working, target_width)
    cropped = _center_crop(working, target_width)

    # if horizontal, transpose results back so they look normal
    if direction == "horizontal":
        current = _transpose_image(current)
        vis     = _transpose_image(vis)
        resized = _transpose_image(resized)
        cropped = _transpose_image(cropped)
        # original stays as-is for the comparison
        orig_for_compare = image
    else:
        orig_for_compare = image

    # save
    save_image(paths["seams"]  / f"{image_name}_seams_red.png", vis)
    save_image(paths["carved"] / f"{image_name}_carved.png", current)

    _save_labeled_comparison(
        orig_for_compare, resized, cropped, current,
        paths["compare"] / f"{image_name}_compare.png",
        f"{direction}, {energy_name}"
    )

    print(f"    → outputs/{direction}/{energy_name}/  [{elapsed:.2f} s]")


def main():
    images_path = ROOT_DIR / "images" / "seams"
    image_list = sorted(list(images_path.glob("*.png")) + list(images_path.glob("*.jpg")))

    if not image_list:
        print("No images found in images/seams/")
        return

    total_t0 = time.perf_counter()

    for img_path in image_list:
        image = load_image(img_path)
        name = img_path.stem
        print(f"\n=== {img_path.name} ===")

        for direction in ("vertical", "horizontal"):
            for energy_name, energy_fn in ENERGY_FUNCTIONS.items():
                run_one(image, name, direction, energy_name, energy_fn)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\nAll done in {total_elapsed:.1f} s")
    print(f"Results are under: {OUTPUT_ROOT}/vertical/  and  {OUTPUT_ROOT}/horizontal/")


if __name__ == "__main__":
    main()