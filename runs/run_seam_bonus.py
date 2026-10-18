"""Bonus: Object Removal + Seam Insertion (Sections 4.3 & 4.6)

Usage:
  - Put photo + mask pairs in images/bonus/
  - Mask must be named <photo_stem>_mask.png (or .jpg)
  - White / bright pixels = object to remove

Run:
  uv run python runs/run_seam_bonus.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_ROOT = ROOT_DIR / "outputs" / "bonus"


from filters import (
    compute_finite_differences,
    compute_dog_derivatives,
)
from utils.utils import load_image, save_image


# ------------------------------------------------------------------
# Energy functions
# ------------------------------------------------------------------
def energy_finite(image: np.ndarray) -> np.ndarray:
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    _, _, magnitude = compute_finite_differences(gray.astype(np.float64))
    return magnitude.astype(np.float64)


def energy_dog(image: np.ndarray, ksize: int = 13, sigma: float = 2.0) -> np.ndarray:
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    *_, magnitude = compute_dog_derivatives(
        gray.astype(np.float64), ksize=ksize, sigma=sigma, padding_mode="constant"
    )
    return magnitude


ENERGY_FUNCTIONS = {
    "finite": energy_finite,
    "dog": energy_dog,
}


# ------------------------------------------------------------------
# Seam finding / removal
# ------------------------------------------------------------------
def _get_seam_pixels(energy: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost vertical seam (row-vectorised DP). Returns [(r, c), ...]."""
    H, W = energy.shape
    M = energy.astype(np.float64).copy()
    backtrack = np.zeros((H, W), dtype=np.int32)
    cols = np.arange(W)
    offsets = np.array([0, -1, 1])  # same tie-break order as before: centre, left, right

    for r in range(1, H):
        prev = M[r - 1]
        left = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        stack = np.stack([prev, left, right])
        idx = np.argmin(stack, axis=0)
        M[r] += stack[idx, cols]
        backtrack[r] = cols + offsets[idx]

    c = int(np.argmin(M[-1]))
    seam = []
    for r in range(H - 1, -1, -1):
        seam.append((r, c))
        if r > 0:
            c = int(backtrack[r, c])
    seam.reverse()
    return seam


def _remove_seam(image: np.ndarray, seam: list[tuple[int, int]]) -> np.ndarray:
    H, W = image.shape[:2]
    keep = np.ones((H, W), dtype=bool)
    rows = np.fromiter((r for r, _ in seam), dtype=np.int64, count=len(seam))
    cols = np.fromiter((c for _, c in seam), dtype=np.int64, count=len(seam))
    keep[rows, cols] = False
    return image[keep].reshape((H, W - 1) + image.shape[2:])


def remove_object(image: np.ndarray, mask: np.ndarray, energy_fn) -> tuple[np.ndarray, int]:
    """Force seams through the mask until the object disappears. Returns (reduced, k)."""
    current = image.copy()
    current_mask = mask.astype(bool).copy()
    k = 0
    pbar = tqdm(desc="    Removing object", leave=False)
    while current_mask.any():
        energy = energy_fn(current).astype(np.float64).copy()
        energy[current_mask] = -1e6
        seam = _get_seam_pixels(energy)
        current = _remove_seam(current, seam)
        current_mask = _remove_seam(current_mask, seam)
        k += 1
        pbar.update(1)
        pbar.set_postfix(k=k, remaining=int(current_mask.sum()))
    pbar.close()
    return current, k


# ------------------------------------------------------------------
# Seam insertion
# ------------------------------------------------------------------
def find_insertion_seams(reduced: np.ndarray, k: int, energy_fn) -> list:
    """Find k seams on a COPY of the reduced photo, expressed in the reduced
    photo's own column coordinates (via a column-index map)."""
    H, W = reduced.shape[:2]
    copy = reduced.copy()
    colmap = np.tile(np.arange(W, dtype=np.int32), (H, 1))
    seams = []
    for _ in tqdm(range(k), desc="    Finding insertion seams", leave=False):
        seam = _get_seam_pixels(energy_fn(copy))
        seams.append([(r, int(colmap[r, c])) for r, c in seam])
        copy = _remove_seam(copy, seam)
        colmap = _remove_seam(colmap, seam)
    return seams


def insert_seams(image: np.ndarray, seams: list) -> tuple[np.ndarray, np.ndarray]:
    """Widen image by len(seams). Each seam pixel is written twice: first copy
    averaged with its left neighbour, second with its right neighbour. At the
    border the pixel itself stands in for the missing neighbour.

    Returns (widened image, boolean map of inserted pixels in the new image)."""
    H, W = image.shape[:2]
    k = len(seams)

    is_seam = np.zeros((H, W), dtype=bool)
    for seam in seams:
        for r, c in seam:
            is_seam[r, c] = True

    img = image.astype(np.float64)
    out = np.zeros((H, W + k) + image.shape[2:], dtype=np.float64)
    inserted = np.zeros((H, W + k), dtype=bool)

    for r in range(H):
        dst = 0
        for c in range(W):
            pix = img[r, c]
            if is_seam[r, c]:
                left = img[r, c - 1] if c > 0 else pix
                right = img[r, c + 1] if c < W - 1 else pix
                out[r, dst] = (pix + left) / 2.0
                inserted[r, dst] = True
                dst += 1
                out[r, dst] = (pix + right) / 2.0
                inserted[r, dst] = True
                dst += 1
            else:
                out[r, dst] = pix
                dst += 1
    return out.astype(image.dtype), inserted


def draw_inserted(image: np.ndarray, inserted: np.ndarray, color=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Paint inserted pixels in the given colour on a copy of the image."""
    base = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
    vis = base.astype(np.float32).copy()
    vis[inserted] = color
    return np.clip(vis, 0, 1)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
def run_bonus(image_path: Path, mask_path: Path, energy_name: str, energy_fn):
    base = OUTPUT_ROOT / energy_name
    folders = {
        "mask":     base / "mask_overlay",
        "removed":  base / "object_removed",
        "restored": base / "restored",
        "seams":    base / "restored_with_seams",
    }
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    mask = load_image(mask_path)
    if mask.ndim == 3:
        mask = mask.mean(axis=2)
    mask = mask > 0.5

    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Mask {mask.shape} and photo {image.shape[:2]} sizes differ for {image_path.name}"
        )

    print(f"  [{energy_name}] {image_path.name}")

    # 1. red mask overlay
    base_img = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
    overlay = base_img.astype(np.float32).copy()
    overlay[mask] = [1.0, 0.0, 0.0]
    save_image(folders["mask"] / f"{image_path.stem}_mask.png", overlay)

    # 2. remove object
    t0 = time.perf_counter()
    reduced, k = remove_object(image, mask, energy_fn)
    print(f"    Object removed with {k} seams")

    # 3. find k seams on a copy, in reduced-photo coordinates
    insertion_seams = find_insertion_seams(reduced, k, energy_fn)

    # 4. restore original width
    restored, inserted = insert_seams(reduced, insertion_seams)

    # 5. green visualisation of the inserted seams
    restored_seams_vis = draw_inserted(restored, inserted)

    elapsed = time.perf_counter() - t0

    save_image(folders["removed"] / f"{image_path.stem}_removed.png", reduced)
    save_image(folders["restored"] / f"{image_path.stem}_restored.png", restored)
    save_image(folders["seams"] / f"{image_path.stem}_restored_seams.png", restored_seams_vis)

    print(f"    -> outputs/bonus/{energy_name}/  [{elapsed:.2f} s]")


def main():
    images_dir = ROOT_DIR / "images" / "bonus"
    if not images_dir.exists():
        print(f"Create folder {images_dir} and put photo + mask pairs there.")
        print("Mask naming: <photo_stem>_mask.png")
        return

    candidates = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    pairs = []
    for p in candidates:
        if p.stem.endswith("_mask"):
            continue
        mask_p = images_dir / f"{p.stem}_mask.png"
        if not mask_p.exists():
            mask_p = images_dir / f"{p.stem}_mask.jpg"
        if mask_p.exists():
            pairs.append((p, mask_p))

    if not pairs:
        print("No photo+mask pairs found in images/bonus/")
        return

    total_t0 = time.perf_counter()
    for img_path, mask_path in pairs:
        print(f"\n=== {img_path.name} ===")
        for energy_name, energy_fn in ENERGY_FUNCTIONS.items():
            run_bonus(img_path, mask_path, energy_name, energy_fn)

    print(f"\nBonus done in {time.perf_counter() - total_t0:.1f} s")
    print(f"Results in: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()