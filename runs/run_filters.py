"""Execution script for Filters and Edges deliverables.

Runs:
1.1 Convolution from scratch:
    - Sanity check against scipy.signal.convolve2d (mode='same').
    - Timing benchmark across filter sizes 3x3 to 31x31 on 256x256 image.
    - Side-by-side timing table and comparative plot.
1.2 Finite Difference Operator:
    - Grayscale filtering with Dx = [1 0 -1] and Dy = Dx.T.
    - Gradient magnitude and thresholded edge map.
    - Visualization of the 4 maps.
1.3 Derivative of Gaussian (DoG) Filter:
    - Folded Gaussian blur (sigma=2) into Dx and Dy.
    - Single-convolution filtering.
    - Visualization of DoG filters, gradient magnitude, and edges.
    - Comparative visualization between 1.2 and 1.3.
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from filters import (
    convolve2d_loop,
    convolve2d_vectorized,
    convolve2d_separable,
    make_gaussian_kernel_2d,
    build_dog_filters,
    compute_finite_differences,
    rescale_midgray,
)
from utils.utils import load_image, save_image

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILTERS_DIR = ROOT_DIR / "outputs" / "filters"
OUTPUT_CONVOLVE_DIR = OUTPUT_FILTERS_DIR / "convolve"
OUTPUT_DIFF_DIR = OUTPUT_FILTERS_DIR / "diff"
OUTPUT_GAUSS_DIR = OUTPUT_FILTERS_DIR / "gauss"

OUTPUT_CONVOLVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIFF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GAUSS_DIR.mkdir(parents=True, exist_ok=True)


def run_section_1_1():
    print("=" * 80)
    print("SECTION 1.1: CONVOLUTION FROM SCRATCH")
    print("=" * 80)

    # 1. Sanity check
    print("\n--- 1. Sanity Check against scipy.signal.convolve2d (mode='same') ---")
    np.random.seed(42)
    shapes = [(11, 13, 3, 3), (12, 10, 5, 3), (15, 15, 7, 7)]
    def _max_conv_discrepancy(image: np.ndarray, kernel: np.ndarray) -> float:
        target = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
        return max(
            float(np.max(np.abs(convolve2d_loop(image, kernel) - target))),
            float(np.max(np.abs(convolve2d_vectorized(image, kernel) - target))),
            float(np.max(np.abs(convolve2d_separable(image, kernel) - target))),
        )

    all_passed = True
    for H, W, kh, kw in shapes:
        im = np.random.randn(H, W)
        k_ns = np.random.randn(kh, kw)
        k_s = np.random.randn(kh, 1) @ np.random.randn(1, kw)

        max_err = max(_max_conv_discrepancy(im, k_ns), _max_conv_discrepancy(im, k_s))
        if max_err > 1e-10:
            all_passed = False
        print(f"  Test size {H}x{W} with {kh}x{kw} kernel -> Max discrepancy: {max_err:.2e}")

    if all_passed:
        print("  -> Sanity check PASSED: All three implementations match scipy to rounding error.")

    # 2. Benchmark on 256x256 image across odd Gaussian filters from 3x3 to 31x31
    print("\n--- 2. Timing Benchmark on 256x256 Image ---")
    np.random.seed(0)
    im_bench = np.random.rand(256, 256).astype(np.float64)
    kernel_sizes = [3, 5, 7, 9, 11, 15, 21, 31]

    loop_times = []
    vec_times = []
    sep_times = []
    scipy_times = []

    header = f"{'Filter Size':>12} | {'Loop (s)':>12} | {'Vectorized (s)':>15} | {'Separable (s)':>15} | {'Scipy (s)':>12}"
    print(header)
    print("-" * len(header))

    for ksize in kernel_sizes:
        sigma = ksize / 6.0
        k = make_gaussian_kernel_2d(ksize, sigma)

        # Loop timing
        t0 = time.perf_counter()
        _ = convolve2d_loop(im_bench, k)
        t_loop = time.perf_counter() - t0
        loop_times.append(t_loop)

        # Vectorized timing
        t0 = time.perf_counter()
        _ = convolve2d_vectorized(im_bench, k)
        t_vec = time.perf_counter() - t0
        vec_times.append(t_vec)

        # Separable timing
        t0 = time.perf_counter()
        _ = convolve2d_separable(im_bench, k)
        t_sep = time.perf_counter() - t0
        sep_times.append(t_sep)

        # Scipy timing
        t0 = time.perf_counter()
        _ = convolve2d(im_bench, k, mode="same", boundary="fill", fillvalue=0)
        t_scipy = time.perf_counter() - t0
        scipy_times.append(t_scipy)

        print(
            f"{f'{ksize}x{ksize}':>12} | {t_loop:12.4f} | {t_vec:15.5f} | {t_sep:15.5f} | {t_scipy:12.5f}"
        )

    # Plot timing benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(kernel_sizes, loop_times, "o-", label="Loop (4 nested loops)", color="#d62728", lw=2)
    plt.plot(kernel_sizes, vec_times, "s-", label="Vectorized (whole-image slices)", color="#1f77b4", lw=2)
    plt.plot(kernel_sizes, sep_times, "^-", label="Separable (SVD 1D cascade)", color="#2ca02c", lw=2)
    plt.plot(kernel_sizes, scipy_times, "x--", label="scipy.signal.convolve2d", color="#7f7f7f", lw=1.5)
    plt.yscale("log")
    plt.xlabel("Filter Size (k x k)", fontsize=12)
    plt.ylabel("Execution Time (seconds, log scale)", fontsize=12)
    plt.title("Convolution Execution Time vs. Filter Size (256x256 image)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    bench_plot_path = OUTPUT_CONVOLVE_DIR / "timing_benchmark.png"
    plt.savefig(bench_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved benchmark plot to: {bench_plot_path}")


def compute_fd_edges(
    image: np.ndarray,
    threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute finite difference derivatives, gradient magnitude, and thresholded edge map."""
    Ix, Iy, magnitude = compute_finite_differences(image, padding_mode="constant")
    edges = (magnitude > threshold).astype(np.float32)
    return Ix, Iy, magnitude, edges


def run_section_1_2(image_path: Path):
    im_name = image_path.stem
    print("=" * 80)
    print(f"SECTION 1.2: FINITE DIFFERENCE OPERATOR ({im_name})")
    print("=" * 80)

    im = load_image(image_path, gray=True)
    H, W = im.shape
    print(f"Loaded '{image_path.name}' (Shape: {H}x{W}, min: {im.min():.3f}, max: {im.max():.3f})")

    diff_dir = OUTPUT_DIFF_DIR / im_name
    diff_dir.mkdir(parents=True, exist_ok=True)

    # Compute finite differences and thresholded edges
    threshold = 0.20
    Ix, Iy, magnitude, edges = compute_fd_edges(im, threshold=threshold)

    edge_pct = np.mean(edges) * 100.0
    print(f"Selected threshold: {threshold:.2f} -> {edge_pct:.2f}% of pixels classified as edges")

    # Visualization: 4 maps
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # 1. dI/dx (rescaled with 0 at mid-gray)
    axes[0].imshow(rescale_midgray(Ix), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(r"$\partial I / \partial x$", fontsize=13)
    axes[0].axis("off")

    # 2. dI/dy (rescaled with 0 at mid-gray)
    axes[1].imshow(rescale_midgray(Iy), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(r"$\partial I / \partial y$", fontsize=13)
    axes[1].axis("off")

    # 3. Gradient magnitude
    axes[2].imshow(np.clip(magnitude, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Gradient magnitude", fontsize=13)
    axes[2].axis("off")

    # 4. Edges
    axes[3].imshow(edges, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Edges", fontsize=13)
    axes[3].axis("off")

    plt.tight_layout()
    out_fig_path = diff_dir / "finite_differences.png"
    plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved finite differences figure to: {out_fig_path}")

    # Also save individual images via utils.save_image for convenience
    save_image(diff_dir / "dI_dx.png", rescale_midgray(Ix))
    save_image(diff_dir / "dI_dy.png", rescale_midgray(Iy))
    save_image(diff_dir / "magnitude.png", np.clip(magnitude, 0, 1))
    save_image(diff_dir / "edges.png", edges)

    return magnitude, edges


def run_section_1_3(
    image_path: Path,
    fd_magnitude: np.ndarray | None = None,
    fd_edges: np.ndarray | None = None,
):
    im_name = image_path.stem
    print("=" * 80)
    print(f"SECTION 1.3: DERIVATIVE OF GAUSSIAN (DoG) FILTER ({im_name})")
    print("=" * 80)

    im = load_image(image_path, gray=True)

    gauss_dir = OUTPUT_GAUSS_DIR / im_name
    gauss_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build Gaussian kernel and DoG filters
    sigma = 2.0
    ksize = 13  # 2 * ceil(3*sigma) + 1 = 13
    DoG_x, DoG_y = build_dog_filters(ksize=ksize, sigma=sigma, padding_mode="constant")

    print(f"Constructed 2D Gaussian Kernel (size={ksize}x{ksize}, sigma={sigma:.1f})")
    print(f"Filtered with Dx and Dy to create DoG_x and DoG_y (shapes: {DoG_x.shape})")

    # 2. Apply DoG filters to image with a single convolution
    Ix_dog = convolve2d_separable(im, DoG_x, padding_mode="constant")
    Iy_dog = convolve2d_separable(im, DoG_y, padding_mode="constant")
    magnitude_dog = np.sqrt(Ix_dog**2 + Iy_dog**2)

    # 3. Repeat steps of 1.2: gradient magnitude and edge image
    threshold_dog = 0.07
    edges_dog = (magnitude_dog > threshold_dog).astype(np.float32)
    edge_pct_dog = np.mean(edges_dog) * 100.0

    print(f"Selected DoG threshold: {threshold_dog:.2f} -> {edge_pct_dog:.2f}% edge pixels")

    # 4. Visualization: DoG filters as images (saved once in OUTPUT_GAUSS_DIR)
    dog_filters_path = OUTPUT_GAUSS_DIR / "dog_filters.png"
    if not dog_filters_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rescale_midgray(DoG_x), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(r"DoG in $x$, $\sigma = 2$", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(rescale_midgray(DoG_y), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(r"DoG in $y$, $\sigma = 2$", fontsize=12)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(dog_filters_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nSaved DoG filters plot to: {dog_filters_path}")

    # Figure B: The four maps for DoG (matching handout page 3)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(rescale_midgray(Ix_dog), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(r"DoG in $x$, $\sigma = 2$ (0 = mid-gray)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(rescale_midgray(Iy_dog), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(r"DoG in $y$, $\sigma = 2$ (0 = mid-gray)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(np.clip(magnitude_dog / magnitude_dog.max(), 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(r"Gradient magnitude", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(edges_dog, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Edges (threshold = {threshold_dog})", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    dog_maps_path = gauss_dir / "dog.png"
    plt.savefig(dog_maps_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved DoG maps figure to: {dog_maps_path}")

    # Save individual images (consistently lowercase dog)
    save_image(gauss_dir / "dog_x_response.png", rescale_midgray(Ix_dog))
    save_image(gauss_dir / "dog_y_response.png", rescale_midgray(Iy_dog))
    save_image(gauss_dir / "dog_magnitude.png", np.clip(magnitude_dog / (magnitude_dog.max() + 1e-8), 0, 1))
    save_image(gauss_dir / "dog_edges.png", edges_dog)

    # Figure C: Direct side-by-side comparison between Finite Differences and DoG
    if fd_magnitude is None or fd_edges is None:
        _, _, mag_12, edges_12 = compute_fd_edges(im, threshold=0.20)
    else:
        mag_12, edges_12 = fd_magnitude, fd_edges

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(np.clip(mag_12, 0, 1), cmap="gray")
    axes[0, 0].set_title("Finite Diff Gradient Magnitude", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edges_12, cmap="gray")
    axes[0, 1].set_title("Finite Diff Edges (threshold = 0.20)", fontsize=12)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(np.clip(magnitude_dog / (magnitude_dog.max() + 1e-8), 0, 1), cmap="gray")
    axes[1, 0].set_title(r"DoG Gradient Magnitude ($\sigma = 2$)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(edges_dog, cmap="gray")
    axes[1, 1].set_title(f"DoG Edges (threshold = {threshold_dog})", fontsize=12)
    axes[1, 1].axis("off")

    plt.tight_layout()
    comp_path = gauss_dir / "comparison_diff_vs_dog.png"
    plt.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved finite differences vs DoG comparison figure to: {comp_path}")


def main():
    print("RUNNING ALL EXPERIMENTS AND DELIVERABLES FOR FILTERS...\n")

    # Section 1.1
    run_section_1_1()

    # Section 1.2 and 1.3 on facade.png and cameraman.png
    images = [
        ROOT_DIR / "images" / "filters" / "facade.png",
        ROOT_DIR / "images" / "filters" / "cameraman.png",
    ]
    for img_path in images:
        if img_path.exists():
            mag_12, edges_12 = run_section_1_2(img_path)
            run_section_1_3(img_path, fd_magnitude=mag_12, fd_edges=edges_12)

    print("\n" + "=" * 80)
    print("FILTERS COMPLETE! All figures and outputs saved in:", OUTPUT_FILTERS_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
