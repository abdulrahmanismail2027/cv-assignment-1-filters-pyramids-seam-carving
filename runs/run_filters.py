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

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from tabulate import tabulate

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

OUTPUT_FILTERS_DIR = ROOT_DIR / "outputs" / "filters"
OUTPUT_CONVOLVE_DIR = OUTPUT_FILTERS_DIR / "convolve"
OUTPUT_DIFF_DIR = OUTPUT_FILTERS_DIR / "diff"
OUTPUT_GAUSS_DIR = OUTPUT_FILTERS_DIR / "gauss"

OUTPUT_CONVOLVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIFF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GAUSS_DIR.mkdir(parents=True, exist_ok=True)


def run_convolution_from_scratch():
    print("1. Sanity Check against scipy.signal.convolve2d (mode='same')")
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
        print(f"\tTest size {H}x{W} with {kh}x{kw} kernel. Max discrepancy: {max_err:.2e}")

    if all_passed:
        print("\tSanity check passed: All three implementations match scipy to rounding error.")
    else:
        raise ValueError("Sanity check failed: Implementations deviate from scipy.")

    print("\n2. Timing Benchmark on 256x256 Image...")
    np.random.seed(0)
    im_bench = np.random.rand(256, 256).astype(np.float64)
    kernel_sizes = [3, 5, 7, 9, 11, 15, 21, 31]

    loop_times, vec_times, sep_times, scipy_times = [], [], [], []
    table_data = []

    for ksize in kernel_sizes:
        sigma = ksize / 6.0
        k = make_gaussian_kernel_2d(ksize, sigma)

        # Loop timing
        t0 = time.perf_counter()
        convolve2d_loop(im_bench, k)
        t_loop = time.perf_counter() - t0
        loop_times.append(t_loop)

        # Vectorized timing
        t0 = time.perf_counter()
        convolve2d_vectorized(im_bench, k)
        t_vec = time.perf_counter() - t0
        vec_times.append(t_vec)

        # Separable timing
        t0 = time.perf_counter()
        convolve2d_separable(im_bench, k)
        t_sep = time.perf_counter() - t0
        sep_times.append(t_sep)

        # Scipy timing
        t0 = time.perf_counter()
        convolve2d(im_bench, k, mode="same", boundary="fill", fillvalue=0)
        t_scipy = time.perf_counter() - t0
        scipy_times.append(t_scipy)

        # Append row data for the table
        table_data.append([
            f"{ksize}x{ksize}",
            t_loop,
            t_vec,
            t_sep,
            t_scipy
        ])

    headers = ["Filter Size", "Loop (s)", "Vectorized (s)", "Separable (s)", "Scipy (s)"]
    print(tabulate(table_data, headers=headers, tablefmt="pipe", floatfmt=(".0f", ".4f", ".5f", ".5f", ".5f")))

    # Plot timing benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(kernel_sizes, loop_times, "o-", label="Loop (4 nested loops)")
    plt.plot(kernel_sizes, vec_times, "s-", label="Vectorized (whole-image slices)")
    plt.plot(kernel_sizes, sep_times, "^-", label="Separable (SVD 1D cascade)")
    plt.plot(kernel_sizes, scipy_times, "x--", label="scipy.signal.convolve2d")

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

    print(f"\nSaved benchmark plot to: {bench_plot_path}\n")


def compute_fd_edges(
        image: np.ndarray,
        threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute finite difference derivatives, gradient magnitude, and thresholded edge map."""
    Ix, Iy, magnitude = compute_finite_differences(image, padding_mode="constant")
    edges = (magnitude > threshold).astype(np.float32)
    return Ix, Iy, magnitude, edges


def run_finite_difference_operator(image_path: Path):
    im_name = image_path.stem

    print(f"Image ({im_name}):")

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

    # Pre-process the arrays and map them to their titles and filenames
    maps = [
        (rescale_midgray(Ix), r"$\partial I / \partial x$", "dI_dx.png"),
        (rescale_midgray(Iy), r"$\partial I / \partial y$", "dI_dy.png"),
        (np.clip(magnitude, 0, 1), "Gradient magnitude", "magnitude.png"),
        (edges, "Edges", "edges.png")
    ]

    # Setup figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Iterate through axes and data simultaneously
    for ax, (data, title, filename) in zip(axes, maps):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

        # Save the individual image
        save_image(diff_dir / filename, data)

    # Save the combined figure
    plt.tight_layout()
    out_fig_path = diff_dir / "finite_differences.png"
    plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved finite differences figure to: {out_fig_path}\n")

    return magnitude, edges


def run_dog_filter(
        image_path: Path,
        fd_magnitude: np.ndarray,
        fd_edges: np.ndarray,
):
    im_name = image_path.stem
    print(f"Image: ({im_name})")

    im = load_image(image_path, gray=True)

    gauss_dir = OUTPUT_GAUSS_DIR / im_name
    gauss_dir.mkdir(parents=True, exist_ok=True)

    # Build Gaussian kernel and DoG filters
    sigma = 2.0
    ksize = 13  # 2 * ceil(3*sigma) + 1 = 13
    DoG_x, DoG_y = build_dog_filters(ksize=ksize, sigma=sigma, padding_mode="constant")

    print(f"Constructed 2D Gaussian Kernel (size={ksize}x{ksize}, sigma={sigma:.1f})")
    print(f"Filtered with Dx and Dy to create DoG_x and DoG_y (shapes: {DoG_x.shape})")

    # Apply DoG filters to image with a single convolution
    Ix_dog = convolve2d_separable(im, DoG_x, padding_mode="constant")
    Iy_dog = convolve2d_separable(im, DoG_y, padding_mode="constant")
    magnitude_dog = np.sqrt(Ix_dog ** 2 + Iy_dog ** 2)

    # Compute gradient magnitude and edge image
    threshold_dog = 0.07
    edges_dog = (magnitude_dog > threshold_dog).astype(np.float32)
    edge_pct_dog = np.mean(edges_dog) * 100.0

    print(f"Selected DoG threshold: {threshold_dog:.2f} -> {edge_pct_dog:.2f}% edge pixels")

    # Visualization of DoG filters as images
    norm_mag_dog = np.clip(magnitude_dog / (magnitude_dog.max() + 1e-8), 0, 1)

    # DoG Filters
    dog_filters_path = OUTPUT_GAUSS_DIR / "dog_filters.png"
    if not dog_filters_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        filters_data = [
            (rescale_midgray(DoG_x), r"DoG in $x$, $\sigma = 2$"),
            (rescale_midgray(DoG_y), r"DoG in $y$, $\sigma = 2$")
        ]
        for ax, (data, title) in zip(axes, filters_data):
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(dog_filters_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved DoG filters plot to: {dog_filters_path}")

    # The four maps for DoG
    dog_maps_data = [
        (rescale_midgray(Ix_dog), r"DoG in $x$, $\sigma = 2$ (0 = mid-gray)", "dog_x_response.png"),
        (rescale_midgray(Iy_dog), r"DoG in $y$, $\sigma = 2$ (0 = mid-gray)", "dog_y_response.png"),
        (norm_mag_dog, "Gradient magnitude", "dog_magnitude.png"),
        (edges_dog, f"Edges (threshold = {threshold_dog})", "dog_edges.png")
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (data, title, filename) in zip(axes, dog_maps_data):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        save_image(gauss_dir / filename, data)

    plt.tight_layout()
    dog_maps_path = gauss_dir / "dog.png"
    plt.savefig(dog_maps_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved DoG maps figure to: {dog_maps_path}")

    # Comparison between Finite Differences and DoG
    comp_data = [
        (np.clip(fd_magnitude, 0, 1), "Finite Diff Gradient Magnitude"),
        (fd_edges, "Finite Diff Edges (threshold = 0.20)"),
        (norm_mag_dog, r"DoG Gradient Magnitude ($\sigma = 2$)"),
        (edges_dog, f"DoG Edges (threshold = {threshold_dog})")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (data, title) in zip(axes.flat, comp_data):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    comp_path = gauss_dir / "comparison_diff_vs_dog.png"
    plt.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved finite differences vs DoG comparison figure to: {comp_path}\n")


def main():
    print("Convolution from Scratch:")
    run_convolution_from_scratch()

    images = [
        ROOT_DIR / "images" / "filters" / "facade.png",
        ROOT_DIR / "images" / "filters" / "cameraman.png",
    ]

    print(f"Finite Difference Operator:")
    fd = {}
    for img_path in images:
        mag, edges = run_finite_difference_operator(img_path)
        fd[img_path] = (mag, edges)

    print(f"Derivative of Gaussian:")
    for img_path, (mag, edges) in fd.items():
        run_dog_filter(img_path, fd_magnitude=mag, fd_edges=edges)


if __name__ == "__main__":
    main()
