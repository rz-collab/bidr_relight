#!/usr/bin/env python3
"""
AI Generated Interactive ROI selector for 16-bit images.
Allows user to select a rectangular region of interest and displays the result.
"""

from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import argparse
import sys
from pathlib import Path
from bidr_relight.image_process import normalized_linear_to_srgb


def interactive_roi_16bit(img_path, show_roi=True):
    """
    Opens a 16-bit image from a file path, allows interactive rectangle cropping,
    and returns the cropped ROI.

    Parameters:
    -----------
    img_path : str
        Path to the image file
    show_roi : bool, optional
        If True, display the ROI in a new window

    Returns:
    --------
    tuple : (original_image, cropped_roi, roi_coordinates) or None if no selection
        roi_coordinates format: (y, x, height, width)
    """
    # Load image from path
    try:
        img = imread(img_path)
        print(f"Loaded image: {img_path}")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Handle different bit depths
    if img.dtype == np.uint16:
        vmax = 65535
    elif img.dtype == np.uint8:
        vmax = 255
    else:
        vmax = img.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    img = img / vmax
    img = normalized_linear_to_srgb(img)
    ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=vmax)
    ax.set_title("Drag to select ROI — close window when done.")
    roi_coords = {"state": None}

    # Callback: fires when ROI selected
    def onselect(eclick, erelease):
        y1, x1 = int(eclick.ydata), int(eclick.xdata)
        y2, x2 = int(erelease.ydata), int(erelease.xdata)

        # Normalize coordinates (handle dragging in any direction)
        y, h = sorted([y1, y2])
        x, w = sorted([x1, x2])

        roi_coords["state"] = (y, x, h - y, w - x)
        print(f"ROI selected: y={y}, x={x}, h={h - y}, w={w - x}")

    # Create interactive selector
    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],  # left mouse
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
    )

    plt.show()  # User interacts until window closes

    # Nothing selected
    if roi_coords["state"] is None:
        print("No ROI selected.")
        return None

    # Extract ROI
    y, x, h, w = roi_coords["state"]
    crop = img[y : y + h, x : x + w]

    # Print ROI information
    print(f"\n{'=' * 50}")
    print(f"ROI Selected: ({y}, {x}, {h}, {w})")
    print(f"  Position: (y={y}, x={x})")
    print(f"  Size: {h} × {w} pixels")
    print(f"  ROI shape: {crop.shape}")
    print(f"  ROI dtype: {crop.dtype}")
    print(f"  ROI value range: [{crop.min()}, {crop.max()}]")
    print(f"{'=' * 50}\n")

    # Display ROI if requested
    if show_roi:
        # Handle different bit depths for display
        if crop.dtype == np.uint16:
            vmax_roi = 65535
        elif crop.dtype == np.uint8:
            vmax_roi = 255
        else:
            vmax_roi = crop.max()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Show full image with ROI box
        ax1.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=vmax)
        rect = plt.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.set_title("Full Image with ROI")
        ax1.axis("off")

        # Show cropped ROI
        ax2.imshow(crop, cmap=None if crop.ndim == 3 else "gray", vmin=0, vmax=vmax_roi)
        ax2.set_title(f"ROI: {h}×{w} pixels")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    return img, crop, roi_coords["state"]


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI selector for images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.tif
  %(prog)s image.tif --no-plot
        """,
    )

    parser.add_argument("image", type=str, help="Path to input image file")

    parser.add_argument(
        "--no-plot", action="store_true", help="Don't display the ROI after selection"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.image).exists():
        print(f"Error: Input file '{args.image}' not found.")
        sys.exit(1)

    # Run interactive ROI selection
    result = interactive_roi_16bit(args.image, show_roi=not args.no_plot)

    if result is None:
        print("No ROI selected or error occurred.")
        sys.exit(1)

    original, roi, coords = result
    print(f"Original image shape: {original.shape}")
    print(f"ROI coordinates (y, x, h, w): {coords}")
    print("\nDone!")


if __name__ == "__main__":
    main()
