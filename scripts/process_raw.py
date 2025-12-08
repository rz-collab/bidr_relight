import rawpy
from skimage.io import imsave
import argparse
from pathlib import Path
import numpy as np


def process_raw_into_linear(dir):
    """Process RAW images and store them as 16-bit TIFF."""
    raw_formats = ["cr2", "dng"]
    for raw_format in raw_formats:
        imgs = list(Path(dir).glob(f"*.{raw_format}"))
        print(f"Found {len(imgs)} {raw_format} images in {dir}")

        for img_path in imgs:
            with rawpy.imread(str(img_path)) as raw:
                # Debayer to linear 16-bit RGB
                # Postprocess already do black-level substraction, and autoscales camera's white level to bit depth max
                img_linear = raw.postprocess(
                    output_color=rawpy.ColorSpace.raw,
                    output_bps=16,
                    no_auto_bright=True,
                    no_auto_scale=False,
                    use_camera_wb=False,
                    use_auto_wb=False,
                    gamma=(1, 1),
                )

            # Save as 16-bit TIFF
            out_path = img_path.with_suffix(".tiff")
            img_linear = np.clip(img_linear, 0, 65535).astype(np.uint16)
            imsave(out_path, img_linear)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    process_raw_into_linear(dir=args.path)
    print("Process done")
