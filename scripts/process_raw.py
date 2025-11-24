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
                img_linear = raw.postprocess(
                    output_color=rawpy.ColorSpace.raw,
                    output_bps=16,
                    no_auto_bright=True,
                    use_camera_wb=False,
                    use_auto_wb=False,
                    gamma=(1, 1),
                )

                # # Subtract approximate black level
                # bl_min = min(raw.black_level_per_channel)
                # img_linear = np.clip(img_linear.astype(np.int32) - bl_min, 0, 65535).astype(
                #     np.uint16
                # )

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
