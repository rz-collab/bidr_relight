import numpy as np
import cv2

# All functions assumes HxWxC shape np array as image.


def linear_to_log(linear_img: np.ndarray) -> np.ndarray:
    log_img = np.zeros_like(linear_img, dtype=np.float32)
    log_img[linear_img > 0] = np.log(linear_img[linear_img > 0])
    assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1
    return log_img.astype(np.float32)


def log_to_linear(log_img: np.ndarray) -> np.ndarray:
    return np.exp(log_img).astype(np.float32)


def normalized_linear_to_srgb(linear_rgb):
    """
    Converts linear RGB values to sRGB using the standard sRGB transfer function.

    Args:
        linear_rgb (np.ndarray): Input image, float32 or float64, values in [0, 1].

    Returns:
        np.ndarray: sRGB image, uint8 values in [0, 255]
    """
    linear_rgb = np.clip(linear_rgb, 0, 1)
    threshold = 0.0031308
    below = linear_rgb <= threshold
    above = ~below

    srgb = np.zeros_like(linear_rgb)
    srgb[below] = 12.92 * linear_rgb[below]
    srgb[above] = 1.055 * (linear_rgb[above] ** (1 / 2.4)) - 0.055

    srgb = (srgb * 255.0).astype(np.uint8)
    return srgb


def convert_16bit_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Converts a 16-bit image to an 8-bit image for display.

    Parameters:
    -----------
    img : np.ndarray
        Input 16-bit or float image.

    Returns:
    --------
    np.ndarray
        Output image in 8-bit (uint8).
    """
    img_clipped = np.clip(img, 0, 65535)
    img_normalized = (img_clipped / 256.0).astype(np.uint8)
    return img_normalized


def crop_to_match(img, ref):
    """
    Crop `img` to the same HxW as `ref`, centered.
    """
    H, W = ref.shape[:2]
    h, w = img.shape[:2]

    # compute top-left corner
    y0 = max(0, (h - H) // 2)
    x0 = max(0, (w - W) // 2)

    return img[y0 : y0 + H, x0 : x0 + W]


def resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def resize_with_same_aspect(img: np.ndarray, scale: float):
    """
    Resize keeping same aspect ratio with each size equal to power of 2.
    e.g., scale = 0.5 means downsample by 2.
    """

    def nearest_power_of_2(x):
        return 2 ** int(np.round(np.log2(x)))

    H, W = img.shape[:2]
    new_H = nearest_power_of_2(H * scale)
    new_W = nearest_power_of_2(W * scale)
    return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
