# BIDR Relight
# 12/11/25
# CS7180 Advanced Perception
# Contributors: Max Huber, Adharsh Kandula, Richard Zhao

# This file contains image utility functions for color space conversions and resizing.
# Several components were coded/modified with the help of GPT-5 and Claude Sonnet 4.5

import numpy as np
import cv2

# All functions assumes HxWxC shape np array as image.


def linear_to_log(linear_img: np.ndarray) -> np.ndarray:
    """
    Converts a linear image to logarithmic space.

    Args:
        linear_img (np.ndarray): Input image in linear space (float32 or float64), with non-negative values.

    Returns:
        np.ndarray: Image converted to log space (float32), with zeros where input is zero or negative.
    """
    log_img = np.zeros_like(linear_img, dtype=np.float32)
    log_img[linear_img > 0] = np.log(linear_img[linear_img > 0])
    assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1
    return log_img.astype(np.float32)


def log_to_linear(log_img: np.ndarray) -> np.ndarray:
    """
    Converts a logarithmic image back to linear space.

    Args:
        log_img (np.ndarray): Input image in log space (float32 or float64), typically the output of linear_to_log.

    Returns:
        np.ndarray: Image converted back to linear space (float32).
    """
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


def resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Resize an image to the specified height and width using area interpolation.

    Args:
        img (np.ndarray): Input image array (HxWxC or HxW).
        height (int): Desired output height (pixels).
        width (int): Desired output width (pixels).

    Returns:
        np.ndarray: Resized image with shape (height, width, C) or (height, width).
    """
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
