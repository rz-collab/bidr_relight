"""Posterization utilities for improved clustering."""
import numpy as np


def posterize_image(img, levels=8):
    """
    Posterize image to reduce color levels.
    
    Parameters
    ----------
    img : np.ndarray
        Image array (H, W, C)
    levels : int
        Number of discrete levels per channel (2-256)
        
    Returns
    -------
    posterized : np.ndarray
        Posterized image with same shape and dtype
    """
    if levels < 2 or levels > 256:
        raise ValueError("levels must be between 2 and 256")
    
    # Get range
    img_min = img.min()
    img_max = img.max()
    img_range = img_max - img_min
    
    # Normalize to [0, 1]
    normalized = (img - img_min) / (img_range + 1e-10)
    
    # Quantize to levels
    step = 1.0 / levels
    quantized = np.floor(normalized / step) * step + step / 2
    
    # Scale back
    posterized = quantized * img_range + img_min
    
    return posterized.astype(img.dtype)


def posterize_log_image(log_img, levels=8):
    """Posterize in log space (operates on log values directly)."""
    return posterize_image(log_img, levels)