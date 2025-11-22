import numpy as np

def apply_isd_darkening_log(log_img, isd_map, strength=0.5):
    """
    Darkens each pixel correctly along ISD direction in LOG-RGB space.
    Darker pixels are affected less using luminance weighting in linear RGB.

    Args:
        log_img:  (H,W,3) log-linear RGB
        isd_map:  (3,H,W) ISD directions (unit vectors)
        strength: scalar; amount of compression along the ISD
    """

    # Convert ISD to (H,W,3)
    isd = np.transpose(isd_map, (1, 2, 0))

    # Compute linear for luminance weighting
    linear_img = np.exp(log_img)

    # Luminance-based weight
    lum = (0.2126 * linear_img[...,0] +
           0.7152 * linear_img[...,1] +
           0.0722 * linear_img[...,2])
    weight = np.clip(lum, 0.0, 1.0)

    # Apply compression IN LOG SPACE
    log_new = log_img - strength * weight[...,None] * isd

    return log_new
