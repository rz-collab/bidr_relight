# BIDR Relight
# 12/11/25
# CS7180 Advanced Perception
# Contributors: Max Huber, Adharsh Kandula, Richard Zhao

# This file contains utility functions for BIDR theory, such as moving along ISD, 
# projection on log chromaticity plane, etc.
# Several components were coded/modified with the help of GPT-5 and Claude Sonnet 4.5


import numpy as np
import cv2


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
    lum = (
        0.2126 * linear_img[..., 0]
        + 0.7152 * linear_img[..., 1]
        + 0.0722 * linear_img[..., 2]
    )
    weight = np.clip(lum, 0.0, 1.0)

    # Apply compression IN LOG SPACE
    log_new = log_img - strength * weight[..., None] * isd

    return log_new


def project_to_log_chromaticity_plane(
    log_img,
    isd_map,
    plane_offset=np.array((5.5, 5.5, 5.5)),
    use_average_isd=False,
) -> np.ndarray:
    """
    Projects each pixel to a plane orthogonal to the ISD for that pixel.
    Adapted from https://github.com/mmkke/Spectral_Ratio_Illumination_Demo/tree/main

    Both arrays should be (H, W, 3).

    `plane_offset` is the offset/bias of the plane from origin defined as
    \hat{n} \dot (x - plane_offset) = 0.  (i.e., plane crosses point `plane_offset`)
    """
    assert isd_map.shape == log_img.shape

    shifted_log_rgb = log_img - plane_offset

    # Gets the log_rgb component as a vector along the isd direction for each pixel
    dot_product_map = (shifted_log_rgb * isd_map).sum(axis=2)  # (H, W) magnitude
    if use_average_isd:
        H, W, _ = isd_map.shape
        global_isd = get_global_isd(isd_map)  # (3,)
        isd_map = np.broadcast_to(global_isd, (H, W, 3))  # Repeat to get (H,W,3)
    projection = dot_product_map[:, :, np.newaxis] * isd_map  # vector

    # Subtract the projection from the shifted values to get plane-projected values
    projected_rgb = shifted_log_rgb - projection

    # Shift the values back by adding the anchor point
    projected_rgb += plane_offset
    return projected_rgb


def get_global_isd(isd_map):
    """
    Estimate the global ISD from ISD map.
    Currently it's done by averaging and renormalizing.
    """
    assert isd_map.ndim == 3 and isd_map.shape[-1] == 3
    global_isd = np.mean(isd_map, axis=(0, 1))
    global_isd /= np.linalg.norm(global_isd)
    return global_isd


def rotation_matrix_from_vectors(
    vec1, vec2, rot_angle=None, rot_percent: float = 100.0
):
    """
    Find the rotation matrix that rotates from vec1 to vec2 using Rodriguez Formula.

    The `rot_angle` and `rot_percent` are two possible ways to specify a desired rotation behavior.
    Use one or another, not both.
    - If `rot_angle`: rotate for a certain amount of degrees from vec1 to vec2.
    - If `rot_percent`: rotate `rot_percent` from vec1 to vec2.

    Ref: https://mathworld.wolfram.com/RodriguesRotationFormula.html
    """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    w = np.cross(vec1, vec2)  # Rotation axis
    w_norm = np.linalg.norm(w)
    if np.isclose(w_norm, 0):
        return np.eye(3)
    w = w / w_norm

    # Rotation axis skew sym matrix
    w_skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    # Compute the rotation matrix that rotates about \hat{w} for theta angle.
    # Find the desired rotation angle theta:
    if rot_angle is None and np.isclose(rot_percent, 100.0):
        # Default: Use the real theta between vec1 and vec2.
        theta = np.acos(np.dot(vec1, vec2))
    elif rot_angle is not None:
        # Use user-provided theta
        theta = rot_angle
    else:
        # Use `rot_percent` of real theta between vec1 and vec2.
        theta = np.acos(np.dot(vec1, vec2))
        theta = (rot_percent / 100.0) * theta

    # print(f"theta : {theta}")
    R = np.eye(3) + np.sin(theta) * w_skew + (1 - np.cos(theta)) * w_skew @ w_skew
    return R
