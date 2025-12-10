"""Relighting transformation module."""
import numpy as np
import logging

from src.bidr_util import rotation_matrix_from_vectors, get_global_isd

logger = logging.getLogger(__name__)


def apply_relighting(log_img, isd_map_content, isd_map_style, 
                     bin_masks, dark_points,
                     length_scale=1.0, log_transl=None,
                     rot_percent=100.0, rot_angle=None):
    """Apply relighting transformation to content image."""
    global_style_isd = get_global_isd(isd_map_style)
    global_content_isd = get_global_isd(isd_map_content)
    
    logger.info(f"Content ISD: {global_content_isd}")
    logger.info(f"Style ISD: {global_style_isd}")
    
    R = rotation_matrix_from_vectors(
        global_content_isd,
        global_style_isd,
        rot_percent=rot_percent,
        rot_angle=rot_angle,
    )
    
    tf_log_img = np.copy(log_img)
    
    for cyl_idx, cyl_mask in enumerate(bin_masks):
        cyl_dark_point = dark_points[cyl_idx]
        cyl_px_idx = np.where(cyl_mask.ravel())[0]
        
        for px_idx in cyl_px_idx:
            h, w = np.unravel_index(px_idx, cyl_mask.shape)
            log_px = log_img[h, w]
            
            rel = log_px - cyl_dark_point
            tf_log_img[h, w] = cyl_dark_point + length_scale * R @ rel
    
    logger.info("Applied relighting transformation")
    
    if log_transl is not None:
        tf_log_img = tf_log_img + log_transl
        logger.info(f"Applied log translation: {log_transl}")
    
    return tf_log_img