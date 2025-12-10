"""Illumination vector and dark/bright point estimation."""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_signed_distances(log_img, log_chroma, isd_map):
    """Compute signed distance along ISD for each pixel."""
    diff_vec = log_img - log_chroma
    signed_dist_map = (diff_vec * isd_map).sum(axis=2)
    return signed_dist_map


def estimate_cluster_lengths(signed_dist_map, bin_masks):
    """Estimate illumination length for each cluster."""
    lengths = []
    for bin_mask in bin_masks:
        signed_dists = signed_dist_map[bin_mask].ravel()
        p5 = np.percentile(signed_dists, 5)
        p95 = np.percentile(signed_dists, 95)
        lengths.append(p95 - p5)
    return np.array(lengths)


def find_illumination_norm(lengths):
    """Find global illumination vector norm from length distribution."""
    bin_counts, bin_edges = np.histogram(lengths)
    bin_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    count_threshold = 0.3 * bin_counts.max()
    mode_x = bin_x[bin_counts > count_threshold]
    
    illum_norm = mode_x[-1]
    logger.info(f"Estimated illumination vector norm: {illum_norm:.3f}")
    
    return illum_norm


def estimate_dark_bright_points(log_img, isd_maps, signed_dist_map, 
                                bin_masks, lengths, illum_norm,
                                always_use_global=True):
    """Estimate fully dark and bright points for each material cluster."""
    global_dists = signed_dist_map.ravel()
    global_range = np.percentile(global_dists, 95) - np.percentile(global_dists, 5)
    global_median = np.percentile(global_dists, 50)
    
    dark_points = []
    bright_points = []
    
    for bin_idx, bin_mask in enumerate(bin_masks):
        bin_isd = isd_maps[bin_mask].mean(axis=0)
        bin_isd = bin_isd / np.linalg.norm(bin_isd)
        
        length = lengths[bin_idx]
        signed_dists_bin = signed_dist_map[bin_mask].ravel()
        p5 = np.percentile(signed_dists_bin, 5)
        p95 = np.percentile(signed_dists_bin, 95)
        
        bin_indices = np.array(np.where(bin_mask)).T
        p5_idx = np.argmin(np.abs(signed_dists_bin - p5))
        p95_idx = np.argmin(np.abs(signed_dists_bin - p95))
        p5_point = log_img[tuple(bin_indices[p5_idx])]
        p95_point = log_img[tuple(bin_indices[p95_idx])]
        
        is_degenerate = always_use_global or (length < 0.3 * global_range)
        
        if is_degenerate:
            median_dist = np.median(signed_dists_bin)
            if median_dist > global_median:
                bright_point = p95_point
                dark_point = bright_point - illum_norm * bin_isd
            else:
                dark_point = p5_point
                bright_point = dark_point + illum_norm * bin_isd
        else:
            dark_point = p5_point
            bright_point = p95_point
        
        dark_points.append(dark_point)
        bright_points.append(bright_point)
    
    logger.info(f"Estimated dark/bright points for {len(bin_masks)} clusters")
    return np.array(dark_points), np.array(bright_points)


def estimate_illumination(log_img, log_chroma, isd_maps, bin_masks,
                         always_use_global=True):
    """Complete illumination estimation pipeline."""
    signed_dist_map = compute_signed_distances(log_img, log_chroma, isd_maps)
    lengths = estimate_cluster_lengths(signed_dist_map, bin_masks)
    illum_norm = find_illumination_norm(lengths)
    dark_points, bright_points = estimate_dark_bright_points(
        log_img, isd_maps, signed_dist_map, bin_masks, lengths, 
        illum_norm, always_use_global
    )
    
    return illum_norm, dark_points, bright_points, signed_dist_map