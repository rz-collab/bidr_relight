import numpy as np
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering

logger = logging.getLogger(__name__)


def cluster_greedy_proximity(log_chroma_content, bin_radius=1.0):
    """
    Greedy nearest-neighbor clustering in log chromaticity space.
    
    Parameters
    ----------
    log_chroma_content : np.ndarray
        (H, W, 3) array of log chromaticity projections
    bin_radius : float
        Distance threshold for clustering pixels together
        
    Returns
    -------
    bin_masks : list of np.ndarray
        List of (H, W) boolean masks, one per cluster
    bin_map : np.ndarray
        (H*W,) array of cluster IDs for each pixel
    """
    H, W, _ = log_chroma_content.shape
    log_chroma_content_flat = log_chroma_content.reshape(H * W, 3)
    
    bin_map = np.zeros(H * W, dtype=int)
    bin_masks = []
    UNASSIGNED = 0
    bin_id = 0

    for i in range(len(bin_map)):
        if bin_map[i] != UNASSIGNED:
            continue

        # Compute distance between this unassigned px and others in log chroma plane
        dist_to_other_pts = np.linalg.norm(
            log_chroma_content_flat - log_chroma_content_flat[i], axis=1
        )

        # Assign nearby unassigned (including this pixel) to a new bin
        bin_id += 1
        close_mask = dist_to_other_pts < bin_radius
        unassigned_mask = bin_map == UNASSIGNED
        new_bin_mask = np.logical_and(close_mask, unassigned_mask)
        bin_map[new_bin_mask] = bin_id

        bin_masks.append(new_bin_mask.reshape(H, W))

    assert np.all(bin_map != UNASSIGNED)
    logger.info(
        f"Greedy clustering: {bin_id} clusters with radius={bin_radius}"
    )
    
    return bin_masks, bin_map


def cluster_kmeans(log_chroma_content, n_clusters=10, random_state=42):
    """
    K-Means clustering in log chromaticity space.
    
    Parameters
    ----------
    log_chroma_content : np.ndarray
        (H, W, 3) array of log chromaticity projections
    n_clusters : int
        Number of clusters to create
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    bin_masks : list of np.ndarray
        List of (H, W) boolean masks, one per cluster
    bin_map : np.ndarray
        (H*W,) array of cluster IDs for each pixel
    """
    H, W, _ = log_chroma_content.shape
    log_chroma_content_flat = log_chroma_content.reshape(H * W, 3)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(log_chroma_content_flat)
    
    # Create bin masks (1-indexed to match greedy clustering)
    bin_masks = []
    for cluster_id in range(n_clusters):
        mask = (cluster_labels == cluster_id).reshape(H, W)
        bin_masks.append(mask)
    
    # Convert to 1-indexed bin_map
    bin_map = cluster_labels + 1
    
    logger.info(
        f"K-Means clustering: {n_clusters} clusters"
    )
    
    return bin_masks, bin_map


def cluster_hierarchical(log_chroma_content, n_clusters=10, linkage='ward', max_samples=10000, random_state=42):
    """
    Hierarchical (agglomerative) clustering in log chromaticity space.
    Uses sampling for large datasets to avoid memory issues.
    
    Parameters
    ----------
    log_chroma_content : np.ndarray
        (H, W, 3) array of log chromaticity projections
    n_clusters : int
        Number of clusters to create
    linkage : str
        Linkage criterion: 'ward', 'complete', 'average', or 'single'
    max_samples : int
        Maximum number of pixels to use for fitting (to avoid memory issues)
    random_state : int
        Random seed for sampling
        
    Returns
    -------
    bin_masks : list of np.ndarray
        List of (H, W) boolean masks, one per cluster
    bin_map : np.ndarray
        (H*W,) array of cluster IDs for each pixel
    """
    H, W, _ = log_chroma_content.shape
    log_chroma_content_flat = log_chroma_content.reshape(H * W, 3)
    
    # Sample pixels if dataset is too large
    n_pixels = H * W
    if n_pixels > max_samples:
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_pixels, max_samples, replace=False)
        sample_data = log_chroma_content_flat[sample_indices]
        logger.info(f"Sampling {max_samples}/{n_pixels} pixels for hierarchical clustering")
    else:
        sample_data = log_chroma_content_flat
    
    # Fit hierarchical clustering on sample
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric='euclidean'
    )
    hierarchical.fit(sample_data)
    
    # Predict labels for all pixels using nearest centroid
    centroids = np.array([
        sample_data[hierarchical.labels_ == i].mean(axis=0)
        for i in range(n_clusters)
    ])
    
    distances = np.linalg.norm(
        log_chroma_content_flat[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    cluster_labels = np.argmin(distances, axis=1)
    
    # Create bin masks (1-indexed to match greedy clustering)
    bin_masks = []
    for cluster_id in range(n_clusters):
        mask = (cluster_labels == cluster_id).reshape(H, W)
        bin_masks.append(mask)
    
    # Convert to 1-indexed bin_map
    bin_map = cluster_labels + 1
    
    logger.info(
        f"Hierarchical clustering: {n_clusters} clusters with {linkage} linkage"
    )
    
    return bin_masks, bin_map


def cluster_log_chromaticity(
    log_chroma_content,
    method="greedy",
    bin_radius=0.3,
    n_clusters=10,
    random_state=42,
    linkage='ward'
):
    """
    Cluster pixels in log chromaticity space using specified method.
    
    Parameters
    ----------
    log_chroma_content : np.ndarray
        (H, W, 3) array of log chromaticity projections
    method : str
        Clustering method: "greedy", "kmeans", or "hierarchical"
    bin_radius : float
        Distance threshold for greedy clustering
    n_clusters : int
        Number of clusters for K-Means and hierarchical
    random_state : int
        Random seed for K-Means
    linkage : str
        Linkage criterion for hierarchical clustering
        
    Returns
    -------
    bin_masks : list of np.ndarray
        List of (H, W) boolean masks, one per cluster
    bin_map : np.ndarray
        (H*W,) array of cluster IDs for each pixel
    """
    if method == "greedy":
        return cluster_greedy_proximity(log_chroma_content, bin_radius)
    elif method == "kmeans":
        return cluster_kmeans(log_chroma_content, n_clusters, random_state)
    elif method == "hierarchical":
        return cluster_hierarchical(log_chroma_content, n_clusters, linkage)
    else:
        raise ValueError(
            f"Unknown clustering method: {method}. "
            f"Choose 'greedy', 'kmeans', or 'hierarchical'"
        )