import numpy as np
import torch
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import os
import logging

from bidr_relight.isd_estimator.mock import MockISDModel
from bidr_relight.isd_estimator.unet import ResNet50UNet
from bidr_relight.clustering import cluster_log_chromaticity

from bidr_relight.image_process import (
    resize_with_same_aspect,
    linear_to_log,
    resize,
)
from bidr_relight.bidr_process import (
    project_to_log_chromaticity_plane,
    get_global_isd,
    rotation_matrix_from_vectors,
)

# from bidr_relight.illuminant_estimation import RecursiveRetinex

from bidr_relight.plot import (
    plot_img_rgb_logrgb,
    plot_content_log_chroma,
    plot_plane,
    calculate_shared_limits,
    plane_view_from_normal,
    plot_log_chroma_plane_pre_clustering,
    plot_log_chroma_plane_post_clustering,
    plot_cluster_spatial_distribution,
    plot_transformed_img_logrgb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def relight_content_image(
    content_path,
    style_path,
    isd_model,
    output_path,
    resize_scale=1 / 4,
    clustering_method="greedy",
    bin_radius=1.0,
    n_clusters=4,
    view_isd=False,
    length_scale=1.0,
    log_transl=None,
    rot_percent=100.0,  # you either use rot_percent or rot_angle
    rot_angle=None,
    always_use_global_illum_norm=True,
    target_vector=None,
    content_roi=None,
    style_roi=None,
):
    """
    Vectorized relighting pipeline using ISDs and optional illuminant transfer.

    Parameters
    ----------
    content_path : str
        Path to the content image.
    style_path : str
        Path to the style image.
    isd_model : callable
        Model that takes a content image tensor and returns ISD map: shape (1, 3, H, W)
    isd_model_path : str
        Path to isd model weights
    output_path : str
        Path to save the transformed image.
    resize_scale : float
        Scale each image by this factor maintaining same aspect ratio.
        e.g., resize_scale = 1/2 means downsample by 2. Useful for low RAM,
        as model alone consumes >25GB.
    bin_radius: float
        pixels are clustered into bins of `bin_radius` size in log chroma plane.
    shading_only : bool, default False
        If True, only compress along the ISD without changing illuminant color.
    compression_factor : float, default 0.7
        Factor to compress intensity along the ISD.
    """
    CONTENT = 0
    STYLE = 1
    model = isd_model
    model.eval()

    # --- 1. Load and preprocess images ---
    img_paths = [content_path, style_path]
    imgs_roi = [content_roi, style_roi]
    imgs = []
    imgs_bit_depth = []
    log_imgs = []
    log_norm_imgs = []
    imgs_roi_color = []  # unnormalized linear rgb color

    for i in range(len(img_paths)):
        img = imread(img_paths[i])
        img_bit_depth = np.iinfo(img.dtype).bits

        if imgs_roi[i] is not None:
            # Extract ROI and compute its average linear RGB.
            y, x, h, w = imgs_roi[i]
            roi = img[y : y + h, x : x + w]
            roi_color = np.mean(roi.reshape(-1, 3), axis=0)
            imgs_roi_color.append(roi_color)

        if i == CONTENT:
            img = resize_with_same_aspect(img, scale=resize_scale)
        else:
            # Match style image size with content image.
            H, W, _ = imgs[CONTENT].shape
            img = resize(img, H, W)

        # Drop alpha if present
        img = img[:, :, :3]

        # Convert to log RGB and normalize to unit range
        log_img = linear_to_log(img)
        log_norm_img = log_img / np.log(2**img_bit_depth - 1)
        log_norm_img = log_norm_img.astype(np.float32)

        imgs.append(img)
        imgs_bit_depth.append(img_bit_depth)
        log_imgs.append(log_img)
        log_norm_imgs.append(log_norm_img)

    # --- 2. Use pretrained ISD estimator to get ISD maps ---
    isd_maps = []

    for log_norm_img in log_norm_imgs:
        # Estimate ISD map
        log_norm_img_tensor = (
            torch.from_numpy(log_norm_img).permute(2, 0, 1).unsqueeze(0)
        )
        isd_map = model(log_norm_img_tensor)

        # Convert back to np.array
        isd_map = isd_map.detach().squeeze(0).numpy()  # (3, H, W)
        isd_map = np.transpose(isd_map, (1, 2, 0))  # (H, W, 3)

        # Normalize output to unit vector
        isd_norm = np.linalg.norm(isd_map, axis=2, keepdims=True)
        isd_norm[isd_norm == 0] = 1
        isd_map = isd_map / isd_norm
        isd_maps.append(isd_map)

    # --- 3. Segment pixels by material: We group pixels whose projections are close in the 2D log-chromaticity plane (the plane orthogonal to the ISD). ---
    plane_offset = np.array((10.4, 10.4, 10.4))
    log_chroma_content = project_to_log_chromaticity_plane(
        log_imgs[CONTENT],
        isd_maps[CONTENT],
        plane_offset=plane_offset,
        use_average_isd=False,
    )  # (H, W, 3)

    # Visualize before clustering
    # For plotting RGB/logRGB, sample points:
    num_samples = 100000
    log_chroma_content_flat = log_chroma_content.reshape(-1, 3)
    if num_samples > len(log_chroma_content_flat):
        sample_indices = np.arange(len(log_chroma_content_flat))
    else:
        sample_indices = np.random.choice(
            len(log_chroma_content_flat), num_samples, replace=False
        )

    plot_log_chroma_plane_pre_clustering(
        log_chroma_content,
        isd_maps[CONTENT],
        imgs[CONTENT],
        imgs_bit_depth[CONTENT],
        sample_indices,
    )

    # Perform clustering
    # bla = np.exp(log_chroma_content) / (2 ** imgs_bit_depth[CONTENT] - 1)
    bla = log_chroma_content
    bin_masks, bin_map = cluster_log_chromaticity(
        bla,
        method=clustering_method,
        bin_radius=bin_radius,
        n_clusters=n_clusters,
    )

    # Visualize after clustering
    plot_log_chroma_plane_post_clustering(
        log_chroma_content,
        isd_maps[CONTENT],
        bin_masks,
        bin_radius if clustering_method == "greedy" else None,
        sample_indices,
    )
    plot_cluster_spatial_distribution(bin_masks, imgs[CONTENT], imgs_bit_depth[CONTENT])

    # --- 4. Find the global illumination vector of content image. ---
    # Details:
    # Under the assumption of uniform spectral ratio (i.e., same ambient and direct),
    # each material's vector between fully lit and fully dark in log RGB will have the same direction (ISD)
    # and same norm. We will denote this as "illumination vector" (referred as N in BIDR paper) and estimate this
    # as the rightmost mode of length distribution.

    # Compute signed dist along isd for each pixel.
    diff_vec = log_imgs[CONTENT] - log_chroma_content
    signed_dist_map = (diff_vec * isd_maps[CONTENT]).sum(
        axis=2
    )  # dot product into (H,W)

    # Get the 5th and 95th percentile of signed dist distribution for each bin of pixels.
    lengths = []
    for bin_mask in bin_masks:
        signed_dists = signed_dist_map[bin_mask].ravel()

        p5 = np.percentile(signed_dists, 5)
        p95 = np.percentile(signed_dists, 95)

        # plt.figure()
        # plt.hist(signed_dists)
        # add percentiles lines
        # for p in [p5, p95]:
        #     plt.axvline(p, color="red", linestyle="--", linewidth=1)
        length = p95 - p5
        lengths.append(length)

    # Create a histogram from this length array
    bin_counts, bin_edges = np.histogram(np.array(lengths))
    bin_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Use center as their position

    # Extract peaks/modes from this histogram.
    # Modes are defined as those histogram bins with relatively high counts.
    # The count threshold is dynamically set to 30% of max count.
    count_threshold = 0.3 * bin_counts.max()
    # mode_counts = bin_counts[bin_counts > count_threshold]
    mode_x = bin_x[bin_counts > count_threshold]

    # Use the rightmost mode as the illum vector norm.
    illum_vector_norm = mode_x[-1]
    logger.info(f"Estimated illumination vector norm {illum_vector_norm}")

    # --- 5. Estimate fully (dark, bright) pairs for each material. ---
    # Identify clusters with only lit or only shaded pixels and estimate missing points.
    # First, compute the global range (95th - 5th percentile) for the whole image
    global_signed_dists = signed_dist_map.ravel()
    global_p95 = np.percentile(global_signed_dists, 95)
    global_p5 = np.percentile(global_signed_dists, 5)

    # For each cluster, determine if it's fully lit, fully shaded, or mixed
    dark_points = []
    bright_points = []
    global_content_isd = get_global_isd(isd_maps[CONTENT])

    for bin_idx, bin_mask in enumerate(bin_masks):
        bin_isd = isd_maps[CONTENT][bin_mask].mean(axis=0)
        bin_isd = bin_isd / np.linalg.norm(bin_isd)
        # bin_chroma = log_chroma_content[bin_mask].mean(axis=0)

        length = lengths[bin_idx]
        signed_dists_bin = signed_dist_map[bin_mask].ravel()

        p5 = np.percentile(signed_dists_bin, 5)
        p95 = np.percentile(signed_dists_bin, 95)
        bin_indices = np.array(np.where(bin_mask)).T
        p5_idx = np.argmin(np.abs(signed_dists_bin - p5))
        p95_idx = np.argmin(np.abs(signed_dists_bin - p95))
        p5_point = log_imgs[CONTENT][tuple(bin_indices[p5_idx])]
        p95_point = log_imgs[CONTENT][tuple(bin_indices[p95_idx])]

        if always_use_global_illum_norm:
            is_degenerate = True
        else:
            is_degenerate = length < 0.3 * illum_vector_norm
        if is_degenerate:
            if np.abs(global_p95 - p95) < np.abs(global_p5 - p5):
                # Fully lit: use real p95 as bright, estimate dark
                bright_point = p95_point
                dark_point = bright_point - illum_vector_norm * bin_isd
            else:
                # Fully dark: use real p5 as dark, estimate bright
                dark_point = p5_point
                bright_point = dark_point + illum_vector_norm * bin_isd
        else:
            # Mixed: use real p5/p95 as endpoints
            # dark_point = bin_chroma + p5 * bin_isd
            # bright_point = bin_chroma + p95 * bin_isd
            dark_point = p5_point
            bright_point = p95_point

        dark_points.append(dark_point)
        bright_points.append(bright_point)

    dark_points = np.array(dark_points)
    bright_points = np.array(bright_points)
    logger.info(
        f"Estimated dark and bright points for {len(bin_masks)} material clusters"
    )

    # print("Dark points: ", dark_points)
    # print("Bright points: ", bright_points)

    # --- 6. Pivot each material around their dark point from content ISD to the average style ISD. ---

    # For each cylinder, we rotate its pixels about the cylinder's dark point from content ISD to style ISD.
    # By default, this pure rotation maintains length of px from their corresponding dark point.
    # If a proportional `length_scale` is provided (not =1.0), we rotate + scale.

    global_style_isd = get_global_isd(isd_maps[STYLE])
    global_content_isd = get_global_isd(isd_maps[CONTENT])
    tf_log_content = np.copy(log_imgs[CONTENT])

    # Compute rotation matrix that rotates content ISD to style ISD,
    R = rotation_matrix_from_vectors(
        global_content_isd,
        global_style_isd,
        rot_percent=rot_percent,
        rot_angle=rot_angle,
    )
    if target_vector is not None:
        R = rotation_matrix_from_vectors(
            global_content_isd,
            target_vector,
            rot_percent=rot_percent,
            rot_angle=rot_angle,
        )

    logger.info(
        f"Average Style ISD: {global_style_isd}. Average Content ISD: {global_content_isd}"
    )

    for cyl_idx, cyl_mask in enumerate(bin_masks):
        # Get cylinder's (dark,bright) pair
        cyl_dark_point = dark_points[cyl_idx]
        cyl_bright_point = bright_points[cyl_idx]
        # print(np.linalg.norm(cyl_bright_point - cyl_dark_point))
        # print(
        #     np.acos(
        #         np.dot(
        #             (cyl_bright_point - cyl_dark_point)
        #             / np.linalg.norm(cyl_bright_point - cyl_dark_point),
        #             global_content_isd,
        #         )
        #     )
        #     * 180
        #     / np.pi
        # )

        # Iterate through pixels that belongs to this cluster to apply the transformation
        cyl_px_idx = np.where(cyl_mask.ravel())[0]
        for px_idx in cyl_px_idx:
            h, w = np.unravel_index(px_idx, cyl_mask.shape)
            log_px = log_imgs[CONTENT][h, w]

            # Rotate (with optional linear scaling)
            rel = log_px - cyl_dark_point
            transformed_log_px = cyl_dark_point + length_scale * R @ rel
            tf_log_content[h, w] = transformed_log_px

    logger.info("Pivoted all pixels for each material cluster.")

    # --- 7. Optional global translation in log RGB for all pixels to change ambient illuminant.---

    if content_roi is not None and style_roi is not None:
        # Compute global translation to match their log RGB color.
        log_transl = np.log(imgs_roi_color[STYLE] + 1e-8) - np.log(
            imgs_roi_color[CONTENT] + 1e-8
        )
        logger.info(
            f"Content's ROI color (linear) {imgs_roi_color[CONTENT]},  Style's ROI color (linear) {imgs_roi_color[STYLE]}"
        )
        logger.info(f"{log_transl=}")

    # # Extract ROI and compute its average linear RGB.
    # if content_roi is not None and style_roi is not None:
    #     # Get tf content ROI log RGB.
    #     content_roi = np.array(content_roi)
    #     content_roi = (content_roi * resize_scale).astype(np.uint32)
    #     y, x, h, w = content_roi
    #     tf_roi_img = tf_log_content[y : y + h, x : x + w]
    #     tf_roi_logrgb = np.mean(tf_roi_img.reshape(-1, 3), axis=0)

    #     # Get style ROI log RGB
    #     style_roi_logrgb = np.log(imgs_roi_color[STYLE] + 1e-8)

    #     # Compute global translation to match their log RGB color.
    #     log_transl = style_roi_logrgb - tf_roi_logrgb
    #     logger.info(
    #         f"Transformed Content's ROI color (linear) {tf_roi_logrgb},  Style's ROI color (linear) {imgs_roi_color[STYLE]}"
    #     )
    #     logger.info(f"{log_transl=}")

    if log_transl is not None:
        tf_log_content = tf_log_content + log_transl

    # --- 8. Plots: log chroma, illum norm distribution, sRGB, logRGB. ---

    # Prepare data for plotting
    content_img, style_img = imgs
    content_bit_depth, style_bit_depth = imgs_bit_depth
    norm_content_img = content_img / (2**content_bit_depth - 1)
    norm_style_img = style_img / (2**style_bit_depth - 1)

    log_content_img, log_style_img = log_imgs
    log_chroma_normal = global_content_isd
    log_chroma_offset = plane_offset

    # Compute bounds/xyz limits for log rgb.
    # Useful to see projections correctness when all log RGB plots share same limits.
    log_chroma_content_flat = log_chroma_content.reshape(-1, 3)
    log_content_flat = log_content_img.reshape(-1, 3)
    log_style_flat = log_style_img.reshape(-1, 3)
    tf_log_content_flat = tf_log_content.reshape(-1, 3)

    bounds = calculate_shared_limits(
        [
            log_style_flat,
            log_content_flat,
            log_chroma_content_flat,
            tf_log_content_flat,
        ],
        padding=0.2,
    )
    x_limits, y_limits, z_limits = bounds

    # Setting up axs
    fig = plt.figure(figsize=(20, 40))
    axs = dict()
    axs["style_img"] = fig.add_subplot(8, 2, 1)
    axs["content_img"] = fig.add_subplot(8, 2, 2)
    axs["style_rgb"] = fig.add_subplot(8, 2, 3, projection="3d")
    axs["content_rgb"] = fig.add_subplot(8, 2, 4, projection="3d")
    axs["style_log_rgb"] = fig.add_subplot(8, 2, 5, projection="3d")
    axs["content_log_rgb"] = fig.add_subplot(8, 2, 6, projection="3d")
    axs["mixed_rgb"] = fig.add_subplot(8, 2, 7, projection="3d")
    axs["mixed_log_rgb"] = fig.add_subplot(8, 2, 8, projection="3d")
    axs["content_projected_img"] = fig.add_subplot(8, 2, 9)
    axs["content_projected_log_rgb"] = fig.add_subplot(8, 2, 10, projection="3d")
    axs["clustered_content_log_rgb"] = fig.add_subplot(8, 2, 11, projection="3d")
    axs["tf_content_img"] = fig.add_subplot(8, 2, 13)
    axs["tf_content_log_rgb"] = fig.add_subplot(8, 2, 14, projection="3d")
    axs["mixed_tf_log_rgb"] = fig.add_subplot(8, 2, 15, projection="3d")

    # Make log RGB plots same limits, aspect ratio
    log_rgb_plots_idx = [
        "style_log_rgb",
        "content_log_rgb",
        "mixed_log_rgb",
        "content_projected_log_rgb",
        "clustered_content_log_rgb",
        "tf_content_log_rgb",
        "mixed_tf_log_rgb",
    ]
    for i in log_rgb_plots_idx:
        axs[i].set_box_aspect([1, 1, 1])
        axs[i].set_xlim(x_limits)
        axs[i].set_ylim(y_limits)
        axs[i].set_zlim(z_limits)

    # Plots
    # For plotting RGB/logRGB, sample points:
    num_samples = 5000
    log_chroma_content_flat = log_chroma_content.reshape(-1, 3)
    sample_indices = np.random.choice(
        len(log_chroma_content_flat), num_samples, replace=False
    )
    plot_img_rgb_logrgb(
        axs,
        norm_content_img,
        norm_style_img,
        log_content_img,
        log_style_img,
        sample_indices,
        bin_masks,
        dark_points,  # Uncomment if you want to see them plotted.
        bright_points,
    )
    plot_content_log_chroma(
        axs,
        log_chroma_content,
        content_bit_depth,
        norm_content_img,
        sample_indices,
    )
    plot_plane(
        [axs["content_log_rgb"], axs["content_projected_log_rgb"]],
        normal=log_chroma_normal,
        point=log_chroma_offset,
        bounds=bounds,
    )
    plot_transformed_img_logrgb(
        axs,
        tf_log_content,
        log_content_img,
        content_bit_depth,
        sample_indices,
    )

    # Make log RGB plots same view
    if view_isd:
        elev, azim = plane_view_from_normal(log_chroma_normal)
    else:
        elev = axs[log_rgb_plots_idx[-1]].elev
        azim = axs[log_rgb_plots_idx[-1]].azim
        elev = 30
        azim = -30

    for i in log_rgb_plots_idx:
        axs[i].view_init(elev, azim)

    plt.tight_layout()
    plt.show()

    return tf_log_content
