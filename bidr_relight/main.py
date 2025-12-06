import numpy as np
import torch
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import os
import logging

from bidr_relight.isd_estimator.mock import MockISDModel
from bidr_relight.isd_estimator.unet import ResNet50UNet

from bidr_relight.image_process import (
    resize_with_same_aspect,
    linear_to_log,
    normalized_linear_to_srgb,
)
from bidr_relight.bidr_process import (
    project_to_log_chromaticity_plane,
    get_global_isd,
)

# from bidr_relight.illuminant_estimation import RecursiveRetinex

from bidr_relight.plot import (
    plot_img_rgb_logrgb,
    plot_content_log_chroma,
    plot_plane,
    calculate_shared_limits,
    plane_view_from_normal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def relight_content_image(
    content_path,
    style_path,
    isd_model,
    isd_model_path,
    output_path,
    resize_scale=1 / 4,
    bin_radius=1.0,
    shading_only=False,
    compression_factor=0.7,
    view_isd=False,
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
    if isd_model == "unet":
        model = ResNet50UNet(
            in_channels=3,
            out_channels=3,
            pretrained=True,
            checkpoint=isd_model_path,
            se_block=True,
            dropout=0.0,
        )

    elif isd_model == "vit":
        # TODO
        pass
    else:
        model = MockISDModel()
    model.eval()

    # --- 1. Load and preprocess images ---
    img_paths = [content_path, style_path]
    imgs = []
    imgs_bit_depth = []
    log_imgs = []
    log_norm_imgs = []

    for i in range(len(img_paths)):
        img = imread(img_paths[i])
        img_bit_depth = np.iinfo(img.dtype).bits
        img = resize_with_same_aspect(img, scale=resize_scale)

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

    # Iterate through each unassigned pixel, and assign this and other unassigned pixels into a new group/bin by proximity in log chroma plane.
    H, W, _ = log_chroma_content.shape
    log_chroma_content_flat = log_chroma_content.reshape(H * W, 3)
    bin_map = np.zeros(H * W)
    bin_masks = []  # boolean mask of pixels for each bin
    UNASSIGNED = 0
    bin_id = 0

    for i in range(len(bin_map)):
        if bin_map[i] != UNASSIGNED:
            continue

        # Compute distance between this unassigned px and others in log chroma plane
        dist_to_other_pts = np.linalg.norm(
            log_chroma_content_flat - log_chroma_content_flat[i], axis=1
        )  # (HW, )

        # Assign nearby unassigned (including this pixel) to a new bin
        bin_id += 1
        close_mask = dist_to_other_pts < bin_radius
        unassigned_mask = bin_map == UNASSIGNED
        new_bin_mask = np.logical_and(close_mask, unassigned_mask)
        bin_map[new_bin_mask] = bin_id

        bin_masks.append(new_bin_mask.reshape(H, W))

    assert np.all(bin_map != UNASSIGNED)
    logger.info(
        f"Clustered pixels into {bin_id} bins in log chromaticity plane by proximity ({bin_radius=})"
    )

    # --- 4. Find the global illumination vector of content image. ---
    # Details:
    # Under the assumption of uniform spectral ratio (i.e., same ambient and direct),
    # each material's vector between fully lit and fully dark in log RGB will have the same direction (ISD)
    # and same norm. We will denote this as "illumination vector" (referred as N in BIDR paper) and estimate this
    # as the rightmost mode of length distribution.

    # Compute signed dist along isd for each pixel.
    diff_vec = log_imgs[CONTENT] - log_chroma_content
    signed_dist_map = (diff_vec * isd_map).sum(axis=2)  # dot product into (H,W)

    # Get the 5th and 95th percentile of signed dist distribution for each bin of pixels.
    lengths = []
    for bin_mask in bin_masks:
        signed_dists = signed_dist_map[bin_mask].ravel()

        p5 = np.percentile(signed_dists, 5)
        p95 = np.percentile(signed_dists, 95)

        length = p95 - p5
        lengths.append(length)

    # Create a histogram from this length array
    bin_counts, bin_edges = np.histogram(np.array(lengths))
    bin_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Use center as their position

    # Extract peaks/modes from this histogram.
    # Modes are defined as those histogram bins with relatively high counts.
    # The count threshold is dynamically set to 30% of max count.
    count_threshold = 0.3 * bin_counts.max()
    mode_counts = bin_counts[bin_counts > count_threshold]
    mode_x = bin_x[bin_counts > count_threshold]

    # Use the rightmost mode as the illum vector norm.
    illum_vector_norm = mode_x[np.argmax(mode_counts)]
    logger.info(f"Estimated illumination vector norm {illum_vector_norm}")

    # --- 5. Estimate fully (dark, bright) pairs for each material. ---
    # To deal with material with only dark or bright pixels:
    # Its opposite point is estimated using global illum vector norm and its ISD.
    # TODO...

    # --- 6. Pivot each material around their dark point from content ISD to the average style ISD. ---
    global_style_isd = get_global_isd(isd_maps[STYLE])
    # TODO...

    # --- 7. Plots: log chroma, illum norm distribution, sRGB, logRGB. ---
    # TODO: Missing some plotting codes

    # Prepare data for plotting
    content_img, style_img = imgs
    content_bit_depth, style_bit_depth = imgs_bit_depth
    norm_content_img = content_img / (2**content_bit_depth - 1)
    norm_style_img = style_img / (2**style_bit_depth - 1)

    log_content_img, log_style_img = log_imgs
    log_chroma_normal = get_global_isd(isd_maps[CONTENT])
    log_chroma_offset = plane_offset

    # Compute bounds/xyz limits for log rgb.
    # Useful to see projections correctness when all log RGB plots share same limits.
    log_chroma_content_flat = log_chroma_content.reshape(-1, 3)
    log_content_flat = log_content_img.reshape(-1, 3)
    log_style_flat = log_style_img.reshape(-1, 3)
    bounds = calculate_shared_limits(
        [
            log_style_flat,
            log_content_flat,
            log_chroma_content_flat,
        ],
        padding=0.2,
    )
    x_limits, y_limits, z_limits = bounds

    # Setting up axs
    fig = plt.figure(figsize=(20, 20))
    axs = dict()
    axs["style_img"] = fig.add_subplot(5, 2, 1)
    axs["content_img"] = fig.add_subplot(5, 2, 2)
    axs["style_rgb"] = fig.add_subplot(5, 2, 3, projection="3d")
    axs["content_rgb"] = fig.add_subplot(5, 2, 4, projection="3d")
    axs["style_log_rgb"] = fig.add_subplot(5, 2, 5, projection="3d")
    axs["content_log_rgb"] = fig.add_subplot(5, 2, 6, projection="3d")
    axs["mixed_rgb"] = fig.add_subplot(5, 2, 7, projection="3d")
    axs["mixed_log_rgb"] = fig.add_subplot(5, 2, 8, projection="3d")
    axs["content_projected_img"] = fig.add_subplot(5, 2, 9)
    axs["content_projected_log_rgb"] = fig.add_subplot(5, 2, 10, projection="3d")

    # Make log RGB plots same limits, aspect ratio
    log_rgb_plots_idx = [
        "style_log_rgb",
        "content_log_rgb",
        "mixed_log_rgb",
        "content_projected_log_rgb",
    ]
    for i in log_rgb_plots_idx:
        axs[i].set_box_aspect([1, 1, 1])
        axs[i].set_xlim(x_limits)
        axs[i].set_ylim(y_limits)
        axs[i].set_zlim(z_limits)

    # Plots
    plot_img_rgb_logrgb(
        axs, norm_content_img, norm_style_img, log_content_img, log_style_img
    )
    plot_content_log_chroma(
        axs, log_chroma_content, content_bit_depth, norm_content_img
    )
    plot_plane(
        [axs["content_log_rgb"], axs["content_projected_log_rgb"]],
        normal=log_chroma_normal,
        point=log_chroma_offset,
        bounds=bounds,
    )

    # Make log RGB plots same view
    if view_isd:
        elev, azim = plane_view_from_normal(log_chroma_normal)
    else:
        elev = axs[log_rgb_plots_idx[-1]].elev
        azim = axs[log_rgb_plots_idx[-1]].azim

    for i in log_rgb_plots_idx:
        axs[i].view_init(elev, azim)

    plt.tight_layout()
    plt.show()

    # TODO (DEBUG): im only returning these for debug. remove later
    return log_chroma_content, log_imgs, isd_maps, imgs

    # # OLD CODE FOR DARKENING AND ILLUMINANT ESTIMATE.
    # I guess darkening could be useful, but maybe include this later.
    # ILLUMINANT estimate idk what this achieve.
    # # # --- 3. Estimate direct illuminant chromaticity ---
    # # if not shading_only:
    # #     epsilon = 1e-6
    # #     L_content = RecursiveRetinex(content)
    # #     L_style = RecursiveRetinex(style)
    # #     log_L_content = np.log(L_content + epsilon)
    # #     log_L_style = np.log(L_style + epsilon)
    # #     delta_illuminant = log_L_style - log_L_content
    # # else:
    # #     delta_illuminant = np.zeros(3, dtype=np.float32)
    # #
    # # --- 5. Vectorized transformation ---
    # # FIX: Compression in log space
    # # To reduce brightness by compression_factor, add log(compression_factor) along ISD
    # # log_compression = np.log(compression_factor)
    # #
    # # # Move along ISD direction (ISD points dark→bright, so negative = darker)
    # # transformed_log = (
    # #     log_content + log_compression * predicted_isd_map + delta_illuminant
    # # )
