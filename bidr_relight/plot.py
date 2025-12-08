import matplotlib.pyplot as plt
import numpy as np

from bidr_relight.image_process import normalized_linear_to_srgb
import numpy as np


def plot_ax(
    pixels,
    colors,
    ax,
    title,
    axis_labels,
    point_size=1,
    alpha=0.3,
):
    """
    Scatter plot on existing 3D axis

    Args:
        pixels: (N, 3) array of RGB values
        colors: (N, 3) array for point colors (must be normalized [0,1])
    """
    ax.scatter(
        pixels[:, 0],
        pixels[:, 1],
        pixels[:, 2],
        c=colors,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax.set_xlabel(axis_labels[0], fontsize=10)
    ax.set_ylabel(axis_labels[1], fontsize=10)
    ax.set_zlabel(axis_labels[2], fontsize=10)
    ax.set_title(title, fontsize=12)


def plot_bin_masks(bin_mask: np.ndarray):
    fig = plt.figure(figsize=(6, 6), facecolor="gray")
    ax = fig.add_axes([0, 0, 1, 1])  # fill the entire figure
    ax.imshow(bin_mask, cmap="gray")
    ax.axis("off")  # remove axes and ticks
    plt.show()


def plane_view_from_normal(normal):
    """
    Compute a Matplotlib 3D view (elev, azim) for a plane normal vector.
    normal: array-like, shape (3,)
    Returns: elev, azim
    """
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # normalize

    nx, ny, nz = normal

    # Elevation (angle above xy-plane)
    elev = np.degrees(np.arcsin(nz))  # arcsin of z-component

    # Azimuth (rotation around z-axis)
    azim = np.degrees(np.arctan2(ny, nx))  # atan2(y, x)

    return elev, azim


def plot_content_log_chroma(
    axs,
    log_chroma_content,
    content_bit_depth,
    norm_content_img,
):
    """
    Plots log chromaticity image of the content image and the log RGB scatter plot for the log chromaticity.
    The content image is only used to color the log RGb scatter plot.
    """

    # Projected content image.
    linear_chroma = np.exp(log_chroma_content).astype(np.float32)
    max_linear_chroma = 2**content_bit_depth - 1
    img_clipped = np.clip(linear_chroma, 0, max_linear_chroma)
    img_normalized = (img_clipped / max_linear_chroma) * 255.0
    img_normalized = img_normalized.astype(np.uint8)
    assert img_normalized.min() >= 0 and img_normalized.max() <= 255 + 1e-3, (
        f"Image values out of expected range: {img_normalized.min()} to {img_normalized.max()}"
    )
    axs["content_projected_img"].imshow(img_normalized)
    axs["content_projected_img"].set_title(
        "Content's Log Chromaticity Image", fontsize=12
    )
    axs["content_projected_img"].axis("off")

    # Projected content log RGB:
    # Sample pixels for log-RGB plotting
    num_samples = 5000
    log_chroma_flat = log_chroma_content.reshape(-1, 3)
    content_flat = norm_content_img.reshape(-1, 3)
    if len(content_flat) > num_samples:
        indices = np.random.choice(len(content_flat), num_samples, replace=False)
        content_sampled = content_flat[indices]
        log_chroma_sampled = log_chroma_flat[indices]
    else:
        content_sampled = content_flat
        log_chroma_sampled = log_chroma_flat

    plot_ax(
        log_chroma_sampled,
        colors=content_sampled,
        ax=axs["content_projected_log_rgb"],
        title="Content's Log Chromaticity Normalized Log RGB",
        axis_labels=["Log(Red)", "Log(Green)", "Log(Blue)"],
        point_size=2,
        alpha=0.3,
    )


def plot_transformed_img_logrgb(
    axs,
    tf_log_img,
    bit_depth,
):
    # Plot 1. Transformed Image
    linear_img = np.exp(tf_log_img).astype(np.float32)
    norm_linear_img = linear_img / (2**bit_depth - 1)
    norm_linear_img = np.clip(norm_linear_img, 0.0, 1.0)
    img = normalized_linear_to_srgb(norm_linear_img)
    axs["tf_content_img"].imshow(img)
    axs["tf_content_img"].set_title("Transformed Content Image", fontsize=12)
    axs["tf_content_img"].axis("off")

    # Plot 2. Transformed Image LOGRGB
    # Sample pixels for log-RGB plotting
    num_samples = 5000
    content_flat = tf_log_img.reshape(-1, 3)
    color_flat = img.reshape(-1, 3) / 255.0
    if len(content_flat) > num_samples:
        indices = np.random.choice(len(content_flat), num_samples, replace=False)
        content_sampled = content_flat[indices]
        color_sampled = color_flat[indices]
    else:
        content_sampled = content_flat
        color_sampled = color_flat

    plot_ax(
        content_sampled,
        colors=color_sampled,
        ax=axs["tf_content_log_rgb"],
        title="Transformed Content Log RGB",
        axis_labels=["Log(Red)", "Log(Green)", "Log(Blue)"],
        point_size=2,
        alpha=0.3,
    )


def plot_img_rgb_logrgb(
    axs,
    norm_content_img,
    norm_style_img,
    log_norm_content_img,
    log_norm_style_img,
    log_cluster_bin_masks=None,
):
    # Row 1: Images
    # Convert linear to sRGB for visualization.
    content_srgb_img = normalized_linear_to_srgb(norm_content_img)
    style_srgb_img = normalized_linear_to_srgb(norm_style_img)

    axs["style_img"].imshow(style_srgb_img)
    axs["style_img"].set_title("Style", fontsize=12)
    axs["style_img"].axis("off")

    axs["content_img"].imshow(content_srgb_img)
    axs["content_img"].set_title("Content", fontsize=12)
    axs["content_img"].axis("off")

    # Sample pixels for RGB/logRGB plotting
    num_samples = 5000
    log_content_flat = log_norm_content_img.reshape(-1, 3)
    log_style_flat = log_norm_style_img.reshape(-1, 3)
    content_flat = norm_content_img.reshape(-1, 3)
    style_flat = norm_style_img.reshape(-1, 3)

    if len(log_content_flat) > num_samples:
        indices = np.random.choice(len(log_content_flat), num_samples, replace=False)
        log_content_sampled = log_content_flat[indices]
        log_style_sampled = log_style_flat[indices]
        content_sampled = content_flat[indices]
        style_sampled = style_flat[indices]
    else:
        log_content_sampled = log_content_flat
        log_style_sampled = log_style_flat
        content_sampled = content_flat
        style_sampled = style_flat

    # Row 2: RGB Space
    plot_ax(
        style_sampled,
        colors=style_sampled,
        ax=axs["style_rgb"],
        title="Style RGB Space",
        axis_labels=["R", "G", "B"],
        point_size=2,
        alpha=0.3,
    )
    plot_ax(
        content_sampled,
        colors=content_sampled,
        ax=axs["content_rgb"],
        title="Content RGB Space",
        axis_labels=["R", "G", "B"],
        point_size=2,
        alpha=0.3,
    )

    # Row 3: Log-RGB Space
    plot_ax(
        log_style_sampled,
        colors=style_sampled,
        ax=axs["style_log_rgb"],
        title="Style Log-RGB Space",
        axis_labels=["Log(R)", "Log(G)", "Log(B)"],
        point_size=2,
        alpha=0.3,
    )
    plot_ax(
        log_content_sampled,
        colors=content_sampled,
        ax=axs["content_log_rgb"],
        title="content Log-RGB Space",
        axis_labels=["Log(R)", "Log(G)", "Log(B)"],
        point_size=2,
        alpha=0.3,
    )

    # Row 4: overlayed RGB and log RGB.
    axs["mixed_rgb"].scatter(
        style_sampled[:, 0],
        style_sampled[:, 1],
        style_sampled[:, 2],
        c="green",
        s=2,
        alpha=0.2,
        label="Style",
    )
    axs["mixed_rgb"].scatter(
        content_sampled[:, 0],
        content_sampled[:, 1],
        content_sampled[:, 2],
        c="blue",
        s=2,
        alpha=0.2,
        label="Original",
    )
    axs["mixed_rgb"].set_xlabel("Red", fontsize=10)
    axs["mixed_rgb"].set_ylabel("Green", fontsize=10)
    axs["mixed_rgb"].set_zlabel("Blue", fontsize=10)
    axs["mixed_rgb"].set_title("RGB Comparison", fontsize=12)
    axs["mixed_rgb"].legend()

    axs["mixed_log_rgb"].scatter(
        log_style_sampled[:, 0],
        log_style_sampled[:, 1],
        log_style_sampled[:, 2],
        c="green",
        s=2,
        alpha=0.2,
        label="Style",
    )
    axs["mixed_log_rgb"].scatter(
        log_content_sampled[:, 0],
        log_content_sampled[:, 1],
        log_content_sampled[:, 2],
        c="blue",
        s=2,
        alpha=0.2,
        label="Original",
    )
    axs["mixed_log_rgb"].set_xlabel("log(Red)", fontsize=10)
    axs["mixed_log_rgb"].set_ylabel("log(Green)", fontsize=10)
    axs["mixed_log_rgb"].set_zlabel("log(Blue)", fontsize=10)
    axs["mixed_log_rgb"].set_title("Log-RGB Comparison", fontsize=12)
    axs["mixed_log_rgb"].legend()

    if log_cluster_bin_masks:
        # Cluster overlay
        num_clusters = len(log_cluster_bin_masks)
        cmap = plt.cm.get_cmap("tab20" if num_clusters <= 20 else "hsv", num_clusters)

        for i in range(num_clusters):
            bin_mask_flat = log_cluster_bin_masks[i].ravel()
            bin_mask_flat_sampled = bin_mask_flat[indices]
            axs["clustered_content_log_rgb"].scatter(
                log_content_sampled[bin_mask_flat_sampled, 0],
                log_content_sampled[bin_mask_flat_sampled, 1],
                log_content_sampled[bin_mask_flat_sampled, 2],
                c=[cmap(i)],
                s=2,
                alpha=0.2,
            )

        axs["clustered_content_log_rgb"].set_xlabel("log(Red)", fontsize=10)
        axs["clustered_content_log_rgb"].set_ylabel("log(Green)", fontsize=10)
        axs["clustered_content_log_rgb"].set_zlabel("log(Blue)", fontsize=10)
        axs["clustered_content_log_rgb"].set_title(
            "Content Log-RGB Clustered", fontsize=12
        )


def plot_plane(axs, normal, point, bounds, alpha=0.3, color="red"):
    """
    Plot a plane perpendicular to a normal vector passing through a point.
    """
    # Normalize the normal vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    point = np.array(point)

    # Plane equation: a(x-x0) + b(y-y0) + c(z-z0) = 0
    a, b, c = normal
    x0, y0, z0 = point

    # Use provided bounds
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds

    # Choose which variable to solve for based on largest normal component
    if abs(c) > abs(a) and abs(c) > abs(b):
        # Solve for z
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X, Y = np.meshgrid(x, y)
        Z = z0 - (a * (X - x0) + b * (Y - y0)) / c
    elif abs(b) > abs(a):
        # Solve for y
        x = np.linspace(x_min, x_max, 20)
        z = np.linspace(z_min, z_max, 20)
        X, Z = np.meshgrid(x, z)
        Y = y0 - (a * (X - x0) + c * (Z - z0)) / b
    else:
        # Solve for x
        y = np.linspace(y_min, y_max, 20)
        z = np.linspace(z_min, z_max, 20)
        Y, Z = np.meshgrid(y, z)
        X = x0 - (b * (Y - y0) + c * (Z - z0)) / a

    for ax in axs:
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color)


def calculate_shared_limits(data_arrays, padding=0.1):
    """
    Calculate shared axis limits from multiple datasets.

    Args:
        data_arrays: List of (N, 3) arrays or single (N, 3) array
        padding: Padding fraction to add to bounds (default 0.1 = 10%)

    Returns:
        tuple: (x_limits, y_limits, z_limits) where each is [min, max]
    """
    # Handle single array or list of arrays
    if isinstance(data_arrays, list):
        all_data = np.vstack(data_arrays)
    else:
        all_data = data_arrays

    # Calculate min/max for each dimension
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()

    # Calculate ranges
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Add padding
    x_limits = [x_min - padding * x_range, x_max + padding * x_range]
    y_limits = [y_min - padding * y_range, y_max + padding * y_range]
    z_limits = [z_min - padding * z_range, z_max + padding * z_range]

    return x_limits, y_limits, z_limits
    plt.tight_layout()
    plt.show()


def plot_log_chroma_plane_pre_clustering(
    log_chroma_content, isd_map, content_img, content_bit_depth
):
    """
    Visualize the 2D log chromaticity plane before clustering.
    Shows pixel distribution colored by their original RGB values.

    Args:
        log_chroma_content: (H, W, 3) log chromaticity projection
        isd_map: (H, W, 3) ISD direction vectors
        content_img: (H, W, 3) original linear RGB image
        content_bit_depth: bit depth for normalization
    """
    H, W, _ = log_chroma_content.shape

    # Flatten spatial dimensions
    log_chroma_flat = log_chroma_content.reshape(H * W, 3)

    # Sample for visualization (too many points slow down plotting)
    num_samples = 200000
    if len(log_chroma_flat) > num_samples:
        indices = np.random.choice(len(log_chroma_flat), num_samples, replace=False)
        sampled_chroma = log_chroma_flat[indices]

        # Get corresponding RGB colors for those pixels
        content_flat = content_img.reshape(H * W, 3)
        sampled_colors = content_flat[indices]
    else:
        sampled_chroma = log_chroma_flat
        sampled_colors = content_img.reshape(H * W, 3)

    # Normalize colors to [0, 1] for display
    norm_colors = sampled_colors / (2**content_bit_depth - 1)
    norm_colors = np.clip(norm_colors, 0, 1)

    # Project onto 2D plane perpendicular to mean ISD
    mean_isd = isd_map.reshape(H * W, 3).mean(axis=0)
    mean_isd = mean_isd / np.linalg.norm(mean_isd)

    # Create orthonormal basis for the plane
    # Pick arbitrary vector not parallel to ISD
    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(mean_isd[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = arbitrary - np.dot(arbitrary, mean_isd) * mean_isd
    u = u / np.linalg.norm(u)
    v = np.cross(mean_isd, u)

    # Project sampled points onto 2D basis
    coords_2d = np.zeros((len(sampled_chroma), 2))
    coords_2d[:, 0] = np.dot(sampled_chroma, u)
    coords_2d[:, 1] = np.dot(sampled_chroma, v)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=norm_colors, s=5, alpha=0.6, rasterized=True
    )

    ax.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax.set_title(
        "Log Chromaticity Plane (Pre-Clustering)\nColored by Original RGB", fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def plot_log_chroma_plane_post_clustering(
    log_chroma_content, isd_map, bin_masks, bin_radius
):
    """
    Visualize the 2D log chromaticity plane after clustering.
    Shows pixel distribution colored by cluster ID.

    Args:
        log_chroma_content: (H, W, 3) log chromaticity projection
        isd_map: (H, W, 3) ISD direction vectors
        bin_masks: list of (H, W) boolean masks for each bin
        bin_radius: clustering radius used
    """
    H, W, _ = log_chroma_content.shape

    # Flatten spatial dimensions
    log_chroma_flat = log_chroma_content.reshape(H * W, 3)

    # Create cluster ID map
    cluster_ids = np.zeros(H * W, dtype=int)
    for bin_id, mask in enumerate(bin_masks, start=1):
        cluster_ids[mask.ravel()] = bin_id

    # Sample for visualization
    num_samples = 200000
    if len(log_chroma_flat) > num_samples:
        indices = np.random.choice(len(log_chroma_flat), num_samples, replace=False)
        sampled_chroma = log_chroma_flat[indices]
        sampled_clusters = cluster_ids[indices]
    else:
        sampled_chroma = log_chroma_flat
        sampled_clusters = cluster_ids

    # Project onto 2D plane perpendicular to mean ISD
    mean_isd = isd_map.reshape(H * W, 3).mean(axis=0)
    mean_isd = mean_isd / np.linalg.norm(mean_isd)

    # Create orthonormal basis for the plane
    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(mean_isd[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = arbitrary - np.dot(arbitrary, mean_isd) * mean_isd
    u = u / np.linalg.norm(u)
    v = np.cross(mean_isd, u)

    # Project sampled points onto 2D basis
    coords_2d = np.zeros((len(sampled_chroma), 2))
    coords_2d[:, 0] = np.dot(sampled_chroma, u)
    coords_2d[:, 1] = np.dot(sampled_chroma, v)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a colormap that shows distinct clusters
    num_clusters = len(bin_masks)
    cmap = plt.cm.get_cmap("tab20" if num_clusters <= 20 else "hsv", num_clusters)

    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=sampled_clusters,
        cmap=cmap,
        s=5,
        alpha=0.6,
        rasterized=True,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster ID", fontsize=12)

    ax.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax.set_title(
        f"Log Chromaticity Plane (Post-Clustering)\n{num_clusters} clusters with radius={bin_radius}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def plot_cluster_spatial_distribution(bin_masks, content_img, content_bit_depth):
    """
    Show spatial distribution of clusters overlaid on the original image.

    Args:
        bin_masks: list of (H, W) boolean masks for each bin
        content_img: (H, W, 3) original linear RGB image
        content_bit_depth: bit depth for normalization
    """
    from bidr_relight.image_process import normalized_linear_to_srgb

    H, W, _ = content_img.shape
    num_clusters = len(bin_masks)

    # Create cluster ID image
    cluster_img = np.zeros((H, W), dtype=int)
    for bin_id, mask in enumerate(bin_masks, start=1):
        cluster_img[mask] = bin_id

    # Normalize content image
    norm_content = content_img / (2**content_bit_depth - 1)
    norm_content = np.clip(norm_content, 0, 1)
    srgb_content = normalized_linear_to_srgb(norm_content)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    ax1.imshow(srgb_content)
    ax1.set_title("Original Content Image", fontsize=14)
    ax1.axis("off")

    # Cluster overlay
    cmap = plt.cm.get_cmap("tab20" if num_clusters <= 20 else "hsv", num_clusters)
    im = ax2.imshow(cluster_img, cmap=cmap)
    ax2.set_title(f"Material Clusters ({num_clusters} clusters)", fontsize=14)
    ax2.axis("off")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Cluster ID", fontsize=12)

    plt.tight_layout()
    plt.show()
