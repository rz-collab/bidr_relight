# BIDR Relight
# 12/11/25
# CS7180 Advanced Perception
# Contributors: Max Huber, Adharsh Kandula, Richard Zhao

# This file contains plotting functions for visualizing images, color spaces, ISD,
# and clustering.
# Several components were coded/modified with the help of GPT-5 and Claude Sonnet 4.5



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D

from src.image_util import normalized_linear_to_srgb


# ============================================================================
# 3D Scatter Plotting
# ============================================================================


def plot_3d_scatter(pixels, colors, ax, title, axis_labels, point_size=1, alpha=0.3):
    """Unified 3D scatter plot on existing axis."""
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


def plot_rgb_space(
    pixels, colors, title="RGB Space", save_path=None, alpha=0.3, point_size=1
):
    """Plot RGB space with origin marker."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_3d_scatter(pixels, colors, ax, title, ["R", "G", "B"], point_size, alpha)
    ax.scatter(
        [0], [0], [0], c="black", s=100, marker="o", label="Origin", depthshade=False
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.tight_layout()
    _save_or_show(save_path)


def plot_log_rgb_space(
    pixels, colors, title="Log RGB Space", save_path=None, alpha=0.3, point_size=1
):
    """Plot log-RGB space."""
    log_pixels = np.log(pixels + 1e-6)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_3d_scatter(
        log_pixels, colors, ax, title, ["log(R)", "log(G)", "log(B)"], point_size, alpha
    )
    plt.tight_layout()
    _save_or_show(save_path)


# ============================================================================
# Image & Color Space Visualization
# ============================================================================


def plot_img_rgb_logrgb(
    axs,
    norm_content_img,
    norm_style_img,
    log_content_img,
    log_style_img,
    log_cluster_bin_masks=None,
    log_cluster_dark_points=None,
    log_cluster_bright_points=None,
):
    """Complete image + RGB + log-RGB visualization."""
    content_srgb = normalized_linear_to_srgb(norm_content_img)
    style_srgb = normalized_linear_to_srgb(norm_style_img)

    # Images
    axs["style_img"].imshow(style_srgb)
    axs["style_img"].set_title("Style", fontsize=12)
    axs["style_img"].axis("off")

    axs["content_img"].imshow(content_srgb)
    axs["content_img"].set_title("Content", fontsize=12)
    axs["content_img"].axis("off")

    # Sample pixels
    num_samples = 5000
    log_content_flat = log_content_img.reshape(-1, 3)
    log_style_flat = log_style_img.reshape(-1, 3)
    content_flat = norm_content_img.reshape(-1, 3)
    style_flat = norm_style_img.reshape(-1, 3)
    content_color_flat = content_srgb.reshape(-1, 3) / 255.0
    style_color_flat = style_srgb.reshape(-1, 3) / 255.0

    if len(log_content_flat) > num_samples:
        indices = np.random.choice(len(log_content_flat), num_samples, replace=False)
        log_content_sampled = log_content_flat[indices]
        log_style_sampled = log_style_flat[indices]
        content_sampled = content_flat[indices]
        style_sampled = style_flat[indices]
        content_color_sampled = content_color_flat[indices]
        style_color_sampled = style_color_flat[indices]
    else:
        log_content_sampled = log_content_flat
        log_style_sampled = log_style_flat
        content_sampled = content_flat
        style_sampled = style_flat
        content_color_sampled = content_color_flat
        style_color_sampled = style_color_flat

    # RGB Space
    plot_3d_scatter(
        style_sampled,
        style_color_sampled,
        axs["style_rgb"],
        "Style RGB Space",
        ["R", "G", "B"],
        2,
        0.3,
    )
    plot_3d_scatter(
        content_sampled,
        content_color_sampled,
        axs["content_rgb"],
        "Content RGB Space",
        ["R", "G", "B"],
        2,
        0.3,
    )

    # Log-RGB Space
    plot_3d_scatter(
        log_style_sampled,
        style_color_sampled,
        axs["style_log_rgb"],
        "Style Log-RGB Space",
        ["Log(R)", "Log(G)", "Log(B)"],
        2,
        0.3,
    )
    plot_3d_scatter(
        log_content_sampled,
        content_color_sampled,
        axs["content_log_rgb"],
        "Content Log-RGB Space",
        ["Log(R)", "Log(G)", "Log(B)"],
        2,
        0.3,
    )

    # Overlays
    _plot_overlay(
        axs["mixed_rgb"],
        style_sampled,
        content_sampled,
        ["Red", "Green", "Blue"],
        "RGB Comparison",
    )
    _plot_overlay(
        axs["mixed_log_rgb"],
        log_style_sampled,
        log_content_sampled,
        ["log(Red)", "log(Green)", "log(Blue)"],
        "Log-RGB Comparison",
    )

    # Clustering visualization
    if log_cluster_bin_masks:
        _plot_clusters(
            axs["clustered_content_log_rgb"],
            log_content_sampled,
            log_cluster_bin_masks,
            log_cluster_dark_points,
            log_cluster_bright_points,
            indices,
        )


def plot_content_log_chroma(
    axs, log_chroma_content, content_bit_depth, norm_content_img
):
    """Plot log chromaticity image and scatter."""
    # Project to linear and clip
    linear_chroma = np.exp(log_chroma_content).astype(np.float32)
    max_val = 2**content_bit_depth - 1
    img_normalized = np.clip(linear_chroma / max_val * 255.0, 0, 255).astype(np.uint8)

    axs["content_projected_img"].imshow(img_normalized)
    axs["content_projected_img"].set_title(
        "Content's Log Chromaticity Image", fontsize=12
    )
    axs["content_projected_img"].axis("off")

    # Sample and plot
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

    plot_3d_scatter(
        log_chroma_sampled,
        content_sampled,
        axs["content_projected_log_rgb"],
        "Content's Log Chromaticity Normalized Log RGB",
        ["Log(Red)", "Log(Green)", "Log(Blue)"],
        2,
        0.3,
    )


def plot_transformed_img_logrgb(axs, tf_log_img, log_img, bit_depth):
    """Plot transformed image and its log-RGB."""
    # Convert to sRGB
    linear_img = np.exp(tf_log_img).astype(np.float32)
    norm_linear = np.clip(linear_img / (2**bit_depth - 1), 0.0, 1.0)
    img = normalized_linear_to_srgb(norm_linear)

    axs["tf_content_img"].imshow(img)
    axs["tf_content_img"].set_title("Transformed Content Image", fontsize=12)
    axs["tf_content_img"].axis("off")

    # Sample
    num_samples = 5000
    tf_log_flat = tf_log_img.reshape(-1, 3)
    color_flat = img.reshape(-1, 3) / 255.0
    log_flat = log_img.reshape(-1, 3)

    if len(tf_log_flat) > num_samples:
        indices = np.random.choice(len(tf_log_flat), num_samples, replace=False)
        tf_log_sampled = tf_log_flat[indices]
        color_sampled = color_flat[indices]
        log_sampled = log_flat[indices]
    else:
        tf_log_sampled = tf_log_flat
        color_sampled = color_flat
        log_sampled = log_flat

    # Transformed log-RGB
    plot_3d_scatter(
        tf_log_sampled,
        color_sampled,
        axs["tf_content_log_rgb"],
        "Transformed Content Log RGB",
        ["Log(Red)", "Log(Green)", "Log(Blue)"],
        2,
        0.3,
    )

    # Overlay comparison
    axs["mixed_tf_log_rgb"].scatter(
        tf_log_sampled[:, 0],
        tf_log_sampled[:, 1],
        tf_log_sampled[:, 2],
        c="green",
        s=2,
        alpha=0.2,
        label="Transformed Content",
    )
    axs["mixed_tf_log_rgb"].scatter(
        log_sampled[:, 0],
        log_sampled[:, 1],
        log_sampled[:, 2],
        c="blue",
        s=2,
        alpha=0.2,
        label="Original Content",
    )
    axs["mixed_tf_log_rgb"].set_xlabel("log(Red)", fontsize=10)
    axs["mixed_tf_log_rgb"].set_ylabel("log(Green)", fontsize=10)
    axs["mixed_tf_log_rgb"].set_zlabel("log(Blue)", fontsize=10)
    axs["mixed_tf_log_rgb"].set_title("Log-RGB Comparison", fontsize=12)
    axs["mixed_tf_log_rgb"].legend()


# ============================================================================
# Chromaticity & Clustering
# ============================================================================


def plot_log_chroma_plane_pre_clustering(
    log_chroma_content, isd_map, content_img, content_bit_depth, sample_indices
):
    """2D log chromaticity plane before clustering."""
    H, W, _ = log_chroma_content.shape
    log_chroma_flat = log_chroma_content.reshape(H * W, 3)

    # Sample
    sampled_chroma = log_chroma_flat[sample_indices]
    content_flat = content_img.reshape(H * W, 3)
    sampled_colors = content_flat[sample_indices]

    norm_colors = np.clip(sampled_colors / (2**content_bit_depth - 1), 0, 1)
    norm_colors = normalized_linear_to_srgb(norm_colors) / 255.0

    # Project to 2D
    mean_isd = isd_map.reshape(H * W, 3).mean(axis=0)
    mean_isd = mean_isd / np.linalg.norm(mean_isd)

    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(mean_isd[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = arbitrary - np.dot(arbitrary, mean_isd) * mean_isd
    u = u / np.linalg.norm(u)
    v = np.cross(mean_isd, u)

    coords_2d = np.zeros((len(sampled_chroma), 2))
    coords_2d[:, 0] = np.dot(sampled_chroma, u)
    coords_2d[:, 1] = np.dot(sampled_chroma, v)

    # Projected RGB
    projected_3d = coords_2d[:, 0:1] * u + coords_2d[:, 1:2] * v
    projected_rgb = np.clip(
        np.exp(projected_3d) / np.max(np.exp(projected_3d), axis=0, keepdims=True), 0, 1
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=norm_colors, s=5, alpha=0.6, rasterized=True
    )
    ax1.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax1.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax1.set_title("Colored by Original RGB Values", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    ax2.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=projected_rgb,
        s=5,
        alpha=0.6,
        rasterized=True,
    )
    ax2.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax2.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax2.set_title("Colored by Projected Chromaticity", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    fig.suptitle("Log Chromaticity Plane (Pre-Clustering)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_log_chroma_plane_post_clustering(
    log_chroma_content, isd_map, bin_masks, bin_radius, sample_indicess
):
    """2D log chromaticity plane after clustering."""
    H, W, _ = log_chroma_content.shape
    log_chroma_flat = log_chroma_content.reshape(H * W, 3)

    cluster_ids = np.zeros(H * W, dtype=int)
    for bin_id, mask in enumerate(bin_masks, start=1):
        cluster_ids[mask.ravel()] = bin_id

    # Sample
    sampled_chroma = log_chroma_flat[sample_indicess]
    sampled_clusters = cluster_ids[sample_indicess]

    # Project to 2D
    mean_isd = isd_map.reshape(H * W, 3).mean(axis=0)
    mean_isd = mean_isd / np.linalg.norm(mean_isd)

    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(mean_isd[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = arbitrary - np.dot(arbitrary, mean_isd) * mean_isd
    u = u / np.linalg.norm(u)
    v = np.cross(mean_isd, u)

    coords_2d = np.zeros((len(sampled_chroma), 2))
    coords_2d[:, 0] = np.dot(sampled_chroma, u)
    coords_2d[:, 1] = np.dot(sampled_chroma, v)

    fig, ax = plt.subplots(figsize=(12, 10))

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
    """Show spatial distribution of clusters."""
    H, W, _ = content_img.shape
    cluster_img = np.zeros((H, W), dtype=int)
    for bin_id, mask in enumerate(bin_masks, start=1):
        cluster_img[mask] = bin_id

    norm_content = np.clip(content_img / (2**content_bit_depth - 1), 0, 1)
    srgb_content = normalized_linear_to_srgb(norm_content)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(srgb_content)
    ax1.set_title("Original Content Image", fontsize=14)
    ax1.axis("off")

    num_clusters = len(bin_masks)
    cmap = plt.cm.get_cmap("tab20" if num_clusters <= 20 else "hsv", num_clusters)
    im = ax2.imshow(cluster_img, cmap=cmap)
    ax2.set_title(f"Material Clusters ({num_clusters} clusters)", fontsize=14)
    ax2.axis("off")

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Cluster ID", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_log_chroma_plane_posterized(
    log_chroma_original,
    log_chroma_posterized,
    isd_map,
    content_img,
    content_bit_depth,
    levels,
):
    """Compare original vs posterized on 2D chromaticity plane."""
    H, W, _ = log_chroma_original.shape

    # Sample
    num_samples = 100000
    orig_flat = log_chroma_original.reshape(H * W, 3)
    post_flat = log_chroma_posterized.reshape(H * W, 3)
    color_flat = content_img.reshape(H * W, 3)

    if len(orig_flat) > num_samples:
        indices = np.random.choice(len(orig_flat), num_samples, replace=False)
        orig_sampled = orig_flat[indices]
        post_sampled = post_flat[indices]
        color_sampled = color_flat[indices]
    else:
        orig_sampled = orig_flat
        post_sampled = post_flat
        color_sampled = color_flat

    norm_colors = np.clip(color_sampled / (2**content_bit_depth - 1), 0, 1)
    norm_colors = normalized_linear_to_srgb(norm_colors) / 255.0

    # Project to 2D
    mean_isd = isd_map.reshape(H * W, 3).mean(axis=0)
    mean_isd = mean_isd / np.linalg.norm(mean_isd)

    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(mean_isd[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = arbitrary - np.dot(arbitrary, mean_isd) * mean_isd
    u = u / np.linalg.norm(u)
    v = np.cross(mean_isd, u)

    # Project both versions
    orig_2d = np.zeros((len(orig_sampled), 2))
    orig_2d[:, 0] = np.dot(orig_sampled, u)
    orig_2d[:, 1] = np.dot(orig_sampled, v)

    post_2d = np.zeros((len(post_sampled), 2))
    post_2d[:, 0] = np.dot(post_sampled, u)
    post_2d[:, 1] = np.dot(post_sampled, v)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # Original
    ax1.scatter(
        orig_2d[:, 0], orig_2d[:, 1], c=norm_colors, s=3, alpha=0.5, rasterized=True
    )
    ax1.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax1.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax1.set_title("Original", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Posterized
    ax2.scatter(
        post_2d[:, 0], post_2d[:, 1], c=norm_colors, s=3, alpha=0.5, rasterized=True
    )
    ax2.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax2.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax2.set_title(f"Posterized ({levels} levels)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    # Overlay comparison
    ax3.scatter(
        orig_2d[:, 0],
        orig_2d[:, 1],
        c="blue",
        s=2,
        alpha=0.3,
        label="Original",
        rasterized=True,
    )
    ax3.scatter(
        post_2d[:, 0],
        post_2d[:, 1],
        c="red",
        s=2,
        alpha=0.3,
        label="Posterized",
        rasterized=True,
    )
    ax3.set_xlabel("Chromaticity Dimension 1", fontsize=12)
    ax3.set_ylabel("Chromaticity Dimension 2", fontsize=12)
    ax3.set_title("Overlay Comparison", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Effect of Posterization on Log Chromaticity Plane", fontsize=16, y=1.02
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# ISD Visualization
# ============================================================================


def visualize_isd_as_direction(isd_map, save_path="isd_direction.png"):
    """Color-coded ISD direction using spherical coordinates."""
    C, H, W = isd_map.shape
    x, y, z = isd_map[0], isd_map[1], isd_map[2]

    azimuth = np.arctan2(y, x)
    hue = (azimuth + np.pi) / (2 * np.pi)

    elevation = np.arcsin(np.clip(z, -1, 1))
    saturation = (elevation + np.pi / 2) / np.pi

    value = np.ones_like(hue)
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.title("ISD Direction Map (Color = Direction in 3D space)")
    plt.colorbar(label="Direction")
    plt.axis("off")
    plt.tight_layout()
    _save_or_show(save_path)
    return rgb


def visualize_isd_components(isd_map, save_path="isd_components.png"):
    """Visualize R, G, B components separately."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["ISD_R", "ISD_G", "ISD_B"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        component = isd_map[i]
        vmin, vmax = component.min(), component.max()
        normalized = (component - vmin) / (vmax - vmin + 1e-8)

        im = ax.imshow(normalized, cmap="RdBu_r")
        ax.set_title(label)
        ax.axis("off")
        plt.colorbar(im, ax=ax, label=f"Range: [{vmin:.3f}, {vmax:.3f}]")

    plt.tight_layout()
    _save_or_show(save_path)


def visualize_isd_magnitude(isd_map, save_path="isd_magnitude.png"):
    """Check ISD magnitude (should be ~1 if normalized)."""
    magnitude = np.sqrt((isd_map**2).sum(axis=0))

    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, cmap="viridis")
    plt.colorbar(label="ISD Magnitude")
    plt.title(
        f"ISD Magnitude (should ≈1.0)\nMean: {magnitude.mean():.4f}, Std: {magnitude.std():.4f}"
    )
    plt.axis("off")
    plt.tight_layout()
    _save_or_show(save_path)

    if abs(magnitude.mean() - 1.0) > 0.1:
        print("⚠️  Warning: ISD vectors may not be properly normalized!")


def visualize_isd_arrow_field(
    isd_map, original_img, stride=50, save_path="isd_arrows.png"
):
    """Overlay ISD as arrow field."""
    if original_img.shape[0] == 3:
        original_img = np.transpose(original_img, (1, 2, 0))

    H, W = original_img.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(original_img)

    y_coords = np.arange(stride // 2, H, stride)
    x_coords = np.arange(stride // 2, W, stride)

    for y in y_coords:
        for x in x_coords:
            isd_vec = isd_map[:, y, x]
            dx = isd_vec[0] * stride * 0.4
            dy = isd_vec[1] * stride * 0.4
            b_component = isd_vec[2]
            color = plt.cm.RdBu_r((b_component + 1) / 2)

            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=stride * 0.15,
                head_length=stride * 0.15,
                fc=color,
                ec=color,
                alpha=0.7,
                width=1.5,
            )

    ax.set_title("ISD Vector Field")
    ax.axis("off")
    plt.tight_layout()
    _save_or_show(save_path)


# ============================================================================
# Utilities
# ============================================================================


def plot_bin_masks(bin_mask: np.ndarray):
    """Simple binary mask visualization."""
    fig = plt.figure(figsize=(6, 6), facecolor="gray")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(bin_mask, cmap="gray")
    ax.axis("off")
    plt.show()


def plot_plane(axs, normal, point, bounds, alpha=0.3, color="red"):
    """Plot plane perpendicular to normal through point."""
    normal = np.array(normal) / np.linalg.norm(normal)
    point = np.array(point)
    a, b, c = normal
    x0, y0, z0 = point

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds

    if abs(c) > abs(a) and abs(c) > abs(b):
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X, Y = np.meshgrid(x, y)
        Z = z0 - (a * (X - x0) + b * (Y - y0)) / c
    elif abs(b) > abs(a):
        x = np.linspace(x_min, x_max, 20)
        z = np.linspace(z_min, z_max, 20)
        X, Z = np.meshgrid(x, z)
        Y = y0 - (a * (X - x0) + c * (Z - z0)) / b
    else:
        y = np.linspace(y_min, y_max, 20)
        z = np.linspace(z_min, z_max, 20)
        Y, Z = np.meshgrid(y, z)
        X = x0 - (b * (Y - y0) + c * (Z - z0)) / a

    for ax in axs:
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color)


def calculate_shared_limits(data_arrays, padding=0.1):
    """Calculate shared axis limits from datasets."""
    all_data = np.vstack(data_arrays) if isinstance(data_arrays, list) else data_arrays

    mins = all_data.min(axis=0)
    maxs = all_data.max(axis=0)
    ranges = maxs - mins

    x_limits = [mins[0] - padding * ranges[0], maxs[0] + padding * ranges[0]]
    y_limits = [mins[1] - padding * ranges[1], maxs[1] + padding * ranges[1]]
    z_limits = [mins[2] - padding * ranges[2], maxs[2] + padding * ranges[2]]

    return x_limits, y_limits, z_limits


def plane_view_from_normal(normal):
    """Compute Matplotlib 3D view (elev, azim) from plane normal."""
    normal = np.array(normal) / np.linalg.norm(normal)
    nx, ny, nz = normal

    elev = np.degrees(np.arcsin(nz))
    azim = np.degrees(np.arctan2(ny, nx))

    return elev, azim


# ============================================================================
# Private Helpers
# ============================================================================


def _save_or_show(save_path):
    """Save or show figure."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def _plot_overlay(ax, data1, data2, labels, title):
    """Plot two datasets overlaid."""
    ax.scatter(
        data1[:, 0], data1[:, 1], data1[:, 2], c="green", s=2, alpha=0.2, label="Style"
    )
    ax.scatter(
        data2[:, 0],
        data2[:, 1],
        data2[:, 2],
        c="blue",
        s=2,
        alpha=0.2,
        label="Original",
    )
    ax.set_xlabel(labels[0], fontsize=10)
    ax.set_ylabel(labels[1], fontsize=10)
    ax.set_zlabel(labels[2], fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend()


def _plot_clusters(
    ax, log_content_sampled, bin_masks, dark_points, bright_points, indices
):
    """Plot clustered points with markers."""
    num_clusters = len(bin_masks)
    cmap = plt.cm.get_cmap("tab20" if num_clusters <= 20 else "hsv", num_clusters)

    for i in range(num_clusters):
        bin_mask_flat = bin_masks[i].ravel()
        bin_mask_sampled = bin_mask_flat[indices]
        ax.scatter(
            log_content_sampled[bin_mask_sampled, 0],
            log_content_sampled[bin_mask_sampled, 1],
            log_content_sampled[bin_mask_sampled, 2],
            c=[cmap(i)],
            s=2,
            alpha=0.1,
        )

    for i in range(num_clusters):
        if dark_points is not None:
            ax.scatter(
                dark_points[i][0],
                dark_points[i][1],
                dark_points[i][2],
                c=[cmap(i)],
                edgecolors="black",
                linewidth=0.5,
                s=20,
                alpha=1.0,
            )

        if bright_points is not None:
            ax.scatter(
                bright_points[i][0],
                bright_points[i][1],
                bright_points[i][2],
                c=[cmap(i)],
                edgecolors="red",
                linewidth=0.5,
                s=20,
                alpha=1.0,
            )

    ax.set_xlabel("log(Red)", fontsize=10)
    ax.set_ylabel("log(Green)", fontsize=10)
    ax.set_zlabel("log(Blue)", fontsize=10)
    ax.set_title("Content Log-RGB Clustered", fontsize=12)
