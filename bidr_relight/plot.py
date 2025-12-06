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


def plot_rgb_space_ax(pixels, colors, ax, title="RGB Space", point_size=1, alpha=0.3):
    """
    Plot RGB space on existing axis.

    Args:
        pixels: (N, 3) array of RGB values
        colors: (N, 3) array for point colors
        ax: matplotlib 3D axis object
        title: Plot title
        point_size: Point size
        alpha: Point transparency
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

    ax.set_xlabel("Red", fontsize=10)
    ax.set_ylabel("Green", fontsize=10)
    ax.set_zlabel("Blue", fontsize=10)
    ax.set_title(title, fontsize=12)
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.set_zlim([0, 1])


def plot_log_rgb_space_ax(
    pixels, colors, ax, title="Log RGB Space", point_size=1, alpha=0.3
):
    """
    Plot log-RGB space on existing axis.

    Args:
        pixels: (N, 3) array of RGB values [0, 1]
        colors: (N, 3) array for point colors
        ax: matplotlib 3D axis object
        title: Plot title
        point_size: Point size
        alpha: Point transparency
    """
    epsilon = 1e-6
    log_pixels = np.log(pixels + epsilon)

    ax.scatter(
        log_pixels[:, 0],
        log_pixels[:, 1],
        log_pixels[:, 2],
        c=colors,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax.set_xlabel("log(Red)", fontsize=10)
    ax.set_ylabel("log(Green)", fontsize=10)
    ax.set_zlabel("log(Blue)", fontsize=10)
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


def plot_img_rgb_logrgb(
    axs,
    norm_content_img,
    norm_style_img,
    log_norm_content_img,
    log_norm_style_img,
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
