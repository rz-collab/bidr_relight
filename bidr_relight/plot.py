import matplotlib.pyplot as plt
import numpy as np

from bidr_relight.image_process import normalized_linear_to_srgb


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


def plot_content_log_chroma(log_chroma_content, content_bit_depth):
    plt.figure(figsize=(20, 20))

    # Col 1: image
    linear_chroma = np.exp(log_chroma_content).astype(np.float32)
    max_linear_chroma = 2**content_bit_depth - 1
    img_clipped = np.clip(linear_chroma, 0, max_linear_chroma)

    print("max linear chroma", max_linear_chroma)
    print(f"{log_chroma_content.max()=}")

    img_normalized = (img_clipped / max_linear_chroma) * 255.0
    img_normalized = img_normalized.astype(np.uint8)
    assert img_normalized.min() >= 0 and img_normalized.max() <= 255 + 1e-3, (
        f"Image values out of expected range: {img_normalized.min()} to {img_normalized.max()}"
    )
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_normalized)
    ax1.set_title("Content's Log Chromaticity Image", fontsize=12)
    ax1.axis("off")

    # Col 2: log rgb
    print(f"{log_chroma_content.shape=}")
    norm_log_chroma_content = log_chroma_content / np.log(max_linear_chroma)

    # Sample pixels for log-RGB plotting
    num_samples = 10000
    content_flat = norm_log_chroma_content.reshape(-1, 3)

    if len(content_flat) > num_samples:
        indices = np.random.choice(len(content_flat), num_samples, replace=False)
        content_sampled = content_flat[indices]
    else:
        content_sampled = content_flat

    print(f"{content_sampled.shape=}")

    ax2 = plt.subplot(1, 2, 2, projection="3d")
    plot_rgb_space_ax(
        content_sampled,
        content_sampled,
        ax=ax2,
        title="Content's Log Chromaticity Log RGB",
        point_size=2,
        alpha=0.3,
    )


def plot_image_rgb_logrgb(norm_content_img, norm_style_img):
    plt.figure(figsize=(20, 20))

    # Row 1: Images
    # Convert linear to sRGB for visualization.
    content_srgb_img = normalized_linear_to_srgb(norm_content_img)
    style_srgb_img = normalized_linear_to_srgb(norm_style_img)

    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(style_srgb_img)
    ax1.set_title("Style", fontsize=12)
    ax1.axis("off")

    ax2 = plt.subplot(4, 3, 2)
    ax2.imshow(content_srgb_img)
    ax2.set_title("Content", fontsize=12)
    ax2.axis("off")

    # ax3 = plt.subplot(4, 3, 3)
    # ax3.imshow(transformed_rgb)
    # ax3.set_title("Transformed", fontsize=12)
    # ax3.axis("off")

    # Sample pixels for RGB/log-RGB plotting
    num_samples = 5000
    content_flat = norm_content_img.reshape(-1, 3)
    style_flat = norm_style_img.reshape(-1, 3)

    if len(content_flat) > num_samples:
        indices = np.random.choice(len(content_flat), num_samples, replace=False)
        content_sampled = content_flat[indices]
        style_sampled = style_flat[indices]
    else:
        content_sampled = content_flat
        style_sampled = style_flat
        # trans_sampled = trans_flat

    # Row 2: RGB Space
    ax4 = plt.subplot(4, 3, 4, projection="3d")
    plot_rgb_space_ax(
        style_sampled,
        style_sampled,
        ax=ax4,
        title="Style RGB Space",
        point_size=2,
        alpha=0.3,
    )

    ax5 = plt.subplot(4, 3, 5, projection="3d")
    plot_rgb_space_ax(
        content_sampled,
        content_sampled,
        ax=ax5,
        title="Content RGB Space",
        point_size=2,
        alpha=0.3,
    )

    # ax6 = plt.subplot(4, 3, 6, projection="3d")
    # plot_rgb_space_ax(
    #     trans_sampled,
    #     trans_sampled,
    #     ax=ax6,
    #     title="Transformed RGB Space",
    #     point_size=2,
    #     alpha=0.3,
    # )

    # Row 3: Log-RGB Space
    ax7 = plt.subplot(4, 3, 7, projection="3d")
    plot_log_rgb_space_ax(
        style_sampled,
        style_sampled,
        ax=ax7,
        title="Style Log-RGB Space",
        point_size=2,
        alpha=0.3,
    )

    ax8 = plt.subplot(4, 3, 8, projection="3d")
    plot_log_rgb_space_ax(
        content_sampled,
        content_sampled,
        ax=ax8,
        title="Content Log-RGB Space",
        point_size=2,
        alpha=0.3,
    )

    # ax9 = plt.subplot(4, 3, 9, projection="3d")
    # plot_log_rgb_space_ax(
    #     trans_sampled,
    #     trans_sampled,
    #     ax=ax9,
    #     title="Transformed Log-RGB Space",
    #     point_size=2,
    #     alpha=0.3,
    # )

    # Row 4: Comparison by overlaying both RGB and log RGB together
    ax10 = plt.subplot(4, 3, 10, projection="3d")
    ax10.scatter(
        style_sampled[:, 0],
        style_sampled[:, 1],
        style_sampled[:, 2],
        c="green",
        s=2,
        alpha=0.2,
        label="Style",
    )
    ax10.scatter(
        content_sampled[:, 0],
        content_sampled[:, 1],
        content_sampled[:, 2],
        c="blue",
        s=2,
        alpha=0.2,
        label="Original",
    )
    # ax10.scatter(
    #     trans_sampled[:, 0],
    #     trans_sampled[:, 1],
    #     trans_sampled[:, 2],
    #     c="red",
    #     s=2,
    #     alpha=0.2,
    #     label="Transformed",
    # )
    ax10.set_xlabel("Red", fontsize=10)
    ax10.set_ylabel("Green", fontsize=10)
    ax10.set_zlabel("Blue", fontsize=10)
    ax10.set_title("RGB Comparison", fontsize=12)
    ax10.legend()

    ax11 = plt.subplot(4, 3, 11, projection="3d")
    epsilon = 1e-6
    log_orig = np.log(content_sampled + epsilon)
    # log_trans = np.log(trans_sampled + epsilon)
    log_style = np.log(style_sampled + epsilon)
    ax11.scatter(
        log_style[:, 0],
        log_style[:, 1],
        log_style[:, 2],
        c="green",
        s=2,
        alpha=0.2,
        label="Style",
    )
    ax11.scatter(
        log_orig[:, 0],
        log_orig[:, 1],
        log_orig[:, 2],
        c="blue",
        s=2,
        alpha=0.2,
        label="Original",
    )
    # ax11.scatter(
    #     log_trans[:, 0],
    #     log_trans[:, 1],
    #     log_trans[:, 2],
    #     c="red",
    #     s=2,
    #     alpha=0.2,
    #     label="Transformed",
    # )
    ax11.set_xlabel("log(Red)", fontsize=10)
    ax11.set_ylabel("log(Green)", fontsize=10)
    ax11.set_zlabel("log(Blue)", fontsize=10)
    ax11.set_title("Log-RGB Comparison", fontsize=12)
    ax11.legend()

    plt.tight_layout()
    plt.show()
