import numpy as np
import torch
from src.models import MockISDModel, ResNet50UNet
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


def relight_content_image(content_path, style_path, isd_model,
                          output_path, shading_only=False,
                          compression_factor=0.7):
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
    output_path : str
        Path to save the transformed image.
    shading_only : bool, default False
        If True, only compress along the ISD without changing illuminant color.
    compression_factor : float, default 0.7
        Factor to compress intensity along the ISD.
    """
    device = "cpu"

    if isd_model == "mock":
        model = MockISDModel().to(device)
    elif isd_model == "unet":
        model = ResNet50UNet(
            in_channels=3,
            out_channels=3,
            pretrained=True,
            checkpoint="src/weights/UNET_last_model.pth",
            se_block=True,
            dropout=0.0
        ).to(device)
        model.eval()
    elif isd_model == "vit":
        # TODO
        pass
    else:
        model = MockISDModel().to(device)

    # --- 1. Load images and drop alpha if present ---
    content_img = imread(content_path).astype(np.float32) / 255.0
    style_img = imread(style_path).astype(np.float32) / 255.0

    if content_img.shape[2] == 4:
        content_img = content_img[:, :, :3]
    if style_img.shape[2] == 4:
        style_img = style_img[:, :, :3]

    H, W, _ = content_img.shape

    # --- 2. Estimate ISD map for content ---
    img_tensor = torch.from_numpy(content_img).permute(2, 0, 1).unsqueeze(0).to(device)
    isd_map = model(img_tensor)
    isd_norm = torch.norm(isd_map, dim=1, keepdim=True)
    isd_map = torch.where(isd_norm != 0, isd_map / isd_norm, isd_map)
    predicted_isd_map = isd_map[0].cpu().detach().numpy()  # (3, H, W)
    predicted_isd_map = np.transpose(predicted_isd_map, (1, 2, 0))  # (H, W, 3)

    # --- 3. Convert to logRGB ---
    epsilon = 1e-6
    log_content = np.log(content_img + epsilon)
    log_style = np.log(style_img + epsilon)

    # FIX: ISD is already a direction vector in log-RGB space - don't take log of it!
    # Just use predicted_isd_map directly

    # --- 4. Estimate direct illuminant chromaticity ---
    if not shading_only:
        from src.illuminant_estimation import RecursiveRetinex  # placeholder
        L_content = RecursiveRetinex(content_img)
        L_style = RecursiveRetinex(style_img)
        log_L_content = np.log(L_content + epsilon)
        log_L_style = np.log(L_style + epsilon)
        delta_illuminant = log_L_style - log_L_content
    else:
        delta_illuminant = np.zeros(3, dtype=np.float32)

    # --- 5. Vectorized transformation ---
    # FIX: Compression in log space
    # To reduce brightness by compression_factor, add log(compression_factor) along ISD
    log_compression = np.log(compression_factor)

    # Move along ISD direction (ISD points dark→bright, so negative = darker)
    transformed_log = log_content + log_compression * predicted_isd_map + delta_illuminant

    # --- 6. Convert back to RGB ---
    transformed_rgb = np.exp(transformed_log) - epsilon
    transformed_rgb = np.clip(transformed_rgb, 0.0, 1.0)

    imsave(output_path, (transformed_rgb * 255).astype(np.uint8))

    # Sample pixels for RGB/log-RGB plotting
    num_samples = 5000
    pixels_flat = content_img.reshape(-1, 3)
    trans_flat = transformed_rgb.reshape(-1, 3)
    style_flat = style_img.reshape(-1, 3)

    if len(pixels_flat) > num_samples:
        indices = np.random.choice(len(pixels_flat), num_samples, replace=False)
        pixels_sampled = pixels_flat[indices]
        trans_sampled = trans_flat[indices]
        style_sampled = style_flat[indices]
    else:
        pixels_sampled = pixels_flat
        trans_sampled = trans_flat
        style_sampled = style_flat

    # Create visualization
    fig = plt.figure(figsize=(20, 20))

    # Row 1: Images
    ax1 = plt.subplot(4, 3, 1)
    ax1.imshow(style_img)
    ax1.set_title("Style", fontsize=12)
    ax1.axis('off')

    ax2 = plt.subplot(4, 3, 2)
    ax2.imshow(content_img)
    ax2.set_title("Content", fontsize=12)
    ax2.axis('off')

    ax3 = plt.subplot(4, 3, 3)
    ax3.imshow(transformed_rgb)
    ax3.set_title("Transformed", fontsize=12)
    ax3.axis('off')

    # Row 2: RGB Space
    ax4 = plt.subplot(4, 3, 4, projection='3d')
    plot_rgb_space_ax(style_sampled, style_sampled, ax=ax4,
                      title="Style RGB Space", point_size=2, alpha=0.3)

    ax5 = plt.subplot(4, 3, 5, projection='3d')
    plot_rgb_space_ax(pixels_sampled, pixels_sampled, ax=ax5,
                      title="Content RGB Space", point_size=2, alpha=0.3)

    ax6 = plt.subplot(4, 3, 6, projection='3d')
    plot_rgb_space_ax(trans_sampled, trans_sampled, ax=ax6,
                      title="Transformed RGB Space", point_size=2, alpha=0.3)

    # Row 3: Log-RGB Space
    ax7 = plt.subplot(4, 3, 7, projection='3d')
    plot_log_rgb_space_ax(style_sampled, style_sampled, ax=ax7,
                          title="Style Log-RGB Space", point_size=2, alpha=0.3)

    ax8 = plt.subplot(4, 3, 8, projection='3d')
    plot_log_rgb_space_ax(pixels_sampled, pixels_sampled, ax=ax8,
                          title="Content Log-RGB Space", point_size=2, alpha=0.3)

    ax9 = plt.subplot(4, 3, 9, projection='3d')
    plot_log_rgb_space_ax(trans_sampled, trans_sampled, ax=ax9,
                          title="Transformed Log-RGB Space", point_size=2, alpha=0.3)


    # Row 4: Comparison
    ax10 = plt.subplot(4, 3, 10, projection='3d')
    ax10.scatter(style_sampled[:, 0], style_sampled[:, 1], style_sampled[:, 2],
                 c="green", s=2, alpha=0.2, label='Style')
    ax10.scatter(pixels_sampled[:, 0], pixels_sampled[:, 1], pixels_sampled[:, 2],
                c="blue", s=2, alpha=0.2, label='Original')
    ax10.scatter(trans_sampled[:, 0], trans_sampled[:, 1], trans_sampled[:, 2],
                c="red", s=2, alpha=0.2, label='Transformed')
    ax10.set_xlabel('Red', fontsize=10)
    ax10.set_ylabel('Green', fontsize=10)
    ax10.set_zlabel('Blue', fontsize=10)
    ax10.set_title("RGB Comparison", fontsize=12)
    ax10.legend()

    ax11 = plt.subplot(4, 3, 11, projection='3d')
    # Overlay both in log space
    epsilon = 1e-6
    log_orig = np.log(pixels_sampled + epsilon)
    log_trans = np.log(trans_sampled + epsilon)
    log_style = np.log(style_sampled + epsilon)
    ax11.scatter(log_style[:, 0], log_style[:, 1], log_style[:, 2],
                 c="green", s=2, alpha=0.2, label='Style')
    ax11.scatter(log_orig[:, 0], log_orig[:, 1], log_orig[:, 2],
                c="blue", s=2, alpha=0.2, label='Original')
    ax11.scatter(log_trans[:, 0], log_trans[:, 1], log_trans[:, 2],
                c="red", s=2, alpha=0.2, label='Transformed')
    ax11.set_xlabel('log(Red)', fontsize=10)
    ax11.set_ylabel('log(Green)', fontsize=10)
    ax11.set_zlabel('log(Blue)', fontsize=10)
    ax11.set_title("Log-RGB Comparison", fontsize=12)
    ax11.legend()

    plt.tight_layout()
    plt.show()


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
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
               c=colors, s=point_size, alpha=alpha, rasterized=True)

    ax.set_xlabel('Red', fontsize=10)
    ax.set_ylabel('Green', fontsize=10)
    ax.set_zlabel('Blue', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])


def plot_log_rgb_space_ax(pixels, colors, ax, title="Log RGB Space", point_size=1, alpha=0.3):
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

    ax.scatter(log_pixels[:, 0], log_pixels[:, 1], log_pixels[:, 2],
               c=colors, s=point_size, alpha=alpha, rasterized=True)

    ax.set_xlabel('log(Red)', fontsize=10)
    ax.set_ylabel('log(Green)', fontsize=10)
    ax.set_zlabel('log(Blue)', fontsize=10)
    ax.set_title(title, fontsize=12)