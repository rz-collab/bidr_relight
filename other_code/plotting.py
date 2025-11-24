import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import torch
from PIL import Image


def load_and_sample_image(image_path, num_samples=10000, device='cuda'):
    """
    Load image and intelligently sample pixels using GPU.

    Args:
        image_path: Path to image file
        num_samples: Number of pixels to sample (default: 10k)
        device: 'cuda' or 'cpu'

    Returns:
        pixels: (N, 3) array of RGB values [0, 1]
        colors: (N, 3) array for plotting (same as pixels)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]

    H, W, C = img.shape
    total_pixels = H * W

    print(f"Image size: {H}x{W} = {total_pixels:,} pixels")

    # Move to GPU
    img_tensor = torch.from_numpy(img).to(device)
    pixels_flat = img_tensor.reshape(-1, 3)

    # Intelligent sampling strategies
    if num_samples >= total_pixels:
        # Use all pixels if requested more than available
        sampled_pixels = pixels_flat
        print(f"Using all {total_pixels:,} pixels")
    else:
        # Strategy: Mix random sampling with stratified sampling
        # This ensures we capture both dense regions and outliers

        # 80% random uniform sampling
        num_random = int(num_samples * 0.8)
        random_indices = torch.randperm(total_pixels, device=device)[:num_random]
        random_samples = pixels_flat[random_indices]

        # 20% edge detection sampling (high-gradient regions)
        # These often capture material boundaries and shadows
        num_edge = num_samples - num_random

        # Compute simple gradient magnitude on GPU
        img_gray = img_tensor.mean(dim=2)  # Grayscale
        grad_y = torch.abs(img_gray[1:, :] - img_gray[:-1, :])
        grad_x = torch.abs(img_gray[:, 1:] - img_gray[:, :-1])

        # Pad to match original size
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))

        gradient_mag = (grad_y + grad_x).flatten()

        # Sample from high-gradient regions
        _, top_indices = torch.topk(gradient_mag, min(num_edge, len(gradient_mag)))
        edge_samples = pixels_flat[top_indices]

        # Combine samples
        sampled_pixels = torch.cat([random_samples, edge_samples], dim=0)
        print(f"Sampled {len(sampled_pixels):,} pixels ({num_random:,} random + {len(edge_samples):,} edge)")

    # Move back to CPU for plotting
    pixels_np = sampled_pixels.cpu().numpy()
    colors_np = pixels_np.copy()  # For matplotlib color argument

    return pixels_np, colors_np


def plot_rgb_space(pixels, colors, title="RGB Space", save_path=None,
                   alpha=0.3, point_size=1):
    """
    Fast 3D scatter plot of RGB space.

    Args:
        pixels: (N, 3) array of RGB values
        colors: (N, 3) array for point colors
        title: Plot title
        save_path: If provided, save figure instead of showing
        alpha: Point transparency
        point_size: Point size
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
               c=colors, s=point_size, alpha=alpha,
               rasterized=True)  # Rasterize for faster rendering

    # Labels and styling
    ax.set_xlabel('Red', fontsize=12)
    ax.set_ylabel('Green', fontsize=12)
    ax.set_zlabel('Blue', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add origin marker
    ax.scatter([0], [0], [0], c='black', s=100, marker='o',
               label='Origin', depthshade=False)

    # Set limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_log_rgb_space(pixels, colors, title="Log RGB Space",
                       save_path=None, alpha=0.3, point_size=1):
    """
    Fast 3D scatter plot of log-RGB space.

    Args:
        pixels: (N, 3) array of RGB values [0, 1]
        colors: (N, 3) array for point colors
        title: Plot title
        save_path: If provided, save figure instead of showing
        alpha: Point transparency
        point_size: Point size
    """
    # Compute log with small epsilon to avoid log(0)
    epsilon = 1e-6
    log_pixels = np.log(pixels + epsilon)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(log_pixels[:, 0], log_pixels[:, 1], log_pixels[:, 2],
               c=colors, s=point_size, alpha=alpha,
               rasterized=True)

    # Labels and styling
    ax.set_xlabel('log(Red)', fontsize=12)
    ax.set_ylabel('log(Green)', fontsize=12)
    ax.set_zlabel('log(Blue)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Origin in log space is at log(epsilon) ≈ very negative
    # Don't plot it as it's off the chart

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_material_cylinder(img, material_mask, title="BIDR Cylinder for Material",
                           save_path=None):
    """
    Plot a single material's cylinder in both RGB and log-RGB.

    Args:
        img: (H, W, 3) image array [0, 1]
        material_mask: (H, W) boolean mask for material region
        title: Base title
        save_path: Base path for saving (will add _rgb.png, _log.png)
    """
    # Extract material pixels
    material_pixels = img[material_mask]
    colors = material_pixels.copy()

    print(f"Material has {len(material_pixels):,} pixels")

    # Plot RGB space
    plot_rgb_space(material_pixels, colors,
                   title=f"{title} - RGB Space",
                   save_path=save_path + "_rgb.png" if save_path else None)

    # Plot log-RGB space
    plot_log_rgb_space(material_pixels, colors,
                       title=f"{title} - Log RGB Space",
                       save_path=save_path + "_log.png" if save_path else None)


def compare_multiple_materials(img, material_masks, labels, save_path=None):
    """
    Compare multiple materials in the same log-RGB plot.
    Useful for seeing parallel cylinders.

    Args:
        img: (H, W, 3) image
        material_masks: List of (H, W) boolean masks
        labels: List of material names
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    epsilon = 1e-6

    # Plot each material with different marker
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, (mask, label) in enumerate(zip(material_masks, labels)):
        material_pixels = img[mask]

        # Sample if too many pixels
        if len(material_pixels) > 5000:
            indices = np.random.choice(len(material_pixels), 5000, replace=False)
            material_pixels = material_pixels[indices]

        log_pixels = np.log(material_pixels + epsilon)

        ax.scatter(log_pixels[:, 0], log_pixels[:, 1], log_pixels[:, 2],
                   c=material_pixels, s=2, alpha=0.5,
                   marker=markers[i % len(markers)],
                   label=label, rasterized=True)

    ax.set_xlabel('log(Red)', fontsize=12)
    ax.set_ylabel('log(Green)', fontsize=12)
    ax.set_zlabel('log(Blue)', fontsize=12)
    ax.set_title('Multiple Materials in Log-RGB Space', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()

def load_and_show_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img.show()

# ISD

def visualize_isd_as_direction(isd_map, save_path="isd_direction.png"):
    """
    Visualize ISD as color-coded direction.
    Maps 3D unit vector to HSV color space.

    Args:
        isd_map: (3, H, W) numpy array of ISD vectors (unit vectors)
        save_path: Where to save visualization
    """
    C, H, W = isd_map.shape

    # ISD is a 3D unit vector per pixel
    # We'll map the direction to a color

    # Method: Use spherical coordinates
    # Convert (x, y, z) unit vector to (azimuth, elevation)

    x, y, z = isd_map[0], isd_map[1], isd_map[2]

    # Azimuth angle (0 to 2π) → Hue
    azimuth = np.arctan2(y, x)  # Range: [-π, π]
    hue = (azimuth + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Elevation angle (-π/2 to π/2) → Saturation
    elevation = np.arcsin(np.clip(z, -1, 1))  # Range: [-π/2, π/2]
    saturation = (elevation + np.pi / 2) / np.pi  # Normalize to [0, 1]

    # Value: constant (all vectors are unit length)
    value = np.ones_like(hue)

    # Create HSV image
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Save
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.title("ISD Direction Map (Color = Direction in 3D space)")
    plt.colorbar(label="Direction")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ISD direction visualization to {save_path}")
    return rgb


def visualize_isd_components(isd_map, save_path="isd_components.png"):
    """
    Visualize each ISD component (R, G, B) separately.

    Args:
        isd_map: (3, H, W) numpy array
        save_path: Where to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    labels = ['ISD_R (log-Red component)', 'ISD_G (log-Green component)', 'ISD_B (log-Blue component)']

    for i, (ax, label) in enumerate(zip(axes, labels)):
        # Normalize to [0, 1] for visualization
        # ISD components typically in range [-1, 1] for unit vectors
        component = isd_map[i]

        # Normalize for display
        vmin, vmax = component.min(), component.max()
        normalized = (component - vmin) / (vmax - vmin + 1e-8)

        im = ax.imshow(normalized, cmap='RdBu_r')
        ax.set_title(label)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label=f'Range: [{vmin:.3f}, {vmax:.3f}]')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ISD components to {save_path}")


def visualize_isd_magnitude(isd_map, save_path="isd_magnitude.png"):
    """
    Show the magnitude of ISD vectors (should be ~1 if normalized).
    Useful for debugging - checks if vectors are properly normalized.

    Args:
        isd_map: (3, H, W) numpy array
        save_path: Where to save
    """
    # Compute magnitude per pixel
    magnitude = np.sqrt((isd_map ** 2).sum(axis=0))  # (H, W)

    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, cmap='viridis')
    plt.colorbar(label='ISD Magnitude')
    plt.title(f'ISD Magnitude Map (should be ≈1.0)\nMean: {magnitude.mean():.4f}, Std: {magnitude.std():.4f}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ISD magnitude to {save_path}")
    print(f"Magnitude stats: mean={magnitude.mean():.4f}, std={magnitude.std():.4f}")

    # Should be close to 1.0 if properly normalized
    if abs(magnitude.mean() - 1.0) > 0.1:
        print("⚠️  Warning: ISD vectors may not be properly normalized!")


def visualize_isd_as_rgb_shifted(isd_map, save_path="isd_as_rgb.png"):
    """
    Simple visualization: treat ISD components as RGB (shifted/scaled to [0,1]).
    Not physically meaningful but easy to interpret.

    Args:
        isd_map: (3, H, W) numpy array
        save_path: Where to save
    """
    # Shift and scale to [0, 1] range
    shifted = isd_map + 1.0  # Now in [0, 2] approximately
    scaled = shifted / 2.0  # Now in [0, 1]

    # Transpose to (H, W, 3) for display
    rgb_approx = np.transpose(scaled, (1, 2, 0))
    rgb_approx = np.clip(rgb_approx, 0, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_approx)
    plt.title("ISD as Pseudo-RGB (shifted to visible range)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved pseudo-RGB visualization to {save_path}")


def visualize_isd_arrow_field(isd_map, original_img,
                              stride=50, save_path="isd_arrows.png"):
    """
    Overlay ISD as arrow field on original image.
    Shows local ISD direction at sampled points.

    Args:
        isd_map: (3, H, W) ISD vectors
        original_img: (3, H, W) or (H, W, 3) original image for background
        stride: Spacing between arrows (larger = fewer arrows)
        save_path: Where to save
    """
    # Ensure original_img is (H, W, 3)
    if original_img.shape[0] == 3:
        original_img = np.transpose(original_img, (1, 2, 0))

    H, W = original_img.shape[:2]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(original_img)

    # Sample ISD at regular grid
    y_coords = np.arange(stride // 2, H, stride)
    x_coords = np.arange(stride // 2, W, stride)

    # Project 3D ISD to 2D for visualization
    # Use first two components (R, G) for xy direction
    for y in y_coords:
        for x in x_coords:
            isd_vec = isd_map[:, y, x]

            # Project to 2D: use R and G components
            dx = isd_vec[0] * stride * 0.4  # Scale for visibility
            dy = isd_vec[1] * stride * 0.4

            # Color code by B component
            b_component = isd_vec[2]
            color = plt.cm.RdBu_r((b_component + 1) / 2)  # Map [-1,1] to colormap

            # Draw arrow
            ax.arrow(x, y, dx, dy,
                     head_width=stride * 0.15, head_length=stride * 0.15,
                     fc=color, ec=color, alpha=0.7, width=1.5)

    ax.set_title("ISD Vector Field (arrows show local illumination direction)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved arrow field to {save_path}")


def analyze_isd_statistics(isd_map):
    """
    Print statistical summary of ISD map.
    Useful for debugging and understanding the output.

    Args:
        isd_map: (3, H, W) numpy array
    """
    print("\n" + "=" * 50)
    print("ISD STATISTICS")
    print("=" * 50)

    # Per-component statistics
    for i, name in enumerate(['R', 'G', 'B']):
        component = isd_map[i]
        print(f"\nISD_{name} component:")
        print(f"  Range: [{component.min():.4f}, {component.max():.4f}]")
        print(f"  Mean: {component.mean():.4f}")
        print(f"  Std: {component.std():.4f}")

    # Vector magnitude (should be ~1 if normalized)
    magnitude = np.sqrt((isd_map ** 2).sum(axis=0))
    print(f"\nISD Magnitude:")
    print(f"  Mean: {magnitude.mean():.4f} (should be ≈1.0)")
    print(f"  Std: {magnitude.std():.4f} (should be ≈0.0)")
    print(f"  Range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")

    # Average ISD (global estimate)
    avg_isd = isd_map.reshape(3, -1).mean(axis=1)
    avg_isd_normalized = avg_isd / np.linalg.norm(avg_isd)
    print(f"\nAverage ISD (global):")
    print(f"  Raw: [{avg_isd[0]:.4f}, {avg_isd[1]:.4f}, {avg_isd[2]:.4f}]")
    print(f"  Normalized: [{avg_isd_normalized[0]:.4f}, {avg_isd_normalized[1]:.4f}, {avg_isd_normalized[2]:.4f}]")

    # Check if within natural illumination range
    neutral_isd = np.array([0.577, 0.577, 0.577])
    sunset_isd = np.array([0.789, 0.547, 0.299])

    dist_to_neutral = np.linalg.norm(avg_isd_normalized - neutral_isd)
    dist_to_sunset = np.linalg.norm(avg_isd_normalized - sunset_isd)

    print(f"\nDistance to reference ISDs:")
    print(f"  Neutral (overcast): {dist_to_neutral:.4f}")
    print(f"  Sunset: {dist_to_sunset:.4f}")

    print("=" * 50 + "\n")


def visualize_isd_output(predicted_isds, original_img, output_prefix="output"):
    """
    Complete visualization pipeline for ISD model output.

    Args:
        predicted_isds: (3, H, W) numpy array from your model
        original_img: (H, W, 3) or (3, H, W) original image
        output_prefix: Prefix for saved files
    """
    print("Creating ISD visualizations...")

    # Ensure original_img is (H, W, 3)
    if original_img.shape[0] == 3:
        original_img = np.transpose(original_img, (1, 2, 0))

    # 1. Statistical analysis (print to console)
    analyze_isd_statistics(predicted_isds)

    # 2. Direction map (most informative)
    visualize_isd_as_direction(predicted_isds,
                               save_path=f"{output_prefix}_direction.png")

    # 3. Component visualization
    visualize_isd_components(predicted_isds,
                             save_path=f"{output_prefix}_components.png")

    # 4. Magnitude check (debugging)
    visualize_isd_magnitude(predicted_isds,
                            save_path=f"{output_prefix}_magnitude.png")

    # 5. Pseudo-RGB (quick visual check)
    visualize_isd_as_rgb_shifted(predicted_isds,
                                 save_path=f"{output_prefix}_pseudorgb.png")

    # 6. Arrow field overlay (shows local variation)
    visualize_isd_arrow_field(predicted_isds, original_img,
                              stride=40,
                              save_path=f"{output_prefix}_arrows.png")

    print(f"\n✓ All visualizations saved with prefix '{output_prefix}_'")