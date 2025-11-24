import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


def process_image(img):
    """Convert image to linear RGB and compute logRGB"""
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.clip(img_array, 1e-6, 1.0)  # Avoid log(0)
    log_rgb = np.log(img_array)
    return img_array, log_rgb


def find_darkest_pixel(log_rgb):
    """Find the darkest pixel (minimum in logRGB space)"""
    flat_log = log_rgb.reshape(-1, 3)
    # Darkest pixel is the one with minimum sum of log values
    darkest_idx = np.argmin(np.sum(flat_log, axis=1))
    return flat_log[darkest_idx]


def rotate_3d(points, angles, center):
    """Rotate points around center by angles (in radians) around R, G, B axes"""
    # Translate to origin
    centered = points - center

    # Rotation matrices for each axis
    # Rotation around R axis (G-B plane)
    angle_r = angles[0]
    Rr = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_r), -np.sin(angle_r)],
            [0, np.sin(angle_r), np.cos(angle_r)],
        ]
    )

    # Rotation around G axis (R-B plane)
    angle_g = angles[1]
    Rg = np.array(
        [
            [np.cos(angle_g), 0, np.sin(angle_g)],
            [0, 1, 0],
            [-np.sin(angle_g), 0, np.cos(angle_g)],
        ]
    )

    # Rotation around B axis (R-G plane)
    angle_b = angles[2]
    Rb = np.array(
        [
            [np.cos(angle_b), -np.sin(angle_b), 0],
            [np.sin(angle_b), np.cos(angle_b), 0],
            [0, 0, 1],
        ]
    )

    # Apply rotations
    rotated = centered @ Rr.T @ Rg.T @ Rb.T

    # Translate back
    return rotated + center


def create_logrgb_plots(content_log, style_log):
    """Create three 2D scatter plots of logRGB space"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Sample points for visualization
    content_sample = content_log.reshape(-1, 3)[::100]
    style_sample = style_log.reshape(-1, 3)[::100]

    # logR vs logG
    axes[0][0].scatter(
        content_sample[:, 0],
        content_sample[:, 1],
        c="blue",
        alpha=0.3,
        s=1,
        label="Content",
    )
    axes[0][0].scatter(
        style_sample[:, 0], style_sample[:, 1], c="red", alpha=0.3, s=1, label="Style"
    )
    axes[0][0].set_xlabel("log R", fontdict={"fontsize": 20})
    axes[0][0].set_ylabel("log G", fontdict={"fontsize": 20})
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # logR vs logB
    axes[0][1].scatter(
        content_sample[:, 0],
        content_sample[:, 2],
        c="blue",
        alpha=0.3,
        s=1,
        label="Content",
    )
    axes[0][1].scatter(
        style_sample[:, 0], style_sample[:, 2], c="red", alpha=0.3, s=1, label="Style"
    )
    axes[0][1].set_xlabel("log R", fontdict={"fontsize": 20})
    axes[0][1].set_ylabel("log B", fontdict={"fontsize": 20})
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    # logG vs logB
    axes[1][0].scatter(
        content_sample[:, 1],
        content_sample[:, 2],
        c="blue",
        alpha=0.3,
        s=1,
        label="Content",
    )
    axes[1][0].scatter(
        style_sample[:, 1], style_sample[:, 2], c="red", alpha=0.3, s=1, label="Style"
    )
    axes[1][0].set_xlabel("log G", fontdict={"fontsize": 20})
    axes[1][0].set_ylabel("log B", fontdict={"fontsize": 20})
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    # TODO: Plot Key

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def apply_transform(
    content_img, style_img, shift_r, shift_g, shift_b, rot_r, rot_g, rot_b
):
    """Apply logRGB translation and rotation to content image"""
    if content_img is None:
        return None, None, None

    # Process images
    content_linear, content_log = process_image(content_img)

    # Find darkest pixel to use as rotation center
    dark_center = find_darkest_pixel(content_log)

    # Reshape for rotation
    h, w, c = content_log.shape
    flat_log = content_log.reshape(-1, 3)

    # Apply rotation around darkest pixel
    angles = np.radians([rot_r, rot_g, rot_b])  # Convert degrees to radians
    rotated_log = rotate_3d(flat_log, angles, dark_center)
    rotated_log = rotated_log.reshape(h, w, c)

    # Apply translation in logRGB space
    transformed_log = rotated_log + np.array([shift_r, shift_g, shift_b])

    # Convert back to linear RGB
    transformed_linear = np.exp(transformed_log)
    transformed_linear = np.clip(transformed_linear, 0, 1)

    # Convert to uint8
    output_img = (transformed_linear * 255).astype(np.uint8)

    # Create logRGB visualization if style image exists
    logrgb_plot = None
    if style_img is not None:
        _, style_log = process_image(style_img)
        logrgb_plot = create_logrgb_plots(transformed_log, style_log)
    else:
        logrgb_plot = create_logrgb_plots(transformed_log, content_log)

    return Image.fromarray(output_img), logrgb_plot, transformed_log


def reset_transforms():
    """Reset sliders to zero"""
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


with gr.Blocks() as demo:
    gr.Markdown("# LogRGB Image Transformation Visualizer")

    with gr.Row():
        with gr.Column():
            content_input = gr.Image(type="pil", label="Content Image")
            style_input = gr.Image(type="pil", label="Style Image")

        with gr.Column():
            output_img = gr.Image(label="Transformed Content")
            logrgb_plot = gr.Image(label="LogRGB Space Visualization (2D Projections)")

    gr.Markdown("### Translation")
    with gr.Row():
        shift_r = gr.Slider(-10, 10, value=0, step=0.01, label="Log R Shift")
        shift_g = gr.Slider(-10, 10, value=0, step=0.01, label="Log G Shift")
        shift_b = gr.Slider(-10, 10, value=0, step=0.01, label="Log B Shift")

    gr.Markdown("### Rotation (around darkest pixel)")
    with gr.Row():
        rot_r = gr.Slider(
            -180, 180, value=0, step=1, label="Rotation around R axis (degrees)"
        )
        rot_g = gr.Slider(
            -180, 180, value=0, step=1, label="Rotation around G axis (degrees)"
        )
        rot_b = gr.Slider(
            -180, 180, value=0, step=1, label="Rotation around B axis (degrees)"
        )

    with gr.Row():
        reset_btn = gr.Button("Reset Transforms")

    # Hidden state to store log values
    log_state = gr.State()

    # Update on any change
    inputs = [
        content_input,
        style_input,
        shift_r,
        shift_g,
        shift_b,
        rot_r,
        rot_g,
        rot_b,
    ]
    outputs = [output_img, logrgb_plot, log_state]

    content_input.change(apply_transform, inputs, outputs)
    style_input.change(apply_transform, inputs, outputs)
    shift_r.change(apply_transform, inputs, outputs)
    shift_g.change(apply_transform, inputs, outputs)
    shift_b.change(apply_transform, inputs, outputs)
    rot_r.change(apply_transform, inputs, outputs)
    rot_g.change(apply_transform, inputs, outputs)
    rot_b.change(apply_transform, inputs, outputs)

    reset_btn.click(
        reset_transforms, None, [shift_r, shift_g, shift_b, rot_r, rot_g, rot_b]
    )

demo.launch(share=True)
