"""Interactive Gradio demo for step-by-step relighting pipeline.

Dependencies:
    pip install gradio numpy matplotlib pillow imageio
"""
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image

from src.pipeline import RelightingPipeline
from src.bidr_util import get_global_isd
from src.plotting import (
    plot_log_chroma_plane_pre_clustering,
    plot_log_chroma_plane_post_clustering,
    plot_cluster_spatial_distribution,
)


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def visualize_isd_map(isd_map):
    """Visualize ISD map as RGB image."""
    # Normalize to [0, 1] range for visualization
    isd_vis = (isd_map + 1) / 2  # From [-1,1] to [0,1]
    return (isd_vis * 255).astype(np.uint8)


def create_logrgb_comparison(content_log, style_log, tf_log, n_samples=5000):
    """Create 3D log RGB comparison plots."""
    fig = Figure(figsize=(20, 8))
    
    # Sample points
    def sample_flat(log_img):
        flat = log_img.reshape(-1, 3)
        if len(flat) > n_samples:
            idx = np.random.choice(len(flat), n_samples, replace=False)
            return flat[idx]
        return flat
    
    content_pts = sample_flat(content_log)
    style_pts = sample_flat(style_log)
    tf_pts = sample_flat(tf_log)
    
    # Plot 1: Original vs Transformed
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(content_pts[:, 0], content_pts[:, 1], content_pts[:, 2],
                c='blue', s=1, alpha=0.3, label='Original')
    ax1.scatter(tf_pts[:, 0], tf_pts[:, 1], tf_pts[:, 2],
                c='green', s=1, alpha=0.3, label='Transformed')
    ax1.set_xlabel('Log R')
    ax1.set_ylabel('Log G')
    ax1.set_zlabel('Log B')
    ax1.set_title('Original vs Transformed')
    ax1.legend()
    
    # Plot 2: Transformed vs Style
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(tf_pts[:, 0], tf_pts[:, 1], tf_pts[:, 2],
                c='green', s=1, alpha=0.3, label='Transformed')
    ax2.scatter(style_pts[:, 0], style_pts[:, 1], style_pts[:, 2],
                c='red', s=1, alpha=0.3, label='Style')
    ax2.set_xlabel('Log R')
    ax2.set_ylabel('Log G')
    ax2.set_zlabel('Log B')
    ax2.set_title('Transformed vs Style')
    ax2.legend()
    
    return fig_to_pil(fig)


def visualize_log_chromaticity(log_chroma, bit_depth):
    """Visualize log chromaticity as image with robust handling."""
    from src.image_util import normalized_linear_to_srgb
    
    # Convert log chromaticity to linear
    linear_chroma = np.exp(log_chroma).astype(np.float32)
    
    # Check for extreme values
    print(f"  Visualizing log chroma: linear range [{linear_chroma.min():.2f}, {linear_chroma.max():.2f}]")
    
    # Normalize robustly - use percentile clipping to handle outliers
    p_low, p_high = np.percentile(linear_chroma, [1, 99])
    print(f"  Using percentile range: [{p_low:.2f}, {p_high:.2f}]")
    
    # Clip and normalize
    linear_clipped = np.clip(linear_chroma, p_low, p_high)
    if p_high > p_low:
        norm_linear = (linear_clipped - p_low) / (p_high - p_low)
    else:
        # Fallback if image is uniform
        norm_linear = np.clip(linear_chroma / (2**bit_depth - 1), 0.0, 1.0)
    
    # Convert to sRGB for proper display
    img_srgb = normalized_linear_to_srgb(norm_linear)
    
    return img_srgb.astype(np.uint8)

def create_pixel_color_grid(dark_points, bright_points, bit_depth):
    """Create 2xK grid showing actual pixel colors."""
    k = len(dark_points)
    # Convert log to linear RGB
    dark_linear = np.exp(dark_points)
    bright_linear = np.exp(bright_points)
    
    # Normalize to [0,1]
    dark_rgb = np.clip(dark_linear / (2**bit_depth - 1), 0, 1)
    bright_rgb = np.clip(bright_linear / (2**bit_depth - 1), 0, 1)
    
    # Create 2xK image (each cell 50x50 pixels)
    grid_size = 200
    cell_height = 50
    cell_size = grid_size // k
    grid = np.zeros((2 * cell_height, k * cell_size, 3))
    
    for i in range(k):
        # Dark point (top row)
        grid[0:cell_height, i*cell_size:(i+1)*cell_size] = dark_rgb[i]
        # Bright point (bottom row)
        grid[cell_height:, i*cell_size:(i+1)*cell_size] = bright_rgb[i]
    
    return (grid * 255).astype(np.uint8)


def create_scatter_plot(data, title, labels=None):
    """Create 3D scatter plot."""
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is None:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.3, s=1)
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], 
                      alpha=0.5, s=1, label=f'Cluster {label}')
        ax.legend()
    
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title(title)
    
    return fig_to_pil(fig)


def step1_process(content_img, style_img, model_type, resize_scale):
    """Step 1: Load and estimate ISD."""
    if content_img is None or style_img is None:
        return None, None, None, None, None, "Please upload both images"
    
    from src.bidr_util import project_to_log_chromaticity_plane
    
    print("=" * 60)
    print(f"Step 1: Processing {content_img}")
    
    pipeline = RelightingPipeline()
    
    # Pass filepath directly - pipeline now handles loading properly
    pipeline.step1_load_and_estimate_isd(
        content_img, style_img, model_type, 
        "./weights/UNET_run_x10_01_last_model.pth", resize_scale)
    
    # Check what was loaded
    actual_bitdepth = pipeline.content_data["bit_depth"]
    actual_dtype = pipeline.content_data["img"].dtype
    
    print(f"Pipeline result: bit_depth={actual_bitdepth}, dtype={actual_dtype}")
    
    # Visualize ISD maps
    content_isd_vis = visualize_isd_map(pipeline.content_data["isd_map"])
    style_isd_vis = visualize_isd_map(pipeline.style_data["isd_map"])
    
    # Compute adaptive plane offset based on image content
    log_img = pipeline.content_data["log_img"]
    
    # Use median of log values as offset (robust to outliers)
    median_log = np.median(log_img, axis=(0, 1))
    plane_offset = median_log
    
    print(f"Adaptive plane_offset: {plane_offset}")
    print(f"Log RGB range: [{log_img.min():.2f}, {log_img.max():.2f}]")
    print("=" * 60)
    
    # Compute preliminary log chromaticity for preview
    log_chroma_preview = project_to_log_chromaticity_plane(
        pipeline.content_data["log_img"],
        pipeline.content_data["isd_map"],
        plane_offset=plane_offset,
        use_average_isd=True,
    )
    
    # Debug the log chromaticity range
    print(f"Log chroma range: [{log_chroma_preview.min():.2f}, {log_chroma_preview.max():.2f}]")
    
    log_chroma_vis = visualize_log_chromaticity(
        log_chroma_preview,
        pipeline.content_data["bit_depth"]
    )
    
    info = f"""✅ Step 1 Complete
Content: {pipeline.content_data['img'].shape}
Style: {pipeline.style_data['img'].shape}
Content dtype: {pipeline.content_data['img'].dtype}
Content Bit Depth: {pipeline.content_data['bit_depth']} (processing bit depth)
Log RGB range: [{log_img.min():.2f}, {log_img.max():.2f}]
Plane offset: [{plane_offset[0]:.2f}, {plane_offset[1]:.2f}, {plane_offset[2]:.2f}]
Content ISD: {get_global_isd(pipeline.content_data['isd_map'])}
Style ISD: {get_global_isd(pipeline.style_data['isd_map'])}

Note: 8-bit images are auto-converted to 16-bit linear for processing."""
    
    return pipeline, content_isd_vis, style_isd_vis, log_chroma_vis, info


def step2_process(pipeline, method, bin_radius, n_clusters, posterize_levels):
    """Step 2: Cluster materials."""
    if pipeline is None:
        return None, None, None, None, None, "Please complete Step 1 first"
    
    # Convert posterize_levels: 0 means disabled
    post_levels = None if posterize_levels == 0 else int(posterize_levels)
    
    pipeline.step2_cluster_materials(
        method, bin_radius, n_clusters, posterize_levels=post_levels
    )
    
    # Create figures and get them for conversion
    import matplotlib
    matplotlib.use('Agg')
    
    posterize_img = None
    if pipeline.log_chroma_posterized is not None:
        # Create posterize comparison plot
        from src.plotting import plot_log_chroma_plane_posterized
        plt.figure(figsize=(24, 7))
        plot_log_chroma_plane_posterized(
            pipeline.log_chroma_content,
            pipeline.log_chroma_posterized,
            pipeline.content_data["isd_map"],
            pipeline.content_data["img"],
            pipeline.content_data["bit_depth"],
            post_levels,
        )
        posterize_img = fig_to_pil(plt.gcf())
    
    # Pre-clustering plot (uses posterized if enabled)
    clustering_input = (pipeline.log_chroma_posterized 
                       if pipeline.log_chroma_posterized is not None 
                       else pipeline.log_chroma_content)
    
    plt.figure(figsize=(12, 8))
    plot_log_chroma_plane_pre_clustering(
        clustering_input,
        pipeline.content_data["isd_map"],
        pipeline.content_data["img"],
        pipeline.content_data["bit_depth"],
    )
    pre_cluster_img = fig_to_pil(plt.gcf())
    
    # Post-clustering plot
    plt.figure(figsize=(12, 8))
    plot_log_chroma_plane_post_clustering(
        clustering_input,
        pipeline.content_data["isd_map"],
        pipeline.bin_masks,
        bin_radius if method == "greedy" else None,
    )
    post_cluster_img = fig_to_pil(plt.gcf())
    
    # Spatial distribution
    plt.figure(figsize=(12, 8))
    plot_cluster_spatial_distribution(
        pipeline.bin_masks,
        pipeline.content_data["img"],
        pipeline.content_data["bit_depth"],
    )
    spatial_img = fig_to_pil(plt.gcf())
    
    posterize_status = f"\nPosterize: {post_levels} levels" if post_levels else "\nPosterize: Disabled"
    info = f"""✅ Step 2 Complete
Method: {method}
Number of clusters: {len(pipeline.bin_masks)}
Bin radius: {bin_radius if method == 'greedy' else 'N/A'}{posterize_status}"""
    
    return pipeline, posterize_img, pre_cluster_img, post_cluster_img, spatial_img, info

def step3_process(pipeline, always_use_global):
    """Step 3: Estimate illumination."""
    if pipeline is None or pipeline.bin_masks is None:
        return None, None, "Please complete Step 2 first"
    
    # Run step 3
    pipeline.step3_estimate_illumination(always_use_global)
    
    # Create visualization of dark/bright points in log RGB space
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all pixels (sampled)
    log_flat = pipeline.content_data["log_img"].reshape(-1, 3)
    sample_idx = np.random.choice(len(log_flat), size=min(10000, len(log_flat)), replace=False)
    ax.scatter(log_flat[sample_idx, 0], log_flat[sample_idx, 1], 
               log_flat[sample_idx, 2], alpha=0.1, s=1, c='gray')
    
    # Plot dark and bright points
    ax.scatter(pipeline.dark_points[:, 0], pipeline.dark_points[:, 1], 
               pipeline.dark_points[:, 2], c='blue', s=100, label='Dark points', marker='o')
    ax.scatter(pipeline.bright_points[:, 0], pipeline.bright_points[:, 1], 
               pipeline.bright_points[:, 2], c='red', s=100, label='Bright points', marker='^')
    
    # Draw lines connecting dark to bright for each cluster
    for i in range(len(pipeline.dark_points)):
        ax.plot([pipeline.dark_points[i, 0], pipeline.bright_points[i, 0]],
                [pipeline.dark_points[i, 1], pipeline.bright_points[i, 1]],
                [pipeline.dark_points[i, 2], pipeline.bright_points[i, 2]],
                'k-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Log R')
    ax.set_ylabel('Log G')
    ax.set_zlabel('Log B')
    ax.set_title('Illumination Estimation: Dark/Bright Points')
    ax.legend()
    
    illum_img = fig_to_pil(fig)

    color_grid = create_pixel_color_grid(
        pipeline.dark_points, 
        pipeline.bright_points,
        pipeline.content_data["bit_depth"]
    ) 
    
    info = f"""✅ Step 3 Complete
Illumination norm: {pipeline.illum_norm:.3f}
Number of clusters: {len(pipeline.dark_points)}
Dark points shape: {pipeline.dark_points.shape}
Bright points shape: {pipeline.bright_points.shape}"""
    
    return pipeline, illum_img, color_grid, info


def step4_process(pipeline, length_scale, rot_percent, reverse_rotation, log_transl_r, log_transl_g, log_transl_b):
    """Step 4: Apply relighting."""
    if pipeline is None or pipeline.dark_points is None:
        return None, None, None, "Please complete Step 3 first"
    
    # Prepare log translation
    if log_transl_r != 0 or log_transl_g != 0 or log_transl_b != 0:
        log_transl = np.array([log_transl_r, log_transl_g, log_transl_b])
    else:
        log_transl = None
    
    # Run step 4
    pipeline.step4_apply_relighting(length_scale, log_transl, rot_percent, None, reverse_rotation)
    
    # Convert to sRGB for display
    from src.image_util import normalized_linear_to_srgb
    
    # Transform
    tf_linear = np.exp(pipeline.tf_log_content)
    tf_norm = np.clip(tf_linear / (2**pipeline.content_data["bit_depth"] - 1), 0, 1)
    tf_img = normalized_linear_to_srgb(tf_norm).astype(np.uint8)
    
    # Original
    orig_norm = np.clip(pipeline.content_data["img"] / (2**pipeline.content_data["bit_depth"] - 1), 0, 1)
    orig_img = normalized_linear_to_srgb(orig_norm).astype(np.uint8)
    
    # Create before/after comparison
    fig = Figure(figsize=(48, 16))
    ax1 = fig.add_subplot(121)
    ax1.imshow(orig_img)
    ax1.set_title('Original')
    ax1.axis('off')

    ax2 = fig.add_subplot(122)
    ax2.imshow(tf_img)
    ax2.set_title('Transformed')
    ax2.axis('off')

    comparison_img = fig_to_pil(fig)

    logrgb_comparison = create_logrgb_comparison(
        pipeline.content_data["log_img"],
        pipeline.style_data["log_img"],
        pipeline.tf_log_content
    )
        
    info = f"""✅ Step 4 Complete
    Length scale: {length_scale}
    Rotation: {rot_percent}% {'(REVERSED - away from style)' if reverse_rotation else '(toward style)'}
    Log translation: {log_transl}"""
    
    return pipeline, tf_img, comparison_img, logrgb_comparison, info


# Create Gradio interface
with gr.Blocks(title="Interactive Relighting Pipeline") as demo:
    gr.Markdown("# 🎨 Interactive Relighting Pipeline")
    gr.Markdown("""
    Process images step-by-step with full control over parameters
    
    **📝 Note**: 
    - **Best**: 16-bit linear images (RAW/TIFF) for full dynamic range
    - **OK**: 8-bit images (JPEG/PNG) - automatically converted to linear
    - 8-bit images have limited shadow detail and may produce less accurate results
    """)
    
    # State to hold pipeline between steps
    pipeline_state = gr.State(None)
    
    with gr.Tab("📥 Step 1: Load & Estimate ISD"):
        gr.Markdown("### Upload images and estimate Illumination Spectral Direction")
        
        with gr.Row():
            with gr.Column():
                content_input = gr.Image(label="Content Image (16-bit TIFF preferred)", type="filepath")
                style_input = gr.Image(label="Style Image (16-bit TIFF preferred)", type="filepath")
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        ["mock", "unet"],
                        value="mock",
                        label="ISD Model"
                    )
                    resize_scale = gr.Slider(
                        0.1, 1.0, value=0.25, step=0.05,
                        label="Resize Scale"
                    )
                
                step1_btn = gr.Button("▶️ Run Step 1", variant="primary")
            
            with gr.Column():
                step1_info = gr.Textbox(label="Status", lines=8)
                content_isd_output = gr.Image(label="Content ISD Map")
                style_isd_output = gr.Image(label="Style ISD Map")
                log_chroma_preview = gr.Image(label="Log Chromaticity Preview (contrast enhanced)")
        
        step1_btn.click(
            step1_process,
            inputs=[content_input, style_input, model_type, resize_scale],
            outputs=[pipeline_state, content_isd_output, style_isd_output, 
                    log_chroma_preview, step1_info]
        )
    
    with gr.Tab("🎯 Step 2: Cluster Materials"):
        gr.Markdown("### Segment image into material clusters")
        
        with gr.Row():
            with gr.Column():
                clustering_method = gr.Radio(
                    ["greedy", "kmeans", "hierarchical"],
                    value="greedy",
                    label="Clustering Method"
                )
                bin_radius = gr.Slider(
                    0.1, 5.0, value=1.0, step=0.1,
                    label="Bin Radius (greedy only)"
                )
                n_clusters = gr.Slider(
                    2, 10, value=4, step=1,
                    label="Number of Clusters"
                )
                posterize_levels = gr.Slider(
                    0, 32, value=0, step=1,
                    label="Posterize Levels (0=disabled, helps clustering)"
                )
                
                step2_btn = gr.Button("▶️ Run Step 2", variant="primary")
                step2_info = gr.Textbox(label="Status", lines=5)
            
            with gr.Column():
                posterize_comparison = gr.Image(label="Posterization Effect (if enabled)")
                pre_cluster_img = gr.Image(label="Before Clustering")
                post_cluster_img = gr.Image(label="After Clustering")
                spatial_dist_img = gr.Image(label="Spatial Distribution")
        
        step2_btn.click(
            step2_process,
            inputs=[pipeline_state, clustering_method, bin_radius, n_clusters, posterize_levels],
            outputs=[pipeline_state, posterize_comparison, pre_cluster_img, post_cluster_img, 
                    spatial_dist_img, step2_info]
        )
    
    with gr.Tab("💡 Step 3: Estimate Illumination"):
        gr.Markdown("### Estimate global illumination and dark/bright points")
        
        with gr.Row():
            with gr.Column():
                always_use_global = gr.Checkbox(
                    value=True,
                    label="Always Use Global Illumination Norm"
                )
                
                step3_btn = gr.Button("▶️ Run Step 3", variant="primary")
                step3_info = gr.Textbox(label="Status", lines=6)
            
            with gr.Column():
                illum_vis = gr.Image(label="Dark/Bright Points Visualization")
                color_grid = gr.Image(label="Dark/Bright Point Colors")
        
        step3_btn.click(
            step3_process,
            inputs=[pipeline_state, always_use_global],
            outputs=[pipeline_state, illum_vis, color_grid, step3_info]
        )
    
    with gr.Tab("✨ Step 4: Apply Relighting"):
        gr.Markdown("### Apply relighting transformation")
        
        with gr.Row():
            with gr.Column():
                length_scale = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.1,
                    label="Length Scale"
                )
                rot_percent = gr.Slider(
                    0, 500, value=100, step=5,
                    label="Rotation Percentage (100% = full rotation, >100% = over-rotate)"
                )
                reverse_rotation = gr.Checkbox(
                    value=False,
                    label="Reverse Rotation (rotate away from style ISD)"
                )
                
                gr.Markdown("**Log RGB Translation (optional)**")
                log_transl_r = gr.Slider(-2, 2, value=0, step=0.1, label="R")
                log_transl_g = gr.Slider(-2, 2, value=0, step=0.1, label="G")
                log_transl_b = gr.Slider(-2, 2, value=0, step=0.1, label="B")
                
                step4_btn = gr.Button("▶️ Run Step 4", variant="primary")
                step4_info = gr.Textbox(label="Status", lines=5)
            
            with gr.Column():
                final_output = gr.Image(label="Relit Image")
                comparison_output = gr.Image(label="Before/After Comparison")
                logrgb_comparison_output = gr.Image(label="Before/After LogRGB Comparison")
        
        step4_btn.click(
            step4_process,
            inputs=[pipeline_state, length_scale, rot_percent, reverse_rotation,
                   log_transl_r, log_transl_g, log_transl_b],
            outputs=[pipeline_state, final_output, comparison_output, logrgb_comparison_output, step4_info]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## How to Use This Pipeline
        
        ### 📸 Image Requirements
        
        **Supported Formats:**
        - ✅ **Best**: 16-bit linear TIFF/PNG/DNG (full dynamic range)
        - ✅ **OK**: 8-bit JPEG/PNG (auto-converted from sRGB to linear)
        - ⚠️ RAW files (.CR2, .NEF, .ARW) are supported but require rawpy library
        
        **What happens with 8-bit images:**
        - Automatically converted from sRGB gamma to linear
        - Upscaled to 16-bit for processing
        - Results may have limited shadow detail due to original 8-bit quantization
        
        **For best results:**
        - Use 16-bit TIFF files exported from RAW
        - Preserve linear color space (no gamma correction)
        - Maintain full dynamic range (don't clip highlights/shadows)
        
        ### Step 1: Load & Estimate ISD
        - Upload your content and style images (any format)
        - Check console output to see detected bit depth
        - The ISD (Illumination Spectral Direction) maps show estimated lighting direction
        - 8-bit images will show "Converting 8-bit sRGB to 16-bit linear" in console
        
        ### Step 2: Cluster Materials
        - Segment the image into different material regions
        - **Greedy**: Bins pixels in log-chromaticity space by radius
        - **K-means**: Clusters pixels into fixed number of groups
        - **Posterize**: Quantize colors before clustering (optional, helps with noisy images)
        
        ### Step 3: Estimate Illumination
        - Finds the global illumination vector
        - Estimates dark (shadow) and bright (lit) points for each material
        - These endpoints define the lighting range for each cluster
        
        ### Step 4: Apply Relighting
        - Transfer lighting from style to content
        - **Length Scale**: Compress/expand the lighting range (0.5 = darker, 1.5 = brighter)
        - **Rotation**: How much to rotate toward style lighting (0-100%)
        - **Translation**: Shift colors in log RGB space
        
        ## Tips
        - Check the console output for diagnostic info about image loading
        - You can rerun any step with different parameters
        - Changes in early steps require rerunning later steps
        - Try both 8-bit and 16-bit images to see the quality difference
        - For production work, always use 16-bit linear images
        """)

if __name__ == "__main__":
    demo.launch()