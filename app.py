"""Interactive Gradio demo for step-by-step relighting pipeline."""
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


def visualize_log_chromaticity(log_chroma, bit_depth):
    """Visualize log chromaticity as image."""
    # Convert log chromaticity back to displayable image
    log_chroma_img = np.exp(log_chroma)
    log_chroma_img = np.clip(log_chroma_img / (2**bit_depth - 1), 0, 1)
    return (log_chroma_img * 255).astype(np.uint8)


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
    
    pipeline = RelightingPipeline()
    
    # Save uploaded images temporarily
    content_path = "/tmp/content.png"
    style_path = "/tmp/style.png"
    Image.fromarray(content_img).save(content_path)
    Image.fromarray(style_img).save(style_path)
    
    # Run step 1
    pipeline.step1_load_and_estimate_isd(
        content_path, style_path, model_type, "./weights/UNET_run_x10_01_last_model.pth", resize_scale)
    
    # Visualize ISD maps
    content_isd_vis = visualize_isd_map(pipeline.content_data["isd_map"])
    style_isd_vis = visualize_isd_map(pipeline.style_data["isd_map"])
    
    # Compute preliminary log chromaticity for preview
    log_chroma_preview = project_to_log_chromaticity_plane(
        pipeline.content_data["log_img"],
        pipeline.content_data["isd_map"],
        plane_offset=np.array([10.4, 10.4, 10.4]),
        use_average_isd=False,
    )
    log_chroma_vis = visualize_log_chromaticity(
        log_chroma_preview,
        16
    )
    
    info = f"""✅ Step 1 Complete
Content: {pipeline.content_data['img'].shape}
Style: {pipeline.style_data['img'].shape}
Content ISD: {get_global_isd(pipeline.content_data['isd_map'])}
Style ISD: {get_global_isd(pipeline.style_data['isd_map'])}"""
    
    return pipeline, content_isd_vis, style_isd_vis, log_chroma_vis, info


def step2_process(pipeline, method, bin_radius, n_clusters):
    """Step 2: Cluster materials."""
    if pipeline is None:
        return None, None, None, None, "Please complete Step 1 first"
    
    pipeline.step2_cluster_materials(method, bin_radius, n_clusters)
    
    # Create figures and get them for conversion
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Pre-clustering plot
    plt.figure(figsize=(12, 8))
    plot_log_chroma_plane_pre_clustering(
        pipeline.log_chroma_content,
        pipeline.content_data["isd_map"],
        pipeline.content_data["img"],
        pipeline.content_data["bit_depth"],
    )
    pre_cluster_img = fig_to_pil(plt.gcf())
    
    # Post-clustering plot
    plt.figure(figsize=(12, 8))
    plot_log_chroma_plane_post_clustering(
        pipeline.log_chroma_content,
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
    
    info = f"""✅ Step 2 Complete
Method: {method}
Number of clusters: {len(pipeline.bin_masks)}
Bin radius: {bin_radius if method == 'greedy' else 'N/A'}"""
    
    return pipeline, pre_cluster_img, post_cluster_img, spatial_img, info

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
    
    info = f"""✅ Step 3 Complete
Illumination norm: {pipeline.illum_norm:.3f}
Number of clusters: {len(pipeline.dark_points)}
Dark points shape: {pipeline.dark_points.shape}
Bright points shape: {pipeline.bright_points.shape}"""
    
    return pipeline, illum_img, info


def step4_process(pipeline, length_scale, rot_percent, log_transl_r, log_transl_g, log_transl_b):
    """Step 4: Apply relighting."""
    if pipeline is None or pipeline.dark_points is None:
        return None, None, None, "Please complete Step 3 first"
    
    # Prepare log translation
    if log_transl_r != 0 or log_transl_g != 0 or log_transl_b != 0:
        log_transl = np.array([log_transl_r, log_transl_g, log_transl_b])
    else:
        log_transl = None
    
    # Run step 4
    pipeline.step4_apply_relighting(length_scale, log_transl, rot_percent, None)
    
    # Convert back to displayable image
    tf_img = np.exp(pipeline.tf_log_content)
    tf_img = np.clip(tf_img / (2**pipeline.content_data["bit_depth"] - 1), 0, 1)
    tf_img = (tf_img * 255).astype(np.uint8)
    
    # Original for comparison
    orig_img = pipeline.content_data["img"]
    orig_img = (orig_img / (2**pipeline.content_data["bit_depth"] - 1) * 255).astype(np.uint8)
    
    # Create before/after comparison
    fig = Figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(orig_img)
    ax1.set_title('Original Content')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(tf_img)
    ax2.set_title('Relit Content')
    ax2.axis('off')
    
    comparison_img = fig_to_pil(fig)
    
    info = f"""✅ Step 4 Complete
Length scale: {length_scale}
Rotation: {rot_percent}%
Log translation: {log_transl}"""
    
    return pipeline, tf_img, comparison_img, info


# Create Gradio interface
with gr.Blocks(title="Interactive Relighting Pipeline") as demo:
    gr.Markdown("# 🎨 Interactive Relighting Pipeline")
    gr.Markdown("Process images step-by-step with full control over parameters")
    
    # State to hold pipeline between steps
    pipeline_state = gr.State(None)
    
    with gr.Tab("📥 Step 1: Load & Estimate ISD"):
        gr.Markdown("### Upload images and estimate Illumination Spectral Direction")
        
        with gr.Row():
            with gr.Column():
                content_input = gr.Image(label="Content Image", type="numpy")
                style_input = gr.Image(label="Style Image", type="numpy")
                
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
                step1_info = gr.Textbox(label="Status", lines=6)
                content_isd_output = gr.Image(label="Content ISD Map")
                style_isd_output = gr.Image(label="Style ISD Map")
                log_chroma_preview = gr.Image(label="Log Chromaticity Preview")
        
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
                
                step2_btn = gr.Button("▶️ Run Step 2", variant="primary")
                step2_info = gr.Textbox(label="Status", lines=4)
            
            with gr.Column():
                pre_cluster_img = gr.Image(label="Before Clustering")
                post_cluster_img = gr.Image(label="After Clustering")
                spatial_dist_img = gr.Image(label="Spatial Distribution")
        
        step2_btn.click(
            step2_process,
            inputs=[pipeline_state, clustering_method, bin_radius, n_clusters],
            outputs=[pipeline_state, pre_cluster_img, post_cluster_img, 
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
        
        step3_btn.click(
            step3_process,
            inputs=[pipeline_state, always_use_global],
            outputs=[pipeline_state, illum_vis, step3_info]
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
                    0, 100, value=100, step=5,
                    label="Rotation Percentage"
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
        
        step4_btn.click(
            step4_process,
            inputs=[pipeline_state, length_scale, rot_percent, 
                   log_transl_r, log_transl_g, log_transl_b],
            outputs=[pipeline_state, final_output, comparison_output, step4_info]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## How to Use This Pipeline
        
        ### Step 1: Load & Estimate ISD
        - Upload your content and style images
        - The ISD (Illumination Spectral Direction) maps show the estimated lighting direction
        - Choose model type and resize scale for processing
        - **Device**: Select computation device
          - `auto`: Automatically selects MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
          - `mps`: Force Apple Silicon GPU (M1/M2/M3)
          - `cuda`: Force NVIDIA GPU
          - `cpu`: Force CPU (slower but always available)
        
        ### Step 2: Cluster Materials
        - Segment the image into different material regions
        - **Greedy**: Bins pixels in log-chromaticity space by radius
        - **K-means**: Clusters pixels into fixed number of groups
        - Visualizations show how pixels are grouped
        
        ### Step 3: Estimate Illumination
        - Finds the global illumination vector
        - Estimates dark (shadow) and bright (lit) points for each material
        - These endpoints define the lighting range for each cluster
        
        ### Step 4: Apply Relighting
        - Transfer lighting from style to content
        - **Length Scale**: Compress/expand the lighting range
        - **Rotation**: How much to rotate toward style lighting (0-100%)
        - **Translation**: Shift colors in log RGB space
        
        ## Tips
        - You can rerun any step with different parameters
        - Changes in early steps require rerunning later steps
        - Check visualizations at each step to ensure quality
        """)

if __name__ == "__main__":
    demo.launch()