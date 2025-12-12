"""
Batch relighting script: relights all content/style image pairs in folders or single images, saves outputs, and plots cluster endpoint colors.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from src.pipeline import RelightingPipeline
from src.image_util import normalized_linear_to_srgb, log_to_linear


def plot_cluster_endpoints_linear(dark_points, bright_points, out_path=None):
    """Plot the linear RGB colors of each cluster's dark and bright points."""
    n = len(dark_points)
    fig, ax = plt.subplots(1, n, figsize=(2*n, 2))
    if n == 1:
        ax = [ax]
    for i in range(n):
        patch = np.stack([
            dark_points[i],
            bright_points[i],
        ], axis=0).reshape(1, 2, 3)
        patch = np.clip(patch, 0, 1)
        patch_srgb = normalized_linear_to_srgb(patch)
        ax[i].imshow(patch_srgb)
        ax[i].set_title(f"Cluster {i}")
        ax[i].axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


def relight_and_save(content_path, style_path, output_dir, args):
    pipeline = RelightingPipeline()
    pipeline.step1_load_and_estimate_isd(
        content_path, style_path, args.isd_model, args.isd_model_path, args.resize_scale
    )
    pipeline.step2_cluster_materials(
        args.clustering_method, args.bin_radius, args.n_clusters
    )
    pipeline.step3_estimate_illumination(always_use_global=args.always_use_global_illum_norm)
    pipeline.step4_apply_relighting(
        length_scale=args.length_scale, log_transl=args.log_transl, rot_percent=args.rot_percent
    )
    tf_linear = log_to_linear(pipeline.tf_log_content)
    out_name = f"relit_{os.path.splitext(os.path.basename(content_path))[0]}__{os.path.splitext(os.path.basename(style_path))[0]}.png"
    out_path = os.path.join(output_dir, out_name)
    imsave(out_path, (tf_linear).astype(np.uint16))
    print(f"Saved relit image to {out_path}")

    # Save relit image in sRGB
    tf_srgb = normalized_linear_to_srgb(tf_linear)
    srgb_out_path = out_path.replace('.png', '_srgb.png')
    imsave(srgb_out_path, tf_srgb)
    print(f"Saved relit sRGB image to {srgb_out_path}")

    # Plot cluster endpoint colors
    plot_cluster_endpoints_linear(pipeline.dark_points, pipeline.bright_points,
                                 out_path=os.path.join(output_dir, out_name.replace('.png', '_clusters.png')))


def main():
    parser = argparse.ArgumentParser(description="Batch relighting for content/style images.")
    parser.add_argument('--content', type=str, required=True, help='Content image path or folder')
    parser.add_argument('--style', type=str, required=True, help='Style image path or folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--isd_model', type=str, default='mock', help='ISD model type (mock/unet)')
    parser.add_argument('--isd_model_path', type=str, default=None, help='Path to ISD model weights')
    parser.add_argument('--resize_scale', type=float, default=1/4)
    parser.add_argument('--clustering_method', type=str, default='greedy')
    parser.add_argument('--bin_radius', type=float, default=1.0)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--always_use_global_illum_norm', action='store_true')
    parser.add_argument('--length_scale', type=float, default=1.0)
    parser.add_argument('--log_transl', type=float, nargs=3, default=None)
    parser.add_argument('--rot_percent', type=float, default=100.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Handle folders or single files
    if os.path.isdir(args.content) and os.path.isdir(args.style):
        content_files = sorted([f for f in os.listdir(args.content) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
        style_files = sorted([f for f in os.listdir(args.style) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
        # Match by filename (without extension)
        content_basenames = {os.path.splitext(f)[0]: f for f in content_files}
        style_basenames = {os.path.splitext(f)[0]: f for f in style_files}
        common_basenames = sorted(set(content_basenames.keys()) & set(style_basenames.keys()))
        for base in common_basenames:
            c_path = os.path.join(args.content, content_basenames[base])
            s_path = os.path.join(args.style, style_basenames[base])
            relight_and_save(c_path, s_path, args.output, args)
    else:
        relight_and_save(args.content, args.style, args.output, args)

if __name__ == "__main__":
    main()
