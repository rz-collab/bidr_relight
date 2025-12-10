"""Clean, modular relighting pipeline."""
import numpy as np
import matplotlib.pyplot as plt
import logging

from src.isd_estimation import process_image_pair
from src.bidr_util import project_to_log_chromaticity_plane, get_global_isd
from src.clustering import cluster_log_chromaticity
from src.illumination import estimate_illumination
from src.relighting import apply_relighting
from src.plotting import (
    plot_img_rgb_logrgb,
    plot_content_log_chroma,
    plot_plane,
    plot_transformed_img_logrgb,
    plot_log_chroma_plane_pre_clustering,
    plot_log_chroma_plane_post_clustering,
    plot_cluster_spatial_distribution,
    calculate_shared_limits,
    plane_view_from_normal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RelightingPipeline:
    """Modular relighting pipeline with step-by-step execution."""
    
    def __init__(self):
        self.content_data = None
        self.style_data = None
        self.log_chroma_content = None
        self.bin_masks = None
        self.bin_map = None
        self.illum_norm = None
        self.dark_points = None
        self.bright_points = None
        self.signed_dist_map = None
        self.tf_log_content = None
        
    def step1_load_and_estimate_isd(self, content_path, style_path,
                                    model_type="mock", model_path=None,
                                    resize_scale=1/4):
        """Step 1: Load images and estimate ISD maps."""
        logger.info("=== Step 1: Load and Estimate ISD ===")
        
        results = process_image_pair(
            content_path, style_path, model_type, model_path, resize_scale
        )
        
        self.content_data = results["content"]
        self.style_data = results["style"]
        
        logger.info(f"Loaded content: {self.content_data['img'].shape}")
        logger.info(f"Loaded style: {self.style_data['img'].shape}")
        
        return self.content_data, self.style_data
    
    def step2_cluster_materials(self, clustering_method="greedy",
                                bin_radius=1.0, n_clusters=4,
                                plane_offset=None):
        """Step 2: Project to log chromaticity and cluster materials."""
        logger.info("=== Step 2: Cluster Materials ===")
        
        if plane_offset is None:
            plane_offset = np.array([10.4, 10.4, 10.4])
        
        self.log_chroma_content = project_to_log_chromaticity_plane(
            self.content_data["log_img"],
            self.content_data["isd_map"],
            plane_offset=plane_offset,
            use_average_isd=False,
        )
        
        plot_log_chroma_plane_pre_clustering(
            self.log_chroma_content,
            self.content_data["isd_map"],
            self.content_data["img"],
            self.content_data["bit_depth"],
        )
        
        self.bin_masks, self.bin_map = cluster_log_chromaticity(
            self.log_chroma_content,
            method=clustering_method,
            bin_radius=bin_radius,
            n_clusters=n_clusters,
        )
        
        plot_log_chroma_plane_post_clustering(
            self.log_chroma_content,
            self.content_data["isd_map"],
            self.bin_masks,
            bin_radius if clustering_method == "greedy" else None,
        )
        plot_cluster_spatial_distribution(
            self.bin_masks,
            self.content_data["img"],
            self.content_data["bit_depth"],
        )
        
        logger.info(f"Found {len(self.bin_masks)} material clusters")
        
        return self.bin_masks
    
    def step3_estimate_illumination(self, always_use_global=True):
        """Step 3: Estimate global illumination and dark/bright points."""
        logger.info("=== Step 3: Estimate Illumination ===")
        
        (self.illum_norm, self.dark_points, self.bright_points,
         self.signed_dist_map) = estimate_illumination(
            self.content_data["log_img"],
            self.log_chroma_content,
            self.content_data["isd_map"],
            self.bin_masks,
            always_use_global=always_use_global,
        )
        
        return self.illum_norm, self.dark_points, self.bright_points
    
    def step4_apply_relighting(self, length_scale=1.0, log_transl=None,
                              rot_percent=100.0, rot_angle=None):
        """Step 4: Apply relighting transformation."""
        logger.info("=== Step 4: Apply Relighting ===")
        
        self.tf_log_content = apply_relighting(
            self.content_data["log_img"],
            self.content_data["isd_map"],
            self.style_data["isd_map"],
            self.bin_masks,
            self.dark_points,
            length_scale=length_scale,
            log_transl=log_transl,
            rot_percent=rot_percent,
            rot_angle=rot_angle,
        )
        
        return self.tf_log_content
    
    def visualize_results(self, view_isd=False):
        """Create comprehensive visualization of all results."""
        logger.info("=== Generating Visualizations ===")
        
        content_img = self.content_data["img"]
        style_img = self.style_data["img"]
        content_bit_depth = self.content_data["bit_depth"]
        style_bit_depth = self.style_data["bit_depth"]
        
        norm_content = content_img / (2**content_bit_depth - 1)
        norm_style = style_img / (2**style_bit_depth - 1)
        
        log_content = self.content_data["log_img"]
        log_style = self.style_data["log_img"]
        
        log_chroma_flat = self.log_chroma_content.reshape(-1, 3)
        log_content_flat = log_content.reshape(-1, 3)
        log_style_flat = log_style.reshape(-1, 3)
        tf_log_flat = self.tf_log_content.reshape(-1, 3)
        
        bounds = calculate_shared_limits(
            [log_style_flat, log_content_flat, log_chroma_flat, tf_log_flat],
            padding=0.2,
        )
        x_limits, y_limits, z_limits = bounds
        
        fig = plt.figure(figsize=(20, 40))
        axs = {}
        
        axs["style_img"] = fig.add_subplot(8, 2, 1)
        axs["content_img"] = fig.add_subplot(8, 2, 2)
        axs["style_rgb"] = fig.add_subplot(8, 2, 3, projection="3d")
        axs["content_rgb"] = fig.add_subplot(8, 2, 4, projection="3d")
        axs["style_log_rgb"] = fig.add_subplot(8, 2, 5, projection="3d")
        axs["content_log_rgb"] = fig.add_subplot(8, 2, 6, projection="3d")
        axs["mixed_rgb"] = fig.add_subplot(8, 2, 7, projection="3d")
        axs["mixed_log_rgb"] = fig.add_subplot(8, 2, 8, projection="3d")
        axs["content_projected_img"] = fig.add_subplot(8, 2, 9)
        axs["content_projected_log_rgb"] = fig.add_subplot(8, 2, 10, projection="3d")
        axs["clustered_content_log_rgb"] = fig.add_subplot(8, 2, 11, projection="3d")
        axs["tf_content_img"] = fig.add_subplot(8, 2, 13)
        axs["tf_content_log_rgb"] = fig.add_subplot(8, 2, 14, projection="3d")
        axs["mixed_tf_log_rgb"] = fig.add_subplot(8, 2, 15, projection="3d")
        
        log_rgb_plot_keys = [
            "style_log_rgb", "content_log_rgb", "mixed_log_rgb",
            "content_projected_log_rgb", "clustered_content_log_rgb",
            "tf_content_log_rgb", "mixed_tf_log_rgb",
        ]
        
        for key in log_rgb_plot_keys:
            axs[key].set_box_aspect([1, 1, 1])
            axs[key].set_xlim(x_limits)
            axs[key].set_ylim(y_limits)
            axs[key].set_zlim(z_limits)
        
        plot_img_rgb_logrgb(
            axs, norm_content, norm_style, log_content, log_style,
            self.bin_masks, self.dark_points, self.bright_points,
        )
        
        plot_content_log_chroma(
            axs, self.log_chroma_content, content_bit_depth, norm_content,
        )
        
        log_chroma_normal = get_global_isd(self.content_data["isd_map"])
        plot_plane(
            [axs["content_log_rgb"], axs["content_projected_log_rgb"]],
            normal=log_chroma_normal,
            point=np.array([10.4, 10.4, 10.4]),
            bounds=bounds,
        )
        
        plot_transformed_img_logrgb(
            axs, self.tf_log_content, log_content, content_bit_depth,
        )
        
        if view_isd:
            elev, azim = plane_view_from_normal(log_chroma_normal)
        else:
            elev = axs[log_rgb_plot_keys[-1]].elev
            azim = axs[log_rgb_plot_keys[-1]].azim
        
        for key in log_rgb_plot_keys:
            axs[key].view_init(elev, azim)
        
        plt.tight_layout()
        plt.show()


def relight_content_image(content_path, style_path,
                         isd_model="mock", isd_model_path=None,
                         output_path=None, resize_scale=1/4,
                         clustering_method="greedy", bin_radius=1.0,
                         n_clusters=4, shading_only=False,
                         compression_factor=0.7, view_isd=False,
                         length_scale=1.0, log_transl=None,
                         rot_percent=100.0, rot_angle=None,
                         always_use_global_illum_norm=True):
    """Complete relighting pipeline - simplified interface."""
    
    pipeline = RelightingPipeline()
    
    pipeline.step1_load_and_estimate_isd(
        content_path, style_path, isd_model, isd_model_path, resize_scale
    )
    
    pipeline.step2_cluster_materials(
        clustering_method, bin_radius, n_clusters
    )
    
    pipeline.step3_estimate_illumination(always_use_global_illum_norm)
    
    pipeline.step4_apply_relighting(
        length_scale, log_transl, rot_percent, rot_angle
    )
    
    pipeline.visualize_results(view_isd)
    
    return (pipeline.log_chroma_content, 
            [pipeline.content_data["log_img"], pipeline.style_data["log_img"]],
            [pipeline.content_data["isd_map"], pipeline.style_data["isd_map"]],
            [pipeline.content_data["img"], pipeline.style_data["img"]])