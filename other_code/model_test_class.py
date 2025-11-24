
import os
import sys
import logging
import torch
import cv2
import torch.nn.functional  as F
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from math                               import log10
from tqdm                               import tqdm
from torch.utils.data                   import DataLoader
from torchvision.utils                  import save_image, make_grid

# Add the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils.eval_metrics import compute_angular_dist, compute_psnr, compute_ssim, compute_mse, compute_distribution_stats, compute_global_angular_error


module_path = "/projects/SuperResolutionData/spectralRatio/SR_prediction/syn_data_pipeline/src/map_augmentation_src"
sys.path.append(module_path)
from isd_map_processing_classes import ISDProcessor


class ModelTester:
    """
    A class to load a model from a checkpoint, evaluate it on a test dataset,
    compute standard deviation, and visualize predictions using 2D histograms.
    """
    def __init__(self, model, checkpoint_path, test_dataset, is_linear_input=False, save_path = None, device='cuda'):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to evaluate.
            checkpoint_path (str): Path to the checkpoint file (.pth).
            test_dataset (torch.utils.data.Dataset): The test dataset.
            device (str, optional): Device to run inference on. Default is 'cuda'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.save_path = save_path
        self.outlier_info = []
        self.is_linear_input = is_linear_input
        os.makedirs(self.save_path, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized ModelTester on {self.device}")
        self.logger.info(f"Save dir set to: {self.save_path}")

        self.checkpoint = checkpoint_path
        self._load_checkpoint(checkpoint_path)
        self.isd_processor = ISDProcessor()

    def _load_checkpoint(self, checkpoint_path):
        """Loads model weights from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_annotated_isd_values(self, sr_map, mask):
        """Gets annotated pixel values using annotation mask. """
        if sr_map.shape[-2:] != mask.shape:
            raise ValueError(f"Shape mismatch: map shape {sr_map.shape} and mask shape {mask.shape}")

        if sr_map.dim() == 2:
            # Shape: [H, W] → return [N]
            return sr_map[mask.bool()].tolist()

        elif sr_map.dim() == 3:
            # Shape: [C, H, W] → want output shape: [N, C]
            values = sr_map[:, mask.bool()].T  # shape: [N, C]
            return values.tolist()
        else:
            raise ValueError(f"Unsupported tensor shape: {sr_map.shape}")
    
    def compute_chromaticity(self, vector_list):
        """Computes r and g chromaticity values from ISD vectors."""
        vectors_np = np.array(vector_list)
        sum_rgb = np.sum(vectors_np, axis=1, keepdims=True)
        sum_rgb = np.where(sum_rgb == 0, 1e-6, sum_rgb)  # Avoid division by zero

        r_values = vectors_np[:, 0] / sum_rgb[:, 0]
        g_values = vectors_np[:, 1] / sum_rgb[:, 0]
        return r_values, g_values

    def plot_2d_histogram(self, r_values, g_values, title):
        """Plots a 2D histogram of r vs. g chromaticity."""

        plot_dir = os.path.join(self.save_path, f"hist")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{title}.png")

        bins = 100
        hist, xedges, yedges = np.histogram2d(r_values, g_values, bins=bins, range=[[0, 1], [0, 1]])
        cmin = 1  # Minimum count to display
        masked_hist = np.ma.masked_less(hist, cmin)

        plt.figure(figsize=(8, 6))
        plt.imshow(masked_hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Frequency')
        plt.plot([1, 0], [0, 1], color='red', linestyle='--', linewidth=2, label="Diagonal Line")
        plt.xlabel('r (Red Chromaticity)')
        plt.ylabel('g (Green Chromaticity)')
        plt.title(title)
        plt.savefig(plot_path, format = 'png')
        plt.close()


    def test(self, plot_hist=False, outliers=False):
        self.model.eval()
        ssim_scores, psnr_scores, angular_dist, mse_scores, masked_angular_dist, global_angular_dist = [], [], [], [], [], []
        outlier_records = []
        all_targets, all_preds = [], []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Testing:")
            for i, batch in enumerate(pbar):

                # Extract images
                images = batch['image'].to(self.device)
                img_name = batch.get('img_name', None)

                # Model prediction
                predicted = self.model(images)

                # Normalize predicted outputs
                norm_pred = torch.norm(predicted, dim=1, keepdim=True)
                predicted_norm = torch.where(norm_pred != 0, predicted / norm_pred, predicted)

                # Choose correct ground truth
                ground_truth = batch['isd_map'].to(self.device)

                # Compute image-level metrics
                ssim_scores.append(compute_ssim(predicted_norm, ground_truth))
                psnr_scores.append(compute_psnr(predicted_norm, ground_truth))
                mse_scores.append(compute_mse(predicted_norm, ground_truth))

                # Angular distance works for both modes
                dist = compute_angular_dist(predicted_norm, ground_truth)
                angular_dist.append(dist)
                mask = batch['annot_mask'].to(self.device)
                masked_dist = compute_angular_dist(predicted_norm, ground_truth, mask)
                if not np.isnan(masked_dist):
                    masked_angular_dist.append(masked_dist)
                global_angular_dist.append(compute_global_angular_error(predicted_norm, ground_truth))

                outlier_records.append({
                    'index': i,
                    'error': dist,
                    'image': images[0].detach().cpu().numpy().transpose(1, 2, 0),
                    'pred': predicted_norm[0].detach().cpu().numpy().transpose(1, 2, 0),
                    'gt': ground_truth[0].detach().cpu().numpy().transpose(1, 2, 0),
                    'filename': img_name[0] if img_name is not None else 'image'
                })

                # Get spectrla ratio values for hist
                all_targets.extend(self.get_annotated_isd_values(ground_truth[0], mask[0]))
                all_preds.extend(self.get_annotated_isd_values(predicted_norm[0], mask[0]))
            
        
        # Compute and plot chromaticity distributions
        if plot_hist:
            target_r, target_g = self.compute_chromaticity(all_targets)
            pred_r, pred_g = self.compute_chromaticity(all_preds)

            self.plot_2d_histogram(target_r, target_g, "test_isd_hist")
            self.plot_2d_histogram(pred_r, pred_g, "pred_isd_hist")

        # Save outlier results
        if outliers:
            self.save_outliers(outlier_records)

        # Average results
        angular_dist_stats = compute_distribution_stats(angular_dist)
        masked_angular_dist_stats = compute_distribution_stats(masked_angular_dist)
        global_angular_dist_stats = compute_distribution_stats(global_angular_dist)
        metrics = {
            "Angular_Dist": float(np.mean(angular_dist)) if angular_dist else float('nan'),
            "Masked_angular_Dist": float(np.mean(masked_angular_dist)) if masked_angular_dist else float('nan'),
            "global_angular_dist": float(np.mean(global_angular_dist)) if global_angular_dist else float('nan'),
            "angular_dist_stats": angular_dist_stats,
            "masked_angular_dist_stats": masked_angular_dist_stats,
            "global_angular_dist_stats": global_angular_dist_stats,
            "SSIM": float(np.mean(ssim_scores)) if ssim_scores else float('nan'),
            "PSNR": float(np.mean(psnr_scores)) if psnr_scores else float('nan'),
            "MSE": float(np.mean(mse_scores)) if mse_scores else float('nan')
            }

        return metrics

    def save_outliers(self, outlier_records, k=20):
        self.logger.info("Performing outlier analysis...")
        os.makedirs(os.path.join(self.save_path, "outliers", "best"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "outliers", "worst"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "outliers", "median"), exist_ok=True)

        sorted_records = sorted(outlier_records, key=lambda x: x['error'])
        best = sorted_records[:k]
        worst = sorted_records[-k:]
        n = len(sorted_records)
        start = (n - k) // 2
        end = start + k
        median = sorted_records[start:end]
        self.logger.info(f"median {k}...")
        self.outlier_vis(median, 'median')
        self.logger.info(f"best {k}...")
        self.outlier_vis(best, "best")
        self.logger.info(f"worst {k}..")
        self.outlier_vis(worst, "worst")
        self.logger.info(f"Saved top {k} best and worst samples to {os.path.join(self.save_path, 'outliers')}")
        self.save_best_prediction_grids(best, k=5)

    def outlier_vis(self, records, subdir):
        for rank, rec in enumerate(records):
            img_lin = self.get_8bit_linear(rec["image"], is_linear_input=self.is_linear_input)
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))
            axs[0, 0].imshow(self.linear_to_srgb(img_lin / 255))
            axs[0, 0].set_title("Input Image")
            axs[0, 1].imshow(np.clip(rec["pred"], 0, 1))
            axs[0, 1].set_title("Predicted ISD")
            axs[0, 2].imshow(np.clip(rec["gt"], 0, 1))
            axs[0, 2].set_title("Ground Truth ISD")
            axs[1, 0].imshow(self.get_chroma(img_lin))
            axs[1, 0].set_title("RG Chroma")
            axs[1, 1].imshow(self.get_log_chroma(rec["image"], rec["pred"], is_linear_input=self.is_linear_input))
            axs[1, 1].set_title("Predicted Log Chroma")
            axs[1, 2].imshow(self.get_log_chroma(rec["image"], rec["gt"], is_linear_input=self.is_linear_input))
            axs[1, 2].set_title("Ground Truth Log Chroma")
            for ax in axs.flat:
                ax.axis("off")
            fig.suptitle(f"{rec['filename']} | Angular Error: {rec['error']:.2f}°", fontsize=12)
            out_path = os.path.join(self.save_path, "outliers", subdir, f"{rank:02d}_{rec['filename']}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

    def save_best_prediction_grids(self, best_records, k=5):
        """
        Selects the top-k best predictions by angular error and saves
        image grids of ground truth ISD maps and predicted ISD maps.

        Saves:
            <save_path>/best_pred_grid.png
            <save_path>/best_gt_grid.png
        """

        # Sort by angular error (ascending = best)
        top_k_records = best_records[:k]

        preds = []
        gts = []

        for rec in top_k_records:
            pred = rec["pred"]  # shape (H,W,3), np
            gt   = rec["gt"]    # shape (H,W,3), np

            # Convert to CHW and torch.tensor
            pred_t = torch.from_numpy(pred).permute(2, 0, 1).float()
            gt_t   = torch.from_numpy(gt).permute(2, 0, 1).float()

            preds.append(pred_t)
            gts.append(gt_t)

        pred_grid = make_grid(preds, nrow=1, normalize=True, value_range=(0, 1))
        gt_grid   = make_grid(gts,   nrow=1, normalize=True, value_range=(0, 1))

        out_pred = os.path.join(self.save_path, "best_pred_grid.png")
        out_gt   = os.path.join(self.save_path, "best_gt_grid.png")

        save_image(pred_grid, out_pred)
        save_image(gt_grid, out_gt)

        self.logger.info(f"Saved best prediction grid to {out_pred}")
        self.logger.info(f"Saved best ground truth grid to {out_gt}")

    def log_to_linear(self, log_img: np.ndarray) -> np.ndarray:
        """
        Converts a log-RGB image back to linear 16-bit space.

        Parameters:
        -----------
        log_img : np.ndarray
            Log-transformed image.

        Returns:
        --------
        np.ndarray
            Reconstructed linear image (float32).
        """
        return np.exp(log_img).astype(np.float32)
    

    def get_log_chroma(self, image, isd_map, is_linear_input: bool = False, anchor=10.4):
        """
        Projects each pixel to a plane orthogonal to the ISD for that pixel. 
        """
        assert image.min() >= 0.0 and image.max() <= 1.0, f"Expected values in [0, 1]. Got range [{image.min():.4f}, {image.max():.4f}]."

        if is_linear_input:
            # Convert linear-normalized → 16-bit
            linear_16 = image * 65535.0

            # Zero-mask to avoid log(0)
            zero_mask = (linear_16 <= 0)
            linear_16_masked = np.where(zero_mask, 1e-6, linear_16)

            # Compute log and normalize to [0,1]
            log_max = np.log(65535.0)
            log_img = np.log(linear_16_masked) / log_max
            log_img = np.clip(log_img, 0.0, 1.0) * 11.1

        else:
            # Input is already normalized log-RGB (range [0,1] means [0, 11.1] in log)
            log_img = image * 11.1  # Restore full-range log-scale
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1 + 1e-6, f"Log image min and/or max value sout of range | min={log_img.min()} | max={log_img.max()}"

        shifted_log_rgb = log_img - anchor
        dot_product_map = np.einsum('ijk,ijk->ij', shifted_log_rgb, isd_map)

        # Reshape the dot product to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]

        # Multiply with the ISD vector to get the projected RGB values
        projection = dot_product_reshaped * isd_map

        # Subtract the projection from the shifted values to get plane-projected values
        projected_rgb = shifted_log_rgb - projection

        # Shift the values back by adding the anchor point
        log_chroma = projected_rgb + anchor
        
        linear_chroma = np.exp(log_chroma).astype(np.float32)
        img_clipped = np.clip(linear_chroma, 0, 65535)
        img_normalized = (img_clipped / 256.0).astype(np.uint8)
        assert img_normalized.min() >= 0 and img_normalized.max() <= 255 + 1e-3, \
            f"Image values out of expected range: {img_normalized.min()} to {img_normalized.max()}"
        return img_normalized


    def get_chroma(self, image):

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        sum_rgb = r + g + b
        sum_rgb[sum_rgb == 0] = 1e-8

        r_chroma = r / sum_rgb
        g_chroma = g / sum_rgb
        b_chroma = b / sum_rgb

        chroma_img = np.stack([r_chroma, g_chroma, b_chroma], axis=-1)
        chroma_img = np.clip(chroma_img, 0.0, 1.0)
        return chroma_img


    def get_8bit_linear(self, img_rgb, is_linear_input: bool = False):
        """
        Convert a 16-bit image in log-RGB or linear-RGB format into an 8-bit linear RGB image.

        Parameters
        ----------
        img_rgb : np.ndarray
            - log-RGB values in the range [0, 1] (if already_linear=False), or
            - linear RGB values in [0, 65535] (if already_linear=True).
        is_linear_input : bool, optional (default=False)
            If False (default), the input is assumed to be log-RGB and will be converted
            to linear via exponentiation.  
            If True, the input is assumed to already be linear RGB and will be used
            directly without log→linear conversion.

        Returns
        -------
        img_8bit : np.ndarray
            8-bit RGB image of shape (H, W, 3), dtype uint8.
        """

        # Validate input
        assert img_rgb.min() >= 0.0 and img_rgb.max() <= 1.0, \
            f"Expected RGB values in [0, 1], got [{img_rgb.min():.4f}, {img_rgb.max():.4f}]"

        if not is_linear_input:
            # Convert log-RGB [0,1] → log-scale [0,11.1] → linear
            log_rgb = img_rgb * 11.1
            linear_rgb = np.exp(log_rgb)
            # Clip to 16-bit range
            linear_norm = np.clip(linear_rgb, 0, 65535) / 65535
            assert np.isfinite(linear_norm).all(), "Non-finite values after exp/normalization."
        else:
            # Validate linear input
            linear_norm = np.clip(img_rgb, 0.0, 1.0)

        # Normalize to 8-bit
        img_8bit = (255 * (linear_norm)).astype(np.uint8)

        return img_8bit

    def linear_to_srgb(self, linear_rgb):
        """
        Converts linear RGB values to sRGB using the standard sRGB transfer function.

        Args:
            linear_rgb (np.ndarray): Input image, float32 or float64, values in [0, 1].

        Returns:
            np.ndarray: sRGB image, float32, values in [0, 1].
        """
        linear_rgb = np.clip(linear_rgb, 0, 1)
        threshold = 0.0031308
        below = linear_rgb <= threshold
        above = ~below

        srgb = np.zeros_like(linear_rgb)
        srgb[below] = 12.92 * linear_rgb[below]
        srgb[above] = 1.055 * (linear_rgb[above] ** (1/2.4)) - 0.055
        return srgb

    def save_model_test_summary(self, metrics=None, test_name=None):
        """
        Save a summary of model results and validation metrics to a text file.

        Args:
            metrics (dict): Dictionary containing evaluation metrics for the final validation set.
            test_name (str): Name of test

        Returns:
            None
        """
        # Ensure metrics are not None to prevent errors
        ssim = f"{metrics['SSIM']:.4f}" if metrics and "SSIM" in metrics and metrics["SSIM"] is not None else "N/A"
        psnr = f"{metrics['PSNR']:.4f} dB" if metrics and "PSNR" in metrics and metrics["PSNR"] is not None else "N/A"
        mse = f"{metrics['MSE']:.4f}" if metrics and "MSE" in metrics and metrics["MSE"] is not None else "N/A"
        angular_dist = f"{metrics['Angular_Dist']:.2f} degrees" if metrics and "Angular_Dist" in metrics and metrics["Angular_Dist"] is not None else "N/A" 
        masked_angular_dist = f"{metrics['Masked_angular_Dist']:.2f} degrees" if metrics and "Masked_angular_Dist" in metrics and metrics["Masked_angular_Dist"] is not None else "N/A" 
        global_angular_dist = f"{metrics['global_angular_dist']:.2f} degrees" if metrics and "global_angular_dist" in metrics and metrics["global_angular_dist"] is not None else "N/A" 
        angular_dist_stats = metrics['angular_dist_stats']
        angular_stats_str = "\n        ".join(f"--{k}: {v:.2f}" for k, v in angular_dist_stats.items())
        masked_angular_dist_stats = metrics['masked_angular_dist_stats']
        masked_angular_stats_str = "\n        ".join(f"--{k}: {v:.2f}" for k, v in masked_angular_dist_stats.items())
        global_angular_dist_stats = metrics['global_angular_dist_stats']
        global_angular_dist_stats_str = "\n        ".join(f"--{k}: {v:.2f}" for k, v in global_angular_dist_stats.items())

        
        # Create summary content
        summary = f"""
        Model Testing Summary
        ======================
        Test Name: {test_name}

        Training Summary
        -----------------
        Model: {self.model.__class__.__name__}
        Checkpoint: {self.checkpoint}

        Summary Metrics:
        --------------------------------
        SSIM (Structural Similarity Index): {ssim}
        PSNR (Peak Signal-to-Noise Ratio): {psnr}
        MSE (Mean Squared Error): {mse}
        Angular Distance: {angular_dist}
        Masked Angular Distance: {masked_angular_dist}
        Global Angular Distance: {global_angular_dist}

        Angular Distance Stats:
        -----------------------
        {angular_stats_str}

        Masked Angular Distance Stats:
        ------------------------------
        {masked_angular_stats_str}

        Global Angular Distance Stats:
        -------------------------------
        {global_angular_dist_stats_str}
        """

        self.logger.info(summary)

        # Save summary to a text file
        summary_file_path = os.path.join(self.save_path, f"{test_name}_summary.txt")
        self.logger.info(f"Attempting to save summary to: {summary_file_path}")
        try:
            with open(summary_file_path, "w") as file:
                file.write(summary.strip())
            self.logger.info("Save successful!")
        except Exception as e:
            self.logger.error(f"Unable to save summary due to error: {e}")

if __name__ == "__main__":
    pass

