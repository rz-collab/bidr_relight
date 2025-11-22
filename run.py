"""
Script for visualizing application of ISD predictions vs gamma correction for shadow removal. 
"""


import os
import cv2
import numpy as np
import logging
import torch
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.unet_models import ResNet50UNet

####################################################################################################################################################################
# CLASSES
####################################################################################################################################################################

class imgProcessor:
    """
    Image Processor class for visualizing and manipulating spectral images
    using Illuminant Spectral Direction (ISD) maps.

    Attributes:
    -----------
    image : np.ndarray
        The loaded 16-bit input image.

    sr_map : np.ndarray
        The normalized ISD map of the same size as the image.

    roi : tuple or None
        The selected region of interest in (x, y, w, h) format.
    """

    def __init__(self, image: np.ndarray, sr_map: np.ndarray, filename: str = "None"):
        """
        Initializes the processor with an image and its corresponding ISD map.

        Parameters:
        -----------
        image : str
            16-bit input image.
        sr_map : str
            16-bit ISD map, normalized to [0, 1].
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.filename = filename
        self.image = image
        if self.image is None:
            raise ValueError(f"Failed to set image")
        
        self.sr_map = sr_map
        if self.sr_map is None:
            raise ValueError(f"Failed to set spectral ratio map")
        self.logger.info(f"Image: {filename}")
        self.logger.info(f"Image loaded | Shape: {self.image.shape} | Max: {self.image.max()} | Min: {self.image.min()}")
        self.logger.info(f"Map loaded  | Shape: {self.sr_map.shape} | Max: {self.sr_map.max()} | Min: {self.sr_map.min()}")

    def convert_img_to_log_space(self, linear_img: np.ndarray) -> np.ndarray:
        """
        Converts a 16-bit linear image to log-RGB space.
        Pixels with value 0 remain 0 in log space.

        Parameters:
        -----------
        linear_img : np.ndarray
            Input image in 16-bit linear space.

        Returns:
        --------
        np.ndarray
            Log-RGB image as float32.
        """
        log_img = np.zeros_like(linear_img, dtype=np.float32)
        log_img[linear_img > 0] = np.log(linear_img[linear_img > 0])
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        return log_img

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

    def convert_16bit_to_8bit(self, img: np.ndarray) -> np.ndarray:
        """
        Converts a 16-bit image to an 8-bit image for display.

        Parameters:
        -----------
        img : np.ndarray
            Input 16-bit or float image.

        Returns:
        --------
        np.ndarray
            Output image in 8-bit (uint8).
        """
        img_clipped = np.clip(img, 0, 65535)
        img_normalized = (img_clipped / 256.0).astype(np.uint8)
        return img_normalized

    def select_roi(self, img: np.ndarray):
        """
        Allows the user to interactively select an ROI using OpenCV.

        Parameters:
        -----------
        img : np.ndarray
            The image to display for ROI selection.

        Returns:
        --------
        tuple or None:
            (x, y, w, h) ROI tuple and the ROI image, or (None, None) if no selection.
        """
        self.logger.info("Select ROI using mouse...\n")
        roi = cv2.selectROI(f"Select ROI | File: {self.filename}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), showCrosshair=False, fromCenter=False)
        cv2.destroyWindow(f"Select ROI | File: {self.filename}")

        if roi == (0, 0, 0, 0):
            self.logger.info("No ROI selected.\n")
            return None, None

        x, y, w, h = roi
        self.logger.info(f"ROI selected: x={x}, y={y}, w={w}, h={h}\n")
        return roi, img[y:y+h, x:x+w]

    # def shift_pixels_along_isd(self, log_image: np.ndarray, sr_map: np.ndarray, roi: tuple, distance: float) -> np.ndarray:
    #     """
    #     Shifts pixels within an ROI along their ISD vector in log-RGB space.

    #     Parameters:
    #     -----------
    #     log_image : np.ndarray
    #         Input log-RGB image.
    #     isd_map : np.ndarray
    #         Normalized ISD map.
    #     roi : tuple
    #         ROI in the form (x, y, w, h).
    #     distance : float
    #         Scalar distance to shift pixels along ISD.

    #     Returns:
    #     --------
    #     np.ndarray
    #         Log image with ROI region shifted.
    #     """
    #     shifted = log_image.copy()
    #     x, y, w, h = roi

    #     # Normalize ISD vectors to unit length per pixel
    #     norm = np.linalg.norm(sr_map, axis=2, keepdims=True)
    #     norm[norm == 0] = 1.0  
    #     isd_map = sr_map / norm

    #     shifted[y:y+h, x:x+w] += distance * isd_map[y:y+h, x:x+w]
    #     return shifted

    def shift_pixels_along_isd(
        self,
        log_image: np.ndarray,
        sr_map: np.ndarray,
        roi: tuple = None,
        distance: float = 0.0,
        mask: np.ndarray = None) -> np.ndarray:
        """
        Shifts pixels within a specified ROI or binary mask along their ISD vectors in log-RGB space.

        Parameters:
        -----------
        log_image : np.ndarray
            Input log-RGB image.
        sr_map : np.ndarray
            Normalized ISD map.
        roi : tuple, optional
            ROI in the form (x, y, w, h).
        distance : float
            Scalar distance to shift pixels along ISD.
        mask : np.ndarray, optional
            Boolean or binary mask of shape (H, W). Overrides ROI if provided.

        Returns:
        --------
        np.ndarray
            Log image with shifted region.
        """
        shifted = log_image.copy()

        # Normalize ISD vectors
        norm = np.linalg.norm(sr_map, axis=2, keepdims=True)
        norm[norm == 0] = 1.0
        isd_map = sr_map / norm

        if mask is not None:
            if mask.shape != log_image.shape[:2]:
                raise ValueError(f"Mask must match image shape (H, W). Mask={mask.shape} | Image={log_image.shape[:2]}")
            # shifted[mask] += distance * isd_map[mask]
            ys, xs, = np.where(mask)
            shifted[ys, xs, :] += distance * isd_map[ys, xs, :]
        elif roi is not None:
            x, y, w, h = roi
            shifted[y:y+h, x:x+w] += distance * isd_map[y:y+h, x:x+w]
        else:
            raise ValueError("Either `roi` or `mask` must be provided.")

        return shifted
    
    def save_image(self, image: np.ndarray, save_path: str):
        """
        Saves an image to disk.

        Parameters:
        -----------
        image : np.ndarray
            The image to save (can be 8-bit, 16-bit, or float32).
        save_path : str
            Output file path (e.g., 'output.png' or 'output.tif').
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success = cv2.imwrite(save_path, image)
        if success:
            self.logger.info(f"Image saved to {save_path}\n")
        else:
            self.logger.error(f"Failed to save image to {save_path}\n")

    # def interactive_shift_viewer(self, roi=None):
    #     """
    #     Launches an OpenCV window with a trackbar to interactively shift pixels
    #     along ISD vectors within a selected ROI.

    #     - Displays the image in 8-bit linear RGB.
    #     - Trackbar allows shifting from -5.0 to +5.0 in 0.1 increments.
    #     """
    #     log_img = self.convert_img_to_log_space(self.image)
    #     image_8bit = self.convert_16bit_to_8bit(self.image)
    #     if roi is None:
    #         roi, _ = self.select_roi(image_8bit)
    #     if roi is None:
    #         self.logger.warning("No ROI selected. Exiting interactive viewer.\n")
    #         return

    #     window_name = f"Interactive ISD Shift | File: {self.filename}"
    #     cv2.namedWindow(window_name)

    #     def on_trackbar(val):
    #         distance = (val - 50) / 10.0  # Range: [-5.0, 5.0]
    #         shifted_log = self.shift_pixels_along_isd(log_img, self.sr_map, roi, distance)
    #         linear = self.log_to_linear(shifted_log)
    #         vis_img = self.convert_16bit_to_8bit(linear)
    #         self.last_output_img = vis_img
    #         cv2.imshow(window_name, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    #     cv2.createTrackbar("Distance", window_name, 50, 100, on_trackbar)
    #     on_trackbar(50)  # Initialize with no shift

    #     self.logger.info("Adjust trackbar. Press any key to close.\n")
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     self.save_image(self.last_output_img, f"results/sr_corrected_result_{self.filename}.png")

    def interactive_shift_viewer(self, roi=None, mask=None):
        """
        Launches an OpenCV window with a trackbar to interactively shift pixels
        along ISD vectors within a selected ROI or binary mask.

        - Displays the image in 8-bit linear RGB.
        - Trackbar allows shifting from -5.0 to +5.0 in 0.1 increments.
        - If `mask` is provided, it overrides the ROI and applies shifts to masked pixels.
        """
        log_img = self.convert_img_to_log_space(self.image)
        image_8bit = self.convert_16bit_to_8bit(self.image)

        if mask is not None:
            if mask.shape[:2] != self.image.shape[:2]:
                raise ValueError("Mask must match the shape of the image (H, W).")
            mask = mask.astype(np.uint8)
        else:
            if roi is None:
                roi, _ = self.select_roi(image_8bit)
            if roi is None:
                self.logger.warning("No ROI selected. Exiting interactive viewer.\n")
                return

        window_name = f"Interactive ISD Shift | File: {self.filename}"
        cv2.namedWindow(window_name)

        def on_trackbar(val):
            distance = (val - 50) / 10.0  # Range: [-5.0, 5.0]
            if mask is not None:
                shifted_log = self.shift_pixels_along_isd(log_img, self.sr_map, mask=mask, distance=distance)
            else:
                shifted_log = self.shift_pixels_along_isd(log_img, self.sr_map, roi=roi, distance=distance)

            linear = self.log_to_linear(shifted_log)
            vis_img = self.convert_16bit_to_8bit(linear)
            self.last_output_img = vis_img
            cv2.imshow(window_name, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        cv2.createTrackbar("Distance", window_name, 50, 100, on_trackbar)
        on_trackbar(50)  # Initialize with no shift

        self.logger.info("Adjust trackbar. Press any key to close.\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.save_image(self.last_output_img, f"results/sr_corrected_result_{self.filename}.png")

    def interactive_gamma_viewer(self):
        """
        Launches an OpenCV window with a trackbar to interactively apply gamma correction
        to a selected ROI. The correction is applied in linear RGB space on the original image.

        - Works with 16-bit images converted to 8-bit for visualization.
        - Gamma range is [0.1, 3.0] in 0.01 increments.
        - Only the selected ROI is affected.
        """
        # Convert the original image to 8-bit linear RGB for display and editing
        image_8bit = self.convert_16bit_to_8bit(self.image)
        roi, _ = self.select_roi(image_8bit)
        if roi is None:
            self.logger.info("No ROI selected. Using entire image.")
            h, w = image_8bit.shape[:2]
            roi = (0, 0, w, h)

        x, y, w, h = roi
        window_name = f"Interactive Gamma Correction | File: {self.filename}"
        cv2.namedWindow(window_name)

        def on_trackbar(val):
            # Map trackbar [1, 300] to gamma [0.1, 3.0]
            gamma = max(val / 100.0, 0.1)

            # Apply gamma correction to ROI
            img_corrected = image_8bit.copy()
            roi_patch = img_corrected[y:y+h, x:x+w].astype(np.float32) / 255.0
            roi_patch = np.power(roi_patch, gamma)
            img_corrected[y:y+h, x:x+w] = np.uint8(np.clip(roi_patch * 255, 0, 255))

            self.last_output_img = img_corrected
            cv2.imshow(window_name, cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR))

        # Trackbar from 10 → 300 → gamma [0.1 – 3.0]
        cv2.createTrackbar("Gamma x100", window_name, 100, 300, on_trackbar)
        on_trackbar(100)  # Initial gamma = 1.0

        self.logger.info("Adjust gamma with trackbar. Press any key to exit.\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.save_image(self.last_output_img, f"results/gamma_corrected_result_{self.filename}.png")
        return roi
    
    
    def get_log_chroma(self, anchor=100) -> None:
        """
        Projects each pixel to a plane orthogonal to the ISD for that pixel. 
        """
        log_img = self.convert_img_to_log_space(self.image)
        shifted_log_rgb = log_img - anchor
        dot_product_map = np.einsum('ijk,ijk->ij', shifted_log_rgb, self.sr_map)

        # Reshape the dot product to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]

        # Multiply with the ISD vector to get the projected RGB values
        projection = dot_product_reshaped * self.sr_map

        # Subtract the projection from the shifted values to get plane-projected values
        projected_rgb = shifted_log_rgb - projection

        # Shift the values back by adding the anchor point
        projected_rgb += anchor
        
        linear_chroma = self.log_to_linear(projected_rgb)
        chroma_8bit = self.convert_16bit_to_8bit(linear_chroma)
        self.save_image(chroma_8bit, f"results/{self.filename}log_chromaticity.png")

        cv2.imshow('chroma', cv2.cvtColor(chroma_8bit, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    
####################################################################################################################################################################

class ISDMapEstimator:
    def __init__(self, model: object, model_path: str, device: str = "cpu"):
        """
        Initializes the ISDModelPredictor with a pretrained model.

        Parameters:
        -----------
        model_path : str
            Path to the `.pth` file containing the saved model state_dict.
        device : str
            Device to run the model on ('cpu' or 'cuda').
        logger : logging.Logger
            Optional logger instance.
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Device = {self.device}")
        self.sr_map = None

        # Load model
        self.model = model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"'model_state_dict' key not found in checkpoint at: {model_path}")

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load model state_dict: {e}")
        
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Converts a 16-bit linear RGB image to a normalized log-space tensor for model input.

        Parameters:
        -----------
        image : np.ndarray
            Input image in 16-bit format (H, W, 3).

        Returns:
        --------
        torch.Tensor
            Normalized tensor of shape (1, 3, H, W) in log-space.
        """
        log_img = np.zeros_like(image, dtype=np.float32)
        log_img[image > 0] = np.log(image[image > 0])
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        log_img = log_img / 11.1  # normalize to [0, 1]
        input_tensor = torch.from_numpy(log_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        self.logger.info(f"Input Tensor | Shape: {input_tensor.shape} | Dtype: {input_tensor.dtype}\n")
        return input_tensor

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs inference on the input image and returns a normalized ISD map.

        Parameters:
        -----------
        image : np.ndarray
            Input image (H, W, 3), typically 16-bit RGB.

        Returns:
        --------
        np.ndarray
            Normalized ISD map of shape (H, W, 3) in float32.
        """
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            output = self.model(input_tensor)

        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        norm = np.linalg.norm(output_np, axis=2, keepdims=True).astype(np.float32)
        norm[norm == 0] = 1.0
        self.sr_map = output_np / norm

        self.logger.info(f"Map | Shape: {self.sr_map.shape} | Dtype: {self.sr_map.dtype}\n")
        return self.sr_map
    
    def get_pixelwise_angular_dist(self, sr_map_target):

        # Flatten to (-1, 3)
        pred_flat = self.sr_map.reshape(-1, 3)
        target_flat = sr_map_target.reshape(-1, 3)

        pred_norms = np.linalg.norm(pred_flat, ord=2, axis=1)
        target_norms = np.linalg.norm(target_flat, ord=2, axis=1)

        # Valid vectors: both pred and target must be non-zero
        valid_mask = (pred_norms > 0) & (target_norms > 0)

        if not np.any(valid_mask):
            raise ValueError("No valid vectors found to compute angular error.")

        # Normalize only valid entries
        pred_unit = pred_flat[valid_mask] / pred_norms[valid_mask, np.newaxis]
        target_unit = target_flat[valid_mask] / target_norms[valid_mask, np.newaxis]

        # Cosine similarity
        cos_sim = np.sum(pred_unit * target_unit, axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # ensure within valid range

        # Angular error in degrees
        angles = np.arccos(cos_sim) * (180.0 / np.pi)
        return angles.mean()


####################################################################################################################################################################
# HELPER FUNCTION    
####################################################################################################################################################################
def crop_to_even_dims(image: np.ndarray) -> np.ndarray:
    """
    Crops an image so that both height and width are even numbers.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (H, W, C) or (H, W)

    Returns:
    --------
    cropped : np.ndarray
        Image cropped to even height and width.
    """
    h, w = image.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    return image[:new_h, :new_w]

def center_crop_to_divisible(image, factor: int = 32):
    h, w = image.shape[:2]
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    return image[start_h:start_h + new_h, start_w:start_w + new_w]

def parse_args():
    parser = argparse.ArgumentParser(description="Test script with optional parameters")

    parser.add_argument('--log-chroma', action='store_true',
                        help='Use logarithmic chromaticity (default: False)')
    parser.add_argument('--use-model', action='store_true',
                        help='Use the trained model for prediction (default: False)', default=True)
    parser.add_argument('--show-gamma', action='store_true',
                        help='Display gamma information or plots (default: False)')
    parser.add_argument('--image', type=str, default="data/compressed/bottle_sun.tiff",
                        help='Optional image name to use for logging or output (default: None)')
    return parser.parse_args()

####################################################################################################################################################################
# MAIN
####################################################################################################################################################################

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logger = logging.getLogger(__name__)
    # Params
    args = parse_args()
    log_chroma = args.log_chroma
    use_model = args.use_model
    show_gamma = args.show_gamma
    img_name = args.image
    #mask = cv2.imread('dev/lee_samuel_001_mask.png', cv2.IMREAD_UNCHANGED)
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    # print(f"Mask shape: {mask.shape}")
    # mask = crop_to_even_dims(mask)
    # mask = center_crop_to_divisible(mask)
    # print(f"Mask shape: {mask.shape}")
    mask=None

    # Init model
    model = ResNet50UNet(
        in_channels=3, 
        out_channels=3, 
        pretrained=False, 
        checkpoint=None, 
        se_block=True)
    
    # Init estimator wrapper
    model_path = "src/weights/UNET_last_model.pth"
    # model_path = "model/unet.pth"
    device = 'cpu'
    estimator = ISDMapEstimator(
            model = model,
            model_path = model_path,
            device = device
            )
    
    image_dir_path = 'data/'
    if img_name:
        tif_filenames = [img_name]
    else:
        tif_filenames = [f.stem for f in Path(image_dir_path).glob("*.tif")]
    for img_name in tif_filenames:
        logger.info(f"IMAGE FILE NAME: {img_name}")
        img_path = f'{img_name}'
        #sr_map_path = f'{img_name}_isd.tiff'

        # Import image
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image from {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_to_even_dims(image)
        image = center_crop_to_divisible(image)
        logger.info(f" Image | SHape: {image.shape} | Dtype: {image.dtype}")
        # plt.imshow(image / 65535)
        # plt.title("Image")
        # plt.show()
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Image', image_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Import spectral ratio map
        #sr_map = cv2.imread(sr_map_path, cv2.IMREAD_UNCHANGED)
        #if sr_map is None:
        #    raise ValueError(f"Failed to load map from {sr_map_path}")
        # sr_map = cv2.cvtColor(sr_map, cv2.COLOR_BGR2RGB)
        #logger.info(f" Annotated Map | Shape: {sr_map.shape} | Dtype: {sr_map.dtype}")
        #logger.info(f" Annotated Map | Max: {sr_map.max(axis=(0,1))} | Min: {sr_map.min(axis=(0,1))}")
        #sr_map = crop_to_even_dims(sr_map)
        #sr_map = center_crop_to_divisible(sr_map)
        #sr_map = sr_map.astype(np.float32) / 65535  # Normalize
        # plt.imshow(sr_map)
        # plt.title("Ground Truth Map")
        # plt.show()
        #logger.info(f" Annotated Map | Shape: {sr_map.shape} | Dtype: {sr_map.dtype}")
        #logger.info(f" Annotated Map | Max: {sr_map.max(axis=(0,1))} | Min: {sr_map.min(axis=(0,1))}")


        # Run inference on model
        if use_model:
            sr_map_pred = estimator.predict(image)
            logger.info(f" Predicated Map | Shape: {sr_map_pred.shape} | Dtype: {sr_map_pred.dtype}")
            logger.info(f" Predicated Map | Max: {sr_map_pred.max(axis=(0,1))} | Min: {sr_map_pred.min(axis=(0,1))}")

            #mean_ang_dist = estimator.get_pixelwise_angular_dist(sr_map)
            #logger.info(f"Mean pixel-wise angular distance: {mean_ang_dist:.2f} degrees")
            # plt.imshow(sr_map_pred)
            # plt.title("Predicted Map")
            # plt.show()
            sr_map = sr_map_pred

        # Main program
        roi=None
        processor = imgProcessor(image, sr_map, filename=img_name)
        if log_chroma:
            processor.get_log_chroma(anchor=10.4)
        if show_gamma:
            roi = processor.interactive_gamma_viewer()
        processor.interactive_shift_viewer(roi=roi, mask=mask)

if __name__ == "__main__":
    main()