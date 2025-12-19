# BIDR Relight
# 12/11/25
# CS7180 Advanced Perception
# Contributors: Max Huber, Adharsh Kandula, Richard Zhao

# This file contains the ISD (Illumination Spectral Direction) estimation module.
# Several components were coded/modified with the help of GPT-5 and Claude Sonnet 4.5


"""ISD (Illumination Spectral Direction) estimation module."""
import numpy as np
import torch
import imageio.v3 as iio
import logging

from src.image_util import resize_with_same_aspect, linear_to_log
from src.models.mock import MockISDModel
from src.models.unet import ResNet50UNet

logger = logging.getLogger(__name__)


def get_device():
    """
    Get the best available device for torch (MPS > CUDA > CPU).
    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def load_and_preprocess_image(img_input, resize_scale=1.0):
    """
    Load image (from path or numpy array), convert to log space, and normalize.
    Args:
        img_input (str or np.ndarray): File path or image array (H,W,3).
        resize_scale (float): Scale factor for resizing.
    Returns:
        img (np.ndarray): Original image (H, W, 3).
        bit_depth (int): Bit depth of original image.
        log_img (np.ndarray): Log RGB image.
        log_norm_img (np.ndarray): Log RGB normalized to [0,1].
    """
    # If a numpy array was passed directly, use it
    if isinstance(img_input, np.ndarray):
        img = img_input.copy()
        # Determine bit depth from dtype
        if np.issubdtype(img.dtype, np.integer):
            bit_depth = np.iinfo(img.dtype).bits
        else:
            # Float arrays - assume already in [0, 255] range for 8-bit
            bit_depth = 8
            if img.max() <= 1.0:  # Normalized floats
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
    else:
        # Load from filepath using imageio (preserves bit depth better than skimage)
        img = iio.imread(img_input)
        
        # Handle different dtypes
        if np.issubdtype(img.dtype, np.integer):
            bit_depth = np.iinfo(img.dtype).bits
        elif img.dtype == np.float32 or img.dtype == np.float64:
            # Some formats store as float - check range
            if img.max() <= 1.0:
                # Normalized floats, scale to 16-bit
                img = (img * 65535).astype(np.uint16)
                bit_depth = 16
            else:
                # Assume 8-bit range
                img = img.astype(np.uint8)
                bit_depth = 8
        else:
            logger.warning(f"Unexpected dtype {img.dtype}, defaulting to 8-bit")
            img = img.astype(np.uint8)
            bit_depth = 8
    
    logger.info(f"Loaded image: shape={img.shape}, dtype={img.dtype}, bit_depth={bit_depth}, range=[{img.min()}, {img.max()}]")
    
    img = resize_with_same_aspect(img, scale=resize_scale)
    img = img[:, :, :3]  # Drop alpha if present

    log_img = linear_to_log(img)
    log_norm_img = log_img / np.log(2**bit_depth - 1)
    log_norm_img = log_norm_img.astype(np.float32)

    return img, bit_depth, log_img, log_norm_img


def get_isd_model(model_type, model_path=None, device=None):
    """
    Initialize ISD estimation model (mock, unet, or vit).
    Args:
        model_type (str): "unet", "vit", or "mock".
        model_path (str): Path to model checkpoint (for unet/vit).
        device (torch.device or None): Device to use (auto-detect if None).
    Returns:
        model: Instantiated model.
        device: torch.device used.
    """
    if device is None:
        device = get_device()
    
    if model_type == "unet":
        model = ResNet50UNet(
            in_channels=3,
            out_channels=3,
            pretrained=False,
            se_block=True,
            dropout=0.0,
        )
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif model_type == "vit":
        raise NotImplementedError("ViT model not yet implemented")
    else:
        model = MockISDModel()
    
    model.eval()
    model = model.to(device)
    
    return model, device


def estimate_isd_map(log_norm_img, model, device):
    """
    Estimate ISD map for an image using the provided model.
    Args:
        log_norm_img (np.ndarray): Normalized log RGB image (H, W, 3).
        model: ISD estimation model.
        device: torch.device to run inference on.
    Returns:
        np.ndarray: Normalized ISD vectors (H, W, 3).
    """
    # Convert to tensor and move to device
    log_norm_img_tensor = (
        torch.from_numpy(log_norm_img).permute(2, 0, 1).unsqueeze(0)
    )
    log_norm_img_tensor = log_norm_img_tensor.to(device)
    
    # Run model
    with torch.no_grad():
        isd_map = model(log_norm_img_tensor)
    
    # Convert back to numpy
    isd_map = isd_map.cpu().detach().squeeze(0).numpy()  # (3, H, W)
    isd_map = np.transpose(isd_map, (1, 2, 0))  # (H, W, 3)
    
    # Normalize to unit vectors
    isd_norm = np.linalg.norm(isd_map, axis=2, keepdims=True)
    isd_norm[isd_norm == 0] = 1
    isd_map = isd_map / isd_norm
    
    return isd_map


def process_image_pair(content_path, style_path, model_type="mock", 
                       model_path=None, resize_scale=1/4, device=None):
    """
    Process content and style images through ISD estimation pipeline.
    Args:
        content_path (str or np.ndarray): Path to content image or array.
        style_path (str or np.ndarray): Path to style image or array.
        model_type (str): "unet", "vit", or "mock".
        model_path (str): Path to model checkpoint.
        resize_scale (float): Scale factor for resizing.
        device (torch.device or None): Device to use (auto-detect if None).
    Returns:
        dict: Results for content and style images, each with keys:
            'img', 'bit_depth', 'log_img', 'log_norm_img', 'isd_map'.
    """
    # Initialize ISD model and device
    model, device = get_isd_model(model_type, model_path, device)
    
    results = {}
    for name, path in [("content", content_path), ("style", style_path)]:
        # Load and preprocess image
        img, bit_depth, log_img, log_norm_img = load_and_preprocess_image(
            path, resize_scale
        )
        # Estimate ISD map for image
        isd_map = estimate_isd_map(log_norm_img, model, device)
        # Store results
        results[name] = {
            "img": img,
            "bit_depth": bit_depth,
            "log_img": log_img,
            "log_norm_img": log_norm_img,
            "isd_map": isd_map,
        }
    
    return results