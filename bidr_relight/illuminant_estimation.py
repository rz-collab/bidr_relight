import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class RecursiveRetinex:
    """
    Fast CUDA-accelerated Recursive Retinex for illumination estimation.
    Based on the improved algorithm from Zhang et al. (2011) with
    8-direction synchronous interaction.
    """

    def __init__(self, num_iterations=5, device='cuda'):
        """
        Args:
            num_iterations: Number of recursive iterations (default: 5)
            device: 'cuda' or 'cpu'
        """
        self.num_iterations = num_iterations
        self.device = device

        # 8 directions: N, NE, E, SE, S, SW, W, NW
        # Represented as (dy, dx) offsets
        self.directions = torch.tensor([
            [-1, 0],  # N
            [-1, 1],  # NE
            [0, 1],  # E
            [1, 1],  # SE
            [1, 0],  # S
            [1, -1],  # SW
            [0, -1],  # W
            [-1, -1]  # NW
        ], device=device)

    def estimate_illumination(self, img_tensor, beta=1.0, alpha=1.0):
        """
        Estimate illumination using improved recursive Retinex.

        Args:
            img_tensor: (C, H, W) tensor in range [0, 1]
            beta: Scale parameter for distance weighting (default: 1.0)
            alpha: Boundary detection sensitivity (default: 1.0)

        Returns:
            illumination: (C, H, W) estimated illumination in range [0, 1]
        """
        # Move to device and add batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)

        img_tensor = img_tensor.to(self.device)

        # Work in log space
        epsilon = 1e-6
        s = torch.log(img_tensor + epsilon)  # (1, C, H, W)

        # Initialize illumination as input (Equation 4)
        l_n = s.clone()

        # Recursive iterations
        for n in range(self.num_iterations):
            # Distance for this iteration (powers of 2)
            d = 2 ** n

            # 8-direction synchronous interaction (Equation 5, 6)
            l_next = self._synchronous_interaction(
                l_n, s, d, beta, alpha
            )

            l_n = l_next

        # Convert back to linear space
        illumination = torch.exp(l_n)

        return illumination.squeeze(0)  # (C, H, W)

    def _synchronous_interaction(self, l_n, s, d, beta, alpha):
        """
        Perform 8-direction synchronous interaction (Equation 5, 6, 7).

        Args:
            l_n: Current illumination estimate (1, C, H, W)
            s: Log of original image (1, C, H, W)
            d: Distance for translation
            beta: Scale parameter
            alpha: Boundary sensitivity

        Returns:
            l_next: Updated illumination (1, C, H, W)
        """
        B, C, H, W = l_n.shape

        # Store translated versions
        translations = []

        # Apply translation in all 8 directions
        for direction in self.directions:
            dy, dx = direction

            # Translate by (dy*d, dx*d)
            # Use torch.roll for circular translation (simpler than padding)
            l_translated = torch.roll(l_n, shifts=(dy * d, dx * d), dims=(2, 3))
            translations.append(l_translated)

        # Stack all translations: (8, 1, C, H, W)
        translations = torch.stack(translations, dim=0)

        # Compute variance for boundary detection (Equation 7)
        l_variance = torch.var(translations, dim=0)  # (1, C, H, W)

        # Distance weighting factor (Equation 6)
        weight = (1 + beta * d) / (d + beta * d)

        # Boundary-aware weighting (Equation 7)
        # Higher variance → reduce influence
        boundary_factor = 1.0 / (1 + alpha * l_variance)

        # Weighted max and min across directions (Equations 8, 9)
        max_translation, _ = torch.max(translations, dim=0)  # (1, C, H, W)
        min_translation, _ = torch.min(translations, dim=0)

        # Combined update (Equation 10 simplified)
        # Average over all 8 directions with boundary awareness
        avg_translation = translations.mean(dim=0)

        # Update rule (adapted from Equation 5)
        term1 = (l_n + s) / 2
        term2 = (l_n + avg_translation) / 2

        # Apply distance weighting and boundary factor
        l_next = torch.max(term1, term2 * weight * boundary_factor)

        return l_next

    def estimate_reflectance(self, img_tensor, illumination):
        """
        Estimate reflectance given image and illumination.
        From S = L * R, we get R = S / L

        Args:
            img_tensor: (C, H, W) original image
            illumination: (C, H, W) estimated illumination

        Returns:
            reflectance: (C, H, W) estimated reflectance
        """
        epsilon = 1e-6
        reflectance = img_tensor / (illumination + epsilon)
        return torch.clamp(reflectance, 0, 1)

    def estimate_direct_illuminant(self, illumination, percentile=95):
        """
        Estimate direct illuminant color from illumination layer.

        The direct illuminant is approximated by the brightest pixels
        in the illumination map, as these represent areas directly lit
        by the primary light source.

        Args:
            illumination: (C, H, W) illumination estimate
            percentile: Percentile for bright pixel selection (default: 95)

        Returns:
            direct_illuminant: (3,) RGB color of direct illuminant
        """
        C, H, W = illumination.shape

        # Find bright pixels (above percentile threshold)
        # Use average across channels for brightness
        brightness = illumination.mean(dim=0)  # (H, W)
        threshold = torch.quantile(brightness.flatten(), percentile / 100.0)

        # Mask for bright pixels
        bright_mask = brightness > threshold

        # Extract direct illuminant as average of bright pixels
        direct_illuminant = torch.zeros(C, device=self.device)
        for c in range(C):
            bright_values = illumination[c][bright_mask]
            if len(bright_values) > 0:
                direct_illuminant[c] = bright_values.mean()
            else:
                direct_illuminant[c] = illumination[c].max()

        return direct_illuminant

    def estimate_ambient_illuminant(self, illumination, percentile=10):
        """
        Estimate ambient illuminant from dark regions.

        Args:
            illumination: (C, H, W) illumination estimate
            percentile: Percentile for dark pixel selection (default: 10)

        Returns:
            ambient_illuminant: (3,) RGB color of ambient illuminant
        """
        C, H, W = illumination.shape

        # Find dark pixels
        brightness = illumination.mean(dim=0)
        threshold = torch.quantile(brightness.flatten(), percentile / 100.0)
        dark_mask = brightness < threshold

        # Extract ambient as average of dark pixels
        ambient_illuminant = torch.zeros(C, device=self.device)
        for c in range(C):
            dark_values = illumination[c][dark_mask]
            if len(dark_values) > 0:
                ambient_illuminant[c] = dark_values.mean()
            else:
                ambient_illuminant[c] = illumination[c].min()

        return ambient_illuminant


class FastRetinexIlluminant:
    """
    Simplified fast version for just getting direct/ambient illuminant.
    Uses fewer iterations and optimized for speed.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.retinex = RecursiveRetinex(num_iterations=3, device=device)

    def estimate_illuminants(self, image):
        """
        One-shot estimation of direct and ambient illuminants.

        Args:
            image: (H, W, 3) numpy array or (3, H, W) tensor, range [0, 1]

        Returns:
            direct: (3,) RGB direct illuminant color
            ambient: (3,) RGB ambient illuminant color
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:  # (H, W, 3)
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image)

        image = image.float().to(self.device)

        # Estimate illumination
        illum = self.retinex.estimate_illumination(image)

        # Extract direct and ambient
        direct = self.retinex.estimate_direct_illuminant(illum, percentile=95)
        ambient = self.retinex.estimate_ambient_illuminant(illum, percentile=10)

        return direct.cpu().numpy(), ambient.cpu().numpy()

def gray_world_illuminant(img):
    """Simple Gray World assumption."""
    return img.mean(axis=(0, 1))

def max_rgb_illuminant(img):
    """Max RGB assumption."""
    return img.max(axis=(0, 1))

def shades_of_gray_illuminant(img, p=6):
    """Shades of Gray (Minkowski norm)."""
    return (img ** p).mean(axis=(0, 1)) ** (1/p)

# ============= USAGE EXAMPLES =============

def demo_basic_usage():
    """Basic usage example."""
    # Load image
    img = Image.open("test_image.jpg").convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0

    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    # Initialize Retinex
    retinex = RecursiveRetinex(num_iterations=5, device='cuda')

    # Estimate illumination
    with torch.no_grad():
        illumination = retinex.estimate_illumination(img_tensor)
        reflectance = retinex.estimate_reflectance(img_tensor, illumination)

    # Estimate illuminants
    direct = retinex.estimate_direct_illuminant(illumination)
    ambient = retinex.estimate_ambient_illuminant(illumination)

    print(f"Direct illuminant (RGB): {direct.cpu().numpy()}")
    print(f"Ambient illuminant (RGB): {ambient.cpu().numpy()}")

    # Save results
    illumination_np = illumination.permute(1, 2, 0).cpu().numpy()
    reflectance_np = reflectance.permute(1, 2, 0).cpu().numpy()

    Image.fromarray((illumination_np * 255).astype(np.uint8)).save("illumination.png")
    Image.fromarray((reflectance_np * 255).astype(np.uint8)).save("reflectance.png")


def demo_fast_estimation():
    """Fast estimation for batch processing."""
    estimator = FastRetinexIlluminant(device='cuda')

    # Load image
    img = np.array(Image.open("test.jpg").convert('RGB')).astype(np.float32) / 255.0

    # Quick estimation
    direct, ambient = estimator.estimate_illuminants(img)

    print(f"Direct illuminant: R={direct[0]:.3f}, G={direct[1]:.3f}, B={direct[2]:.3f}")
    print(f"Ambient illuminant: R={ambient[0]:.3f}, G={ambient[1]:.3f}, B={ambient[2]:.3f}")

    # Compute spectral ratio (for BIDR)
    spectral_ratio = ambient / direct
    print(f"Spectral ratio S: {spectral_ratio}")


def demo_batch_processing():
    """Process multiple images efficiently."""
    retinex = RecursiveRetinex(num_iterations=4, device='cuda')

    # Load multiple images
    images = []
    for i in range(10):
        img = np.array(Image.open(f"image_{i}.jpg").convert('RGB')).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        images.append(img_tensor)

    # Stack into batch (B, C, H, W)
    batch = torch.stack(images, dim=0).to('cuda')

    # Process entire batch at once (very fast!)
    with torch.no_grad():
        # Note: Current implementation processes channels, not batch
        # For true batch processing, reshape: (B*C, 1, H, W)
        B, C, H, W = batch.shape
        batch_flat = batch.view(B * C, 1, H, W)

        # Estimate illumination for all
        illum_flat = retinex.estimate_illumination(batch_flat)
        illum = illum_flat.view(B, C, H, W)

    # Extract illuminants for each image
    for i in range(B):
        direct = retinex.estimate_direct_illuminant(illum[i])
        print(f"Image {i}: Direct = {direct.cpu().numpy()}")


def compute_spectral_ratio_for_bidr(img_path, device='cuda'):
    """
    Complete pipeline: Image → Illuminants → Spectral Ratio for BIDR.

    Returns spectral ratio S = M_ambient / l_direct that you can use
    to compute ISD = normalize(log(1/S + 1))
    """
    # Load image
    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(device)

    # Estimate illumination
    retinex = RecursiveRetinex(num_iterations=5, device=device)

    with torch.no_grad():
        illumination = retinex.estimate_illumination(img_tensor)

        # Get illuminants
        l_direct = retinex.estimate_direct_illuminant(illumination, percentile=95)
        M_ambient = retinex.estimate_ambient_illuminant(illumination, percentile=5)

    # Compute spectral ratio
    S = M_ambient / (l_direct + 1e-6)

    # Compute ISD from spectral ratio (Equation 17 from Maxwell)
    isd_unnormalized = torch.log(1.0 / S + 1.0)
    isd = isd_unnormalized / torch.norm(isd_unnormalized)

    print(f"Direct illuminant: {l_direct.cpu().numpy()}")
    print(f"Ambient illuminant: {M_ambient.cpu().numpy()}")
    print(f"Spectral ratio S: {S.cpu().numpy()}")
    print(f"ISD: {isd.cpu().numpy()}")

    return {
        'l_direct': l_direct.cpu().numpy(),
        'M_ambient': M_ambient.cpu().numpy(),
        'S': S.cpu().numpy(),
        'ISD': isd.cpu().numpy(),
        'illumination': illumination.cpu().permute(1, 2, 0).numpy()
    }


# ============= OPTIMIZED ALTERNATIVE =============
# If the above is still too slow, here's a simpler approximation

class SimpleIlluminantEstimator:
    """
    Much faster approximation using max pooling and blurring.
    Not strictly Retinex but achieves similar illuminant estimation.
    """

    def __init__(self, device='cuda'):
        self.device = device

    def estimate_illuminants_fast(self, img_tensor, kernel_size=15):
        """
        Fast approximation using morphological operations.

        Args:
            img_tensor: (C, H, W) or (H, W, 3) image in [0, 1]
            kernel_size: Size for max pooling (larger = smoother)

        Returns:
            direct: (3,) RGB direct illuminant
            ambient: (3,) RGB ambient illuminant
        """
        # Convert to (1, C, H, W)
        if isinstance(img_tensor, np.ndarray):
            if img_tensor.shape[-1] == 3:
                img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)
            else:
                img_tensor = torch.from_numpy(img_tensor)

        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.float().to(self.device)

        # Estimate illumination via max pooling (approximates bright regions)
        # This is MUCH faster than recursive Retinex
        padding = kernel_size // 2
        illum_max = F.max_pool2d(img_tensor, kernel_size, stride=1, padding=padding)

        # Smooth with Gaussian blur
        illum_smooth = self._gaussian_blur(illum_max, kernel_size)

        # Direct illuminant: brightest regions
        direct = torch.quantile(illum_smooth.flatten(start_dim=1), 0.98, dim=1)  # (1, C)

        # Ambient: darkest regions of smoothed illumination
        ambient = torch.quantile(illum_smooth.flatten(start_dim=1), 0.02, dim=1)

        return direct.squeeze().cpu().numpy(), ambient.squeeze().cpu().numpy()

    def _gaussian_blur(self, x, kernel_size):
        """Apply Gaussian blur on GPU."""
        # Create 1D Gaussian kernel
        sigma = kernel_size / 6.0
        size = kernel_size

        x_coord = torch.arange(size, device=self.device, dtype=torch.float32)
        x_coord -= size // 2
        gauss = torch.exp(-(x_coord ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # 1D convolution in x direction
        kernel_x = gauss.view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
        x = F.conv2d(x, kernel_x, padding=(0, kernel_size // 2), groups=x.shape[1])

        # 1D convolution in y direction
        kernel_y = gauss.view(1, 1, -1, 1).repeat(x.shape[1], 1, 1, 1)
        x = F.conv2d(x, kernel_y, padding=(kernel_size // 2, 0), groups=x.shape[1])

        return x


# ============= COMPARISON & BENCHMARKING =============

def compare_methods(img_path):
    """Compare Recursive Retinex vs Fast approximation."""
    import time

    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0

    print(f"Image size: {img.shape}")

    # Method 1: Recursive Retinex (accurate but slower)
    print("\n=== Recursive Retinex ===")
    retinex = RecursiveRetinex(num_iterations=5, device='cuda')
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        illum = retinex.estimate_illumination(img_tensor)
        direct_rr = retinex.estimate_direct_illuminant(illum)
        ambient_rr = retinex.estimate_ambient_illuminant(illum)

    torch.cuda.synchronize()
    elapsed_rr = time.time() - start

    print(f"Time: {elapsed_rr:.3f}s")
    print(f"Direct: {direct_rr.cpu().numpy()}")
    print(f"Ambient: {ambient_rr.cpu().numpy()}")

    # Method 2: Fast approximation
    print("\n=== Fast Approximation ===")
    fast = SimpleIlluminantEstimator(device='cuda')

    torch.cuda.synchronize()
    start = time.time()

    direct_fast, ambient_fast = fast.estimate_illuminants_fast(img)

    torch.cuda.synchronize()
    elapsed_fast = time.time() - start

    print(f"Time: {elapsed_fast:.3f}s")
    print(f"Direct: {direct_fast}")
    print(f"Ambient: {ambient_fast}")
    print(f"\nSpeedup: {elapsed_rr / elapsed_fast:.1f}x")