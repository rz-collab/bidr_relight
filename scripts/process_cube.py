import tifffile
import numpy as np

# Load raw TIFF
raw = tifffile.imread("00_0634.tiff")

# Assuming RGGB Bayer pattern
R = raw[0::2, 0::2]
G1 = raw[0::2, 1::2]
G2 = raw[1::2, 0::2]
B = raw[1::2, 1::2]

# Average the two green pixels
G = (G1 + G2) // 2

# Stack into RGB
rgb = np.stack([R, G, B], axis=-1)

# Save as 16-bit PNG
tifffile.imwrite("00_0634_linear.tiff", rgb.astype(np.uint16))
