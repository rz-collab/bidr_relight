# BIDR Relight

by Richard Zhao, Adharsh Khandula, and Max Huber
(Developed on MacOS Sonoma)

## Gradio Demo hosted on Huggingface

Use any two images on the demo hosted on [Huggingface](https://huggingface.co/spaces/maxhuber/bidr-relight).

This pipeline transfers lighting from a **style image** to a **content image** using physics-based Illumination Spectral Direction (ISD). It works best with **16-bit linear images**, but also supports common 8-bit formats.

## 📸 Image Requirements

**Best results (recommended):**

- **16-bit linear TIFF / PNG / DNG**
- Full dynamic range preserved (no clipping, no gamma correction)

**Supported:**

- **8-bit JPEG/PNG**
  - Automatically converted to 16-bit linear
  - Limited shadow detail due to 8-bit quantization

**RAW formats (.CR2, .NEF, .ARW):**

- Supported when using the `rawpy` library.

---

## Pipeline Steps

### 1. Load & Estimate ISD

- Upload content and style images.
- Console shows detected bit depth and any conversions.
- ISD maps visualize the estimated illumination direction.

### 2. Cluster Materials

Segments the image into material regions. Available options:

- **Greedy:** clusters by log-chromaticity radius
- **K-means:** fixed number of clusters
- **Posterize:** optional quantization to stabilize noisy images

### 3. Estimate Illumination

- Computes the global illumination vector.
- Determines **dark (shadow)** and **bright (lit)** endpoints for each material cluster.
- Defines the lighting range used during transfer.

### 4. Apply Relighting

Transfers lighting from the style image to the content image using:

- **Length Scale** — adjusts light intensity (e.g., `0.5` = darker, `1.5` = brighter)
- **Rotation** — blends toward style illumination (0–100%)
- **Translation** — shifts colors in log-RGB space

---

## Tips

- Always use **16-bit linear** images for high-quality results.
- Check the console output for bit-depth and conversion diagnostics.
- You can rerun any step; modifying earlier steps requires rerunning later ones.
- Experiment with both 8-bit and 16-bit images to compare dynamic range.

# Local Usage

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Instructions:

All training and test data is done using sourced RAW images. Please try the Gradio demo with your own images.

## Running BIDR_Relight Demo

1. Clone the Repository to your workstation of choice.
2. Download any RAW content and style images you desire.
3. Run the Gradio app using `python app.py`.

## Relight images using a script

1. Clone the Repository to your workstation of choice.
2. Download any RAW content and style images you desire.
3. To run the script to relight a content and style image, use the provided example for reference:

```bash
python src/demo_batch_relight.py \
  --content data/content_folder/image1.tiff \
  --style data/style_folder/image1.tiff \
  --output results/relit_outputs \
  --isd_model unet \
  --isd_model_path weights/unet.pth
```
