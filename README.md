# DataPoison

**Protect your artwork from unauthorized AI training.**

DataPoison is a free, open-source desktop application that helps artists apply invisible data poisoning techniques to their images and artwork. When AI models train on these protected images without consent, the embedded perturbations disrupt the learning process — while the images remain visually identical to the originals for human viewers.

---

## Why DataPoison?

Generative AI models are trained on billions of images scraped from the internet, often without the creators' knowledge or consent. Opt-out lists are unverifiable and easily ignored. DataPoison gives artists a technical tool to protect their work by making it costly for unauthorized scrapers to use their images in training pipelines.

**Core principle:** All protection techniques are designed to be **invisible to human eyes**. The application enforces strict quality gates (PSNR ≥ 40–48 dB depending on preset) and will automatically reduce protection strength if any technique produces visible artifacts.

---

## Features

- **5 data poisoning techniques** working independently or in combination
- **4 protection presets** from "barely there" to "maximum disruption"
- **Granular customization** — every parameter adjustable with real-time tooltips
- **Batch processing** — protect entire portfolios at once
- **Before/after preview** with amplified difference view to verify perturbation
- **Quality gate** — automatic PSNR/SSIM validation with fallback mechanisms
- **100% offline** — no images are ever sent to a server
- **Dark/light theme** following your system settings

---

## Installation

### Requirements

- **Python 3.11 or later** ([download](https://www.python.org/downloads/))
- **Windows 10/11** (primary target; macOS/Linux may work but are untested)
- **~2.5 GB disk space** for PyTorch and model weights
- GPU acceleration is optional (CUDA support auto-detected)

### Setup

```bash
# Clone or download the repository
cd DataPoison

# (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> **First run note:** PyTorch (~2 GB) and torchvision are downloaded during `pip install`. Pre-trained model weights (~45 MB for ResNet-18) are downloaded automatically on first use.

### Launch

```bash
python main.py
```

---

## Usage Guide

### 1. Add Images

Open the **Batch** tab and click **Add Files**. Select one or more images in any supported format:

| Format | Extensions |
|--------|-----------|
| PNG | `.png` |
| JPEG | `.jpg`, `.jpeg` |
| WebP | `.webp` |
| BMP | `.bmp` |
| TIFF | `.tiff`, `.tif` |

### 2. Choose a Preset

In the left sidebar, select one of the four presets:

| Preset | Protection Level | Visual Impact | Best For |
|--------|-----------------|---------------|----------|
| **Minimal Visual Impact** | Light | Zero visible change (PSNR ≥ 48 dB) | Sharing on social media, portfolios |
| **Moderate Protection** | Balanced | Zero visible change (PSNR ≥ 45 dB) | Most use cases (recommended default) |
| **Strong Protection** | Maximum | Very subtle grain at most (PSNR ≥ 40 dB) | High-value artwork, print work |
| **Custom** | User-defined | Varies | Advanced users who want full control |

Selecting a preset automatically configures all technique parameters.

### 3. Enable/Disable Techniques

In the **Protection Techniques** panel, toggle individual techniques on or off. Each card shows a description of what the technique protects against.

### 4. Adjust Parameters (Optional)

Switch to the **Custom** preset to unlock individual parameter sliders. Hover over any control to see a detailed tooltip explaining what it does, how it affects protection, and its impact on visual quality.

The **Protection vs Invisibility** bar at the bottom of the settings panel shows the real-time trade-off between protection strength and visual imperceptibility.

### 5. Process

- Click **Poison Selected Image** to process one image and preview the result
- Click **Poison All Images** to batch-process your entire queue

The progress bar updates in real time. Processing typically takes 1–10 seconds per image depending on resolution and enabled techniques.

### 6. Preview Results

Switch to the **Preview** tab to see a side-by-side comparison of the original and protected image. Use the **Show Amplified Difference** button to visualize the perturbation at 50× magnification — this confirms the protection is present even though it's invisible at normal viewing.

Quality metrics (PSNR and SSIM) are displayed in the top-right corner.

### 7. Save

Click **Save Protected Images...** and choose an output folder. Images are saved with a `_protected` suffix. **PNG is strongly recommended** for lossless output.

> **Warning:** JPEG compression can partially destroy adversarial perturbations. If you must export as JPEG, use the "Strong Protection" preset to compensate.

---

## Techniques Explained

### 1. Adversarial Noise (FGSM)

**Inspiration:** Fast Gradient Sign Method (Goodfellow et al., 2014)

Adds imperceptible, gradient-guided noise to each pixel that causes AI classifiers to misclassify the image. The perturbation is computed using the gradient of a pre-trained neural network's loss function with respect to the input pixels:

```
perturbed = original + ε × sign(∇loss(original))
```

- **Effective against:** CNN classifiers, image encoders, feature extractors
- **Invisible because:** ε values are constrained to 0.001–0.03 (max ~8/255 pixel change)

### 2. Style Cloak

**Inspiration:** Glaze (Shan et al., UChicago SAND Lab, 2023)

Shifts the image's representation in the AI's internal feature space toward a different perceived style, while keeping pixel-level changes below human perception. Uses an optimization loop with a pre-trained model's intermediate layer:

```
minimize: -feature_distance(perturbed, target_style) + λ × pixel_distance(perturbed, original)
```

- **Effective against:** Style mimicry, fine-tuning, LoRA training
- **Invisible because:** Per-iteration hard clamp to original ± ε, strong regularization (λ), and early stop if PSNR drops below threshold

### 3. Prompt Poison

**Inspiration:** Nightshade (Shan et al., 2024)

Creates perturbations that cause text-to-image diffusion models to learn incorrect associations. Uses a targeted variant of FGSM that pulls the image toward a semantically distant wrong class:

```
perturbed = original - ε × sign(∇loss_target_class(original))
```

Multiple semantically distant targets are applied in sequence for compounding effect.

- **Effective against:** Text-to-image models (Stable Diffusion, DALL-E, Midjourney)
- **Invisible because:** Same low-ε bounds as FGSM, epsilon auto-scaled by √(n_targets)

### 4. High-Frequency Noise

Structured noise at spatial frequencies that are beyond human visual acuity but disruptive to the resizing and normalization operations that AI pipelines apply to training images.

- **High-frequency Gaussian:** Random noise filtered via FFT to only contain high spatial frequencies
- **Moire patterns:** Sine wave grids at periods (7, 9, 11 pixels) that alias under common AI input sizes (224×224, 512×512)

- **Effective against:** Data preprocessing pipelines, image augmentation
- **Invisible because:** Amplitude ≤ 0.015 (~4/255), frequencies above human contrast sensitivity

### 5. Metadata Scramble

Non-visual protection that operates on image metadata and pixel bit planes:

- **EXIF stripping:** Removes all original metadata (GPS, camera info, timestamps)
- **EXIF injection:** Writes misleading fields ("Software: DataPoison v1.0", copyright warnings)
- **LSB watermark:** Embeds "DATAPOISON" in the least significant bit of pixel values (max change: 1/255 per channel)

- **Effective against:** Scraping pipelines, metadata-based filtering, dataset curation tools
- **Invisible because:** LSB modification changes pixel values by at most 1 out of 255

---

## Preset Details

| Parameter | Minimal | Moderate | Strong |
|-----------|---------|----------|--------|
| FGSM ε | 0.003 | 0.010 | 0.025 |
| Style Cloak ε | 0.002 | 0.008 | 0.020 |
| Style Cloak λ | 8.0 | 5.0 | 3.0 |
| Iterations | 20 | 40 | 80 |
| Nightshade ε | 0.003 | 0.010 | 0.025 |
| Nightshade targets | 1 | 2 | 4 |
| Noise amplitude | 0.002 | 0.008 | 0.015 |
| Watermark | Off | LSB (1 bit) | LSB+1 (2 bit) |
| Multi-pass | No | No | Yes |
| PSNR gate | 48 dB | 45 dB | 40 dB |
| SSIM gate | 0.995 | 0.990 | 0.970 |

**Quality gate:** After all techniques are applied, the engine computes PSNR and SSIM between the original and protected image. If either metric falls below the threshold, all epsilon values are halved and the pipeline retries (up to 2 times). This ensures the output is always visually imperceptible.

---

## Project Structure

```
DataPoison/
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── gui/
│   ├── app.py                   # Main window and orchestration
│   ├── image_viewer.py          # Before/after comparison with zoom
│   ├── technique_panel.py       # Technique selection cards
│   ├── settings_panel.py        # Preset selector and parameter controls
│   ├── batch_panel.py           # File queue and progress bar
│   ├── progress_dialog.py       # Modal processing dialog
│   └── tooltip.py               # Hover tooltip widget
├── core/
│   ├── presets.py               # Preset definitions and parameter metadata
│   ├── poison_engine.py         # Orchestrator and PoisonSettings
│   ├── fgsm_poison.py           # FGSM adversarial perturbation
│   ├── style_cloak.py           # Feature-space style cloaking
│   ├── nightshade_poison.py     # Targeted prompt poisoning
│   ├── noise_patterns.py        # FFT high-frequency noise
│   ├── quality_gate.py          # PSNR/SSIM validation
│   ├── metadata_poison.py       # EXIF and LSB watermarking
│   └── utils.py                 # Image conversion utilities
├── models/
│   └── model_loader.py          # Pre-trained model management
└── assets/
    └── styles/                  # Style reference images (optional)
```

---

## Technical Notes

### Performance

| Technique | Time per 1080p image (CPU) | Time (GPU) |
|-----------|--------------------------|------------|
| FGSM | ~0.2s | ~0.05s |
| Style Cloak (40 iter) | ~2–5s | ~0.5–1s |
| Prompt Poison (2 targets) | ~0.4s | ~0.1s |
| High-Frequency Noise | ~0.3s | ~0.3s (CPU-bound) |
| Metadata Scramble | <0.01s | <0.01s |

Images larger than 4096px on the longest edge are automatically downscaled to prevent out-of-memory errors.

### Protection Limitations

- **Not future-proof:** As with any adversarial technique, future AI architectures may develop countermeasures. DataPoison is most effective against current-generation models.
- **Lossy compression:** JPEG and aggressive WebP compression can partially remove perturbations. Always export as PNG for maximum protection.
- **Screenshots and photographs:** Adversarial perturbations are generally robust to re-capture (screenshots, photos of screens), but this is not guaranteed for all techniques.
- **Compound protection:** Enabling multiple techniques simultaneously provides defense-in-depth. The engine automatically scales per-technique epsilon by 1/√(n_enabled) to maintain invisibility.
- **Ethical use:** DataPoison is designed for artists protecting their own work. Do not use it to poison datasets you do not own.

---

## License

This project is provided for educational and ethical protection purposes. Use responsibly to protect your own creative works from unauthorized AI training.
