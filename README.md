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

- **Python 3.10 or later** ([download](https://www.python.org/downloads/))
- **Windows 10/11**, macOS, or Linux
- **8+ GB RAM** recommended
- **~2.5 GB disk space** for PyTorch and model weights
- **NVIDIA GPU** (optional, auto-detected for CUDA acceleration)

### Setup

```bash
# Clone or download the repository
cd DataPoison

# Run the setup script
python setup.py
```

The setup script automatically detects your system and installs optimal dependencies:

| Detected | How It's Used |
|----------|---------------|
| **Python version** | Checks 3.10+ compatibility, warns if outdated |
| **OS & architecture** | Handles platform-specific installs (Windows/macOS/Linux, x86/ARM) |
| **NVIDIA GPU & VRAM** | Selects correct CUDA build (12.8 for RTX 50xx Blackwell, etc.) |
| **GPU compute capability** | Maps SM version (sm_120, sm_89, etc.) to CUDA toolkit |
| **CUDA driver version** | Validates driver supports the selected CUDA toolkit |
| **System RAM** | Warns if below 8 GB recommended |
| **Existing packages** | Skips reinstalling up-to-date dependencies, upgrades old ones |

Example output:
```
[1/4] Detecting system specs...

  Python:     3.12.10 (AMD64) OK
  OS:         Windows 10
  CPU:        Intel Core i9-14900HX (32 cores)
  RAM:        31.7 GB OK
  GPU:        NVIDIA GeForce RTX 5070 Laptop GPU
  VRAM:       8.0 GB
  Compute:    12.0 (sm_120)
  Driver:     595.97 (CUDA 12.9)

[2/4] Checking existing dependencies...

  Pillow             12.2.0               OK
  numpy              2.4.6                OK
  torch              not installed         WILL INSTALL
  ...

  PyTorch target: CUDA 12.8
  Reason: Blackwell architecture (sm_120) requires CUDA 12.8+

[3/4] Installing dependencies...
[4/4] Verifying installation...
```

If no GPU is detected, CPU-only PyTorch is installed automatically.

> **First run:** PyTorch (~2.5 GB) and pre-trained model weights (~45 MB) are downloaded during setup.

### Manual Setup (Alternative)

If you prefer to install manually:

```bash
# Install base dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (choose one):
# For RTX 50xx (Blackwell) or RTX 40xx/30xx (Ada/Ampere):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU-only (no GPU):
pip install torch torchvision
```

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

### What DataPoison Protects Against (and What It Doesn't)

A common question: *"I sent my protected image to Gemini/GPT-4V and it could still describe it — does the protection work?"*

**Yes, but it's important to understand the difference between training and inference:**

| | DataPoison Targets | AI Assistants Do |
|---|---|---|
| **Stage** | **Training** — when AI *learns* from data | **Inference** — when AI *looks at* an image |
| **Goal** | Make the image **toxic to learn from** | **Understand** a single image on demand |
| **How it works** | Poisons gradients during backpropagation | Uses an already-trained model to describe what it sees |
| **Effect of perturbations** | Corrupts the model's *weights* over time | Barely affects a *single forward pass* |

**Why AI assistants can still "see" your image:**

Multimodal models like Gemini, GPT-4V, and Claude are *already fully trained*. When you send them an image, they run a single forward pass to describe it. The adversarial perturbations DataPoison adds are tiny (ε = 0.001–0.025 per pixel) — far too small to prevent a powerful vision model from recognizing image content during a one-off query. These models are robust to small noise at inference time.

**What DataPoison actually prevents:**

The protection activates when someone **scrapes your image and includes it in a training dataset** to teach an AI model. During training, the model processes thousands of images through backpropagation (gradient updates). The perturbations then:

- **FGSM/Nightshade:** Inject wrong gradients that teach the model incorrect associations
- **Style Cloak:** Make the model learn a wrong style representation, preventing style replication
- **HF Noise:** Corrupt the resized/normalized versions the model actually trains on
- **Compound effect:** Thousands of poisoned images in a dataset accumulate errors, degrading the model's output quality

**The analogy:** Think of it like a slow-acting contaminant in a water supply. Someone taking a single sip (AI inference) won't notice anything wrong. But if a factory (AI training pipeline) uses that water as an ingredient in millions of products (model weights), the contamination accumulates and ruins the output.

**Bottom line:** DataPoison protects against your work being *used as training data* without consent — the core ethical concern. It is not an "invisibility cloak" against already-trained AI assistants analyzing a single image. No current technique achieves that while keeping the image visually identical to human eyes.

### Protection Limitations

- **Not future-proof:** As with any adversarial technique, future AI architectures may develop countermeasures. DataPoison is most effective against current-generation models.
- **Lossy compression:** JPEG and aggressive WebP compression can partially remove perturbations. Always export as PNG for maximum protection.
- **Screenshots and photographs:** Adversarial perturbations are generally robust to re-capture (screenshots, photos of screens), but this is not guaranteed for all techniques.
- **Compound protection:** Enabling multiple techniques simultaneously provides defense-in-depth. The engine automatically scales per-technique epsilon by 1/√(n_enabled) to maintain invisibility.
- **Ethical use:** DataPoison is designed for artists protecting their own work. Do not use it to poison datasets you do not own.

---

## Glossary of Technical Terms

### Image Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
A measure of how similar two images are, expressed in decibels (dB). Higher values mean the images are more similar.

| PSNR Range | What It Means |
|------------|---------------|
| > 50 dB | Mathematically near-identical — impossible to distinguish |
| 45–50 dB | Visually identical — no human can tell the difference |
| 40–45 dB | Very subtle — faint grain may appear on close inspection of flat areas |
| 30–40 dB | Noticeable — visible artifacts or noise |
| < 30 dB | Obviously different |

DataPoison uses PSNR as a **quality gate**: after applying all poisoning techniques, it checks whether the protected image still meets the PSNR threshold for the selected preset. If not, it automatically reduces the perturbation strength and retries. This guarantees the output always looks identical to the original.

**SSIM (Structural Similarity Index)**
A perceptual metric that compares the *structure* of two images — luminance, contrast, and texture — rather than raw pixel values. Ranges from 0 (completely different) to 1 (identical).

| SSIM Range | What It Means |
|------------|---------------|
| 0.995–1.0 | Structurally identical to human perception |
| 0.990–0.995 | Imperceptible structural change |
| 0.970–0.990 | Very subtle structural shift |
| < 0.970 | Potentially noticeable |

DataPoison uses SSIM alongside PSNR as a dual quality gate. Both must pass for the output to be accepted. SSIM catches cases where PSNR might pass but the image structure has subtly shifted (e.g., from aggressive style cloaking).

### Perturbation Parameters

**Epsilon (ε)**
The maximum amount any single pixel can change, expressed as a fraction of the full 0–255 range. This is the primary control for the trade-off between protection strength and visual invisibility.

| Epsilon | Pixel Change (out of 255) | Visual Effect |
|---------|--------------------------|---------------|
| 0.001–0.005 | < 1 | Completely invisible |
| 0.005–0.010 | 1–3 | Invisible to human eyes |
| 0.010–0.020 | 3–5 | Very faint grain on close inspection |
| 0.020–0.030 | 5–8 | Subtle grain on flat-color areas |
| > 0.030 | > 8 | Visible noise texture |

When multiple techniques are enabled, DataPoison automatically divides each technique's epsilon by √(n_enabled) to keep the *total* perturbation within bounds.

**Lambda (λ)**
The regularization strength used in Style Cloak. Controls how aggressively the optimizer resists changing pixels from the original. Higher lambda (8–10) = less pixel change, more conservative. Lower lambda (2–3) = more aggressive feature-space movement, slightly more pixel change.

**Iterations**
The number of optimization steps the Style Cloak engine runs. More iterations allow the optimizer to find a better perturbation, but take longer. With strong regularization, even 100 iterations remain invisible.

### AI/ML Terms

**FGSM (Fast Gradient Sign Method)**
An adversarial attack technique from 2014 (Goodfellow et al.) that computes how a neural network's loss changes with respect to each input pixel, then adds noise in that direction. DataPoison uses FGSM *in reverse*: it finds what perturbation confuses the model most, then adds it invisibly to the image.

**Surrogate Model**
A pre-trained AI model (ResNet-18 or VGG-16) that DataPoison uses locally to *compute* perturbations. The perturbations calculated against this model **transfer** to other AI architectures (including diffusion models) due to a well-documented property called *adversarial transferability*. No images leave your machine.

**Feature Space**
The internal representation an AI model creates when it "looks" at an image. Two images that look different to humans might be close in feature space (e.g., two watercolor paintings), while two images that look similar to humans might be far apart in feature space. Style Cloak works by pushing your image far away in feature space while keeping it identical in pixel space.

**Adversarial Transferability**
The phenomenon where perturbations designed to fool one AI model also fool *other*, different AI models. This is why DataPoison only needs a small surrogate model (ResNet-18) to create perturbations that disrupt much larger models like Stable Diffusion, DALL-E, or Midjourney.

### Data Poisoning Terms

**LSB (Least Significant Bit)**
The lowest bit in each byte of pixel data. Changing only the LSB alters pixel values by at most 1 out of 255 — completely invisible to human vision. DataPoison embeds the text "DATAPOISON" in the LSBs as an invisible watermark that survives lossless formats.

**EXIF (Exchangeable Image File Format)**
Metadata embedded in image files: camera model, GPS coordinates, timestamps, software used, etc. DataPoison strips original EXIF data and replaces it with misleading fields ("Software: DataPoison v1.0", copyright warnings) to confuse data scraping pipelines.

**Moire Pattern**
An interference pattern created by overlapping regular grids. DataPoison generates moire patterns at specific frequencies (7, 9, 11 pixel periods) that are invisible at native resolution but create artifacts when AI models resize images to standard input sizes (224×224, 512×512).

**Quality Gate**
An automatic checkpoint that validates the protected image meets the invisibility requirements (PSNR and SSIM thresholds). If the gate fails, DataPoison halves all epsilon values and retries up to 2 times. This ensures you never accidentally produce a visibly altered image.

---

## License

This project is provided for educational and ethical protection purposes. Use responsibly to protect your own creative works from unauthorized AI training.
