# DataPoison - Image Protection Against AI Training

## Context

Artists need tools to protect their digital artwork from being scraped and used to train AI models without consent. Tools like Glaze (style cloaking), Nightshade (prompt poisoning), and adversarial perturbation (FGSM) have proven effective at disrupting AI training pipelines. This application bundles these techniques into an intuitive desktop GUI so artists can apply protections before sharing work online.

**Critical constraint: Poisoned images must be visually identical to the originals — zero perceptible changes to human eyes.** All perturbation methods are tuned for imperceptibility first, with aggressive pixel-change bounds, high regularization weights, and built-in PSNR quality gates.

## Techniques Implemented

| # | Technique | Inspiration | What It Does |
|---|-----------|-------------|--------------|
| 1 | **Adversarial Noise (FGSM)** | Fast Gradient Sign Method | Gradient-based pixel noise that confuses AI classifiers and diffusion models |
| 2 | **Style Cloak** | Glaze (UChicago SAND Lab) | Shifts image representation in AI feature space while looking identical to humans |
| 3 | **Prompt Poison** | Nightshade | Causes text-to-image models to learn wrong associations (targeted misclassification) |
| 4 | **High-Frequency Noise** | UAP / adversarial patterns | Ultra-low-amplitude structured noise at frequencies beyond human perception but disruptive to AI preprocessing |
| 5 | **Metadata Scramble** | Anti-scraping countermeasures | Replaces EXIF data, embeds LSB invisible watermarks ("DATAPOISON") |

## Tech Stack

- **Python 3.11+** on Windows
- **customtkinter** - modern dark/light themed GUI (no Qt licensing issues)
- **PyTorch + torchvision** - pre-trained ResNet-18/VGG-16 for gradient computation
- **Pillow** - image I/O
- **numpy** - array math
- **piexif** - EXIF metadata manipulation

## Project Structure

```
DataPoison/
├── main.py                      # Entry point
├── requirements.txt
├── gui/
│   ├── __init__.py
│   ├── app.py                   # Main window, state, threading orchestration
│   ├── image_viewer.py          # Side-by-side original/poisoned comparison with zoom
│   ├── technique_panel.py       # Technique selection cards with descriptions
│   ├── settings_panel.py        # Preset selector, individual parameter controls, tooltips, trade-off bar
│   ├── batch_panel.py           # Multi-file list with progress bar
│   ├── progress_dialog.py       # Modal progress dialog
│   └── tooltip.py               # Reusable tooltip widget (hover tooltips for all controls)
├── core/
│   ├── __init__.py
│   ├── presets.py               # Preset definitions (Minimal/Moderate/Strong/Custom) + parameter metadata
│   ├── poison_engine.py         # Orchestrator: expanded PoisonSettings, dispatches to engines
│   ├── fgsm_poison.py           # FGSM: epsilon * sign(gradient) perturbation
│   ├── style_cloak.py           # Feature-space optimization toward target style
│   ├── nightshade_poison.py     # Targeted misclassification (reversed FGSM)
│   ├── noise_patterns.py        # FFT high-freq noise + moire (no salt-and-pepper)
│   ├── quality_gate.py          # Preset-aware PSNR/SSIM check with auto-fallback
│   ├── metadata_poison.py       # EXIF rewrite + LSB watermark embedding
│   └── utils.py                 # PIL<->Tensor conversion, validation, save helpers
├── models/
│   ├── __init__.py
│   └── model_loader.py          # Lazy-load/cache torchvision models, device detection
└── assets/
    └── styles/                  # Small reference images for style cloaking targets
```

## Implementation Plan

### Task 1: Project Scaffold + Dependencies
- Create `requirements.txt`: customtkinter, Pillow, numpy, torch, torchvision, piexif, tqdm
- Create `main.py` entry point (customtkinter app bootstrap)
- Create all `__init__.py` files
- Create directory structure

### Task 2: Core Utilities + Model Loader
- `core/utils.py`: `pil_to_tensor()`, `tensor_to_pil()`, `validate_image()`, `save_image()`
- `models/model_loader.py`: lazy-load ResNet-18/VGG-16, freeze params, auto-detect CUDA/CPU

### Task 3: Preset Configuration System
- `core/presets.py` — defines all presets and parameter metadata:

**Preset definitions:**

| Parameter | Minimal Visual Impact | Moderate Protection | Strong Protection | Custom |
|-----------|----------------------|---------------------|-------------------|--------|
| `fgsm_epsilon` | 0.003 | 0.01 | 0.025 | user |
| `style_cloak_epsilon` | 0.002 | 0.008 | 0.02 | user |
| `style_cloak_lambda` | 8.0 | 5.0 | 3.0 | user |
| `style_cloak_iterations` | 20 | 40 | 80 | user |
| `nightshade_epsilon` | 0.003 | 0.01 | 0.025 | user |
| `nightshade_targets` | 1 | 2 | 4 | user |
| `noise_amplitude` | 0.002 | 0.008 | 0.015 | user |
| `noise_patterns_enabled` | 1 (HF only) | 2 (HF+moire) | 2 (HF+moire) | user |
| `metadata_strip_exif` | true | true | true | user |
| `metadata_watermark` | false | true | true | user |
| `metadata_watermark_depth` | 1 (LSB only) | 1 | 2 (LSB+1) | user |
| `quality_gate_psnr` | 48 dB | 45 dB | 40 dB | user |
| `multi_pass` | false | false | true | user |

**Parameter metadata dict** (for each parameter):
```python
PARAM_INFO = {
    "fgsm_epsilon": {
        "label": "FGSM Perturbation Strength",
        "description": "Controls the magnitude of gradient-based adversarial noise added to each pixel.",
        "protection_effect": "Higher values create stronger perturbations that more effectively confuse AI classifiers, but increase risk of visible artifacts.",
        "imperceptibility_effect": "Values below 0.008 are virtually invisible. 0.01-0.02 may produce faint grain on close inspection. Above 0.025, noise becomes visible on flat-color areas.",
        "recommended": "Minimal: 0.001-0.005 | Moderate: 0.008-0.015 | Strong: 0.02-0.03",
        "range": (0.001, 0.05),
        "step": 0.001,
        "default": 0.01,
    },
    # ... similar entries for all parameters
}
```

### Task 4: FGSM Poisoning Engine
- `core/fgsm_poison.py`: ImageNet normalize -> forward pass -> loss.backward() -> epsilon * sign(grad) -> clamp
- **Invisible constraint:** Epsilon value comes from preset; Minimal=0.003 (per-pixel < 1/255), Moderate=0.01, Strong=0.025
- Key: gradient is w.r.t. input pixels, not model weights

### Task 5: Style Cloaking Engine
- `core/style_cloak.py`: Hook intermediate layer (ResNet layer3) -> optimize perturbed image to minimize feature distance to target style while keeping pixel MSE low -> Adam optimizer
- **Invisible constraint:** Strong pixel regularization (lambda from preset), per-iteration clamp to original ± epsilon bound. Max pixel deviation hard-capped. Early stop if PSNR drops below preset threshold
- Iterations from preset: Minimal=20, Moderate=40, Strong=80

### Task 6: Nightshade-Inspired Prompt Poisoning
- `core/nightshade_poison.py`: Classify image -> pick semantically distant target class -> targeted FGSM (minimize loss for wrong class) -> optional multi-pass with different targets
- Number of target classes from preset: Minimal=1, Moderate=2, Strong=4

### Task 7: Noise Patterns + Metadata Poisoning
- `core/noise_patterns.py`: FFT-based high-frequency Gaussian + moire sine grids at aliasing frequencies. Amplitude from preset (Minimal=0.002, Moderate=0.008, Strong=0.015). **No salt-and-pepper** — it produces visible outlier pixels
- `core/metadata_poison.py`: Strip/rewrite EXIF via piexif, embed "DATAPOISON" in LSB of pixel array. Watermark depth from preset (1=LSB only, 2=LSB+1 bit). Higher depth = more robust but slightly less invisible

### Task 8: Poison Engine Orchestrator + Quality Gate
- `core/poison_engine.py`:
  - **Expanded `PoisonSettings` dataclass** with all individual parameters (not just a single epsilon):
    ```python
    @dataclass
    class PoisonSettings:
        preset: str = "moderate"
        fgsm_enabled: bool = True
        fgsm_epsilon: float = 0.01
        style_cloak_enabled: bool = True
        style_cloak_epsilon: float = 0.008
        style_cloak_lambda: float = 5.0
        style_cloak_iterations: int = 40
        style_cloak_target: str = "random"
        nightshade_enabled: bool = True
        nightshade_epsilon: float = 0.01
        nightshade_targets: int = 2
        noise_enabled: bool = True
        noise_amplitude: float = 0.008
        noise_patterns_enabled: list = field(default_factory=lambda: ["hf_gaussian", "moire"])
        metadata_enabled: bool = True
        metadata_strip_exif: bool = True
        metadata_watermark: bool = True
        metadata_watermark_depth: int = 1
        output_format: str = "png"
        model_name: str = "resnet18"
        multi_pass: bool = False
    ```
  - `apply_preset(preset_name)` class method that populates all fields from `core/presets.py`
  - Sequential technique application; when multiple enabled, each technique's epsilon is divided by sqrt(n_enabled)
- `core/quality_gate.py`:
  - **Preset-aware PSNR thresholds:** Minimal=48 dB, Moderate=45 dB, Strong=40 dB
  - Auto-fallback: if PSNR below threshold, reduce all epsilons by 50% and re-run (up to 2 retries)
  - For "Minimal Visual Impact" preset: additional SSIM check (must be > 0.995)
  - Warn user if protection was weakened; report final PSNR and max pixel delta

### Task 9: GUI - Main App Shell + Image Viewer
- `gui/app.py`: 1200x800 window, left sidebar (techniques + settings + buttons), right content area with tabs (Preview / Batch / Report)
- `gui/image_viewer.py`: Side-by-side CTkImage display with zoom slider, amplified difference view toggle (50x gain)

### Task 10: GUI - Preset Selector + Settings Panel + Tooltips
- `gui/settings_panel.py` — complete redesign with two sections:

**Section A: Preset Selector (top)**
- `CTkOptionMenu` or segmented button group with 4 presets: "Minimal Visual Impact" | "Moderate Protection" | "Strong Protection" | "Custom"
- Selecting a preset auto-fills all parameter controls below
- Selecting "Custom" unlocks all sliders for free editing
- Changing any individual slider while on a named preset auto-switches to "Custom"

**Section B: Per-Technique Parameter Controls (scrollable)**
Each enabled technique gets an expandable `CTkFrame` card containing:
- **FGSM:** epsilon slider (0.001–0.05, step 0.001) with label showing current value
- **Style Cloak:** epsilon slider, lambda (regularization) slider (1.0–10.0), iterations slider (10–100), target style dropdown
- **Nightshade:** epsilon slider, number of targets spinner (1–5), multi-pass toggle
- **Noise Patterns:** amplitude slider (0.001–0.03), sub-technique checkboxes (HF Gaussian, Moire)
- **Metadata:** strip EXIF toggle, watermark toggle, watermark depth dropdown (1–2 bits)

**Section C: Trade-off Indicator Bar (bottom of settings)**
- Horizontal bar visualization: left = "Maximum Invisibility", right = "Maximum Protection"
- Marker position computed from current parameter combination (weighted average of normalized epsilon values)
- Color gradient: green (invisible) -> yellow (caution) -> red (visible artifacts likely)
- Updates in real-time as sliders change

**Section D: Common Settings**
- Output format dropdown (PNG / JPEG 95% / WebP)
- Model dropdown (ResNet-18 fast / VGG-16 accurate)

**Tooltips (via `gui/tooltip.py`):**
- Custom `ToolTip` class that binds to any CTk widget
- On hover, shows a styled popup with:
  - Parameter name (bold)
  - What it controls
  - Effect on protection strength
  - Effect on visual imperceptibility
  - Recommended range for current use case
- Tooltip content pulled from `core/presets.PARAM_INFO` dict

### Task 11: GUI - Technique Panel + Batch Panel + Progress
- `gui/technique_panel.py`: Scrollable cards with checkbox + name + description. Checking/unchecking a technique enables/disables its parameter section in settings
- `gui/batch_panel.py`: File list with thumbnails, add/remove buttons, progress bar
- Background `threading.Thread` with `queue.Queue` for thread-safe progress reporting
- `root.after(100ms)` polling loop to update GUI from queue
- `gui/progress_dialog.py`: Modal dialog during processing

### Task 12: Wire Everything Together + Error Handling
- Connect GUI preset selector -> `PoisonSettings.apply_preset()` -> engine
- Connect individual slider changes -> auto-switch to "Custom" preset -> update `PoisonSettings` field -> update trade-off bar
- Handle: corrupt images (skip + warn), model download failure, OOM (auto-downscale to 4096px), cancel mid-processing
- Report tab: generate summary showing preset used, techniques applied, final PSNR, max pixel delta per image

### Task 13: Verification + Polish
- Test all 4 presets on sample images
- **Minimal Visual Impact:** verify PSNR > 48 dB and SSIM > 0.995 (indistinguishable)
- **Moderate Protection:** verify PSNR > 45 dB (imperceptible)
- **Strong Protection:** verify PSNR > 40 dB (very subtle grain at most)
- Verify switching presets updates all sliders to correct values
- Verify editing a slider auto-switches to "Custom" preset
- Verify trade-off bar updates in real-time
- Verify tooltips appear on hover for all parameter controls
- Test batch processing with 10+ images across different presets
- Verify PNG/JPEG/WebP export
- Verify quality gate auto-fallback: force a high-epsilon scenario and confirm epsilon auto-reduction

## Key Algorithm Details

**FGSM:** `perturbed = original + epsilon * sign(grad(loss, input))` where epsilon comes from preset (Minimal=0.003, Moderate=0.01, Strong=0.025). Gradient computed through ImageNet-normalized forward pass on frozen ResNet-18.

**Style Cloak:** Optimize `perturbed` via Adam to minimize `feature_distance(perturbed, target_style) - lambda * pixel_distance(perturbed, original)` with lambda and iterations from preset. Per-iteration hard clamp to original ± epsilon. Early stop at PSNR < preset threshold.

**Nightshade:** Targeted FGSM: `perturbed = original - epsilon * sign(grad(target_class_loss, input))` with epsilon and target count from preset.

**Noise Patterns:** Pure numpy, no gradients. FFT high-pass filtered Gaussian noise + moire sine grids at aliasing frequencies. Amplitude from preset. No salt-and-pepper (visible).

**Metadata:** piexif EXIF rewrite + LSB bit embedding of "DATAPOISON" in pixel array. Depth from preset (1 bit = LSB only, 2 bits = LSB+1).

**Quality Gate (preset-aware):**
- Minimal: PSNR > 48 dB + SSIM > 0.995 (extreme imperceptibility)
- Moderate: PSNR > 45 dB (professional-grade invisibility)
- Strong: PSNR > 40 dB (very subtle at most)
- Below threshold → auto-reduce all epsilons by 50%, retry up to 2 times
- Result includes: PSNR score, SSIM, max pixel delta, preset used, retry count

**Trade-off Bar Calculation:**
- Normalize each epsilon to [0, 1] within its range
- Weighted average across enabled techniques
- Map to bar position: 0.0 = full left (invisible), 1.0 = full right (max protection)
- Color: green [0.0–0.3], yellow [0.3–0.6], red [0.6–1.0]

## Verification

1. Run `python main.py` - application window opens with preset selector visible
2. Select "Minimal Visual Impact" - all sliders move to low values, trade-off bar shows green
3. Select "Strong Protection" - sliders move to higher values, trade-off bar shifts toward red
4. Load a test image, apply "Minimal Visual Impact" - verify PSNR > 48 dB, SSIM > 0.995, zero visible change
5. Apply "Moderate Protection" - verify PSNR > 45 dB, still indistinguishable
6. Apply "Strong Protection" - verify PSNR > 40 dB, very subtle at most
7. Switch to "Custom" - adjust FGSM epsilon slider to 0.04, verify trade-off bar updates in real-time
8. Hover over each slider - verify tooltip appears with description, protection effect, imperceptibility effect
9. Use "Amplified Difference View" (50x gain) to confirm perturbation exists but is invisible at normal viewing
10. Test batch mode with 10+ images, verify progress bar updates and GUI stays responsive
11. Export as PNG, verify file is valid and poisoned
12. Save as JPEG with metadata poison enabled, check EXIF data shows "DataPoison v1.0" software tag
13. Force quality gate fallback: set Custom epsilons very high, confirm auto-reduction kicks in with warning
