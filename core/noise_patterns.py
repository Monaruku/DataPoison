"""Structured noise injection patterns.

Applies ultra-low-amplitude noise at frequencies that are invisible to
human perception but disrupt AI preprocessing pipelines (resizing,
normalization, feature extraction).

Techniques:
  - High-frequency Gaussian: FFT high-pass filtered noise
  - Moire patterns: Sine grids at frequencies that alias under common
    AI resize operations (224x224, 512x512)

NO salt-and-pepper: produces visible outlier pixels.
"""
import numpy as np
import torch


def _high_frequency_gaussian(
    image_array: np.ndarray,
    amplitude: float = 0.008,
    cutoff_freq: float = 0.4,
) -> np.ndarray:
    """
    Generate high-pass filtered Gaussian noise and add it to the image.
    The noise lives in frequency bands above human visual acuity.

    Args:
        image_array: (H, W, 3) float array in [0, 1].
        amplitude: Peak noise amplitude (max pixel change ≈ amplitude).
        cutoff_freq: Normalized cutoff frequency (0.4 = top 60% of spectrum).

    Returns:
        Perturbed (H, W, 3) float array in [0, 1].
    """
    H, W, C = image_array.shape

    # Generate Gaussian noise
    noise = np.random.normal(0, 1, (H, W, C)).astype(np.float32)

    # High-pass filter via FFT
    noise_filtered = np.zeros_like(noise)
    for c in range(C):
        fft = np.fft.fft2(noise[:, :, c])
        # Create frequency grid
        fy = np.fft.fftfreq(H)[:, np.newaxis]
        fx = np.fft.fftfreq(W)[np.newaxis, :]
        freq_mag = np.sqrt(fy**2 + fx**2)
        # Zero out low frequencies
        mask = (freq_mag > cutoff_freq * 0.5).astype(np.float32)
        fft_filtered = fft * mask
        noise_filtered[:, :, c] = np.real(np.fft.ifft2(fft_filtered))

    # Normalize to unit variance then scale by amplitude
    std = noise_filtered.std()
    if std > 0:
        noise_filtered = noise_filtered / std * amplitude

    result = image_array + noise_filtered
    return np.clip(result, 0.0, 1.0)


def _moire_pattern(
    image_array: np.ndarray,
    amplitude: float = 0.008,
) -> np.ndarray:
    """
    Add moire interference patterns at frequencies that alias under
    common AI image resize operations (224, 256, 512 px targets).

    The pattern consists of overlapping sine grids that create beat
    frequencies disruptive to downsampling.

    Args:
        image_array: (H, W, 3) float array in [0, 1].
        amplitude: Peak pattern amplitude.

    Returns:
        Perturbed (H, W, 3) float array in [0, 1].
    """
    H, W, C = image_array.shape

    # Create coordinate grids
    y = np.arange(H, dtype=np.float32)
    x = np.arange(W, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Frequencies chosen to alias at common AI input sizes
    # These create patterns at ~7-11 pixel periods that produce
    # visible artifacts when resized to 224x224 or 512x512
    freqs = [
        (2.0 * np.pi / 7.0, 0.0),      # 7-pixel horizontal
        (0.0, 2.0 * np.pi / 9.0),       # 9-pixel vertical
        (2.0 * np.pi / 11.0, 2.0 * np.pi / 11.0),  # diagonal
    ]

    pattern = np.zeros((H, W), dtype=np.float32)
    for fy, fx in freqs:
        pattern += np.sin(fy * yy + fx * xx)

    # Normalize to [-1, 1] then scale
    pmax = np.abs(pattern).max()
    if pmax > 0:
        pattern = pattern / pmax * amplitude

    # Apply same pattern to all channels
    result = image_array.copy()
    for c in range(C):
        result[:, :, c] += pattern

    return np.clip(result, 0.0, 1.0)


def apply_noise_patterns(
    image_tensor: torch.Tensor,
    amplitude: float = 0.008,
    patterns: list = None,
) -> torch.Tensor:
    """
    Apply structured noise patterns to an image tensor.

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        amplitude: Peak noise amplitude (all patterns scale with this).
        patterns: List of pattern names to apply.
                  Options: "hf_gaussian", "moire". Default: both.

    Returns:
        Perturbed (1, 3, H, W) tensor clamped to [0, 1].
    """
    if patterns is None:
        patterns = ["hf_gaussian", "moire"]

    # Convert to numpy for FFT operations
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    original = img_np.copy()

    # Scale amplitude per pattern to keep total bounded
    per_pattern_amp = amplitude / max(1, len(patterns) ** 0.5)

    if "hf_gaussian" in patterns:
        img_np = _high_frequency_gaussian(img_np, amplitude=per_pattern_amp)

    if "moire" in patterns:
        img_np = _moire_pattern(img_np, amplitude=per_pattern_amp)

    # Clamp to original ± amplitude to prevent drift
    img_np = np.clip(img_np, original - amplitude, original + amplitude)
    img_np = np.clip(img_np, 0.0, 1.0)

    # Convert back to tensor
    result = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float()
    return result.to(image_tensor.device)
