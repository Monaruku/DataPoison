"""Quality gate: validates that poisoned images meet imperceptibility thresholds.

Computes PSNR (and optionally SSIM) between original and poisoned images.
If the quality falls below the preset's threshold, the engine auto-reduces
epsilon and retries up to MAX_RETRIES times.

Thresholds by preset:
  Minimal:  PSNR > 48 dB, SSIM > 0.995
  Moderate: PSNR > 45 dB, SSIM > 0.99
  Strong:   PSNR > 40 dB, SSIM > 0.97
"""
import numpy as np
import torch
import torch.nn.functional as F

MAX_RETRIES = 2


def compute_psnr(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB between two [0,1] tensors."""
    mse = F.mse_loss(perturbed, original).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index (simplified version).
    Uses mean, variance, and covariance of the two images.
    """
    C1 = 0.01 ** 2  # stability constant
    C2 = 0.03 ** 2

    mu_x = original.mean()
    mu_y = perturbed.mean()
    sigma_x2 = original.var()
    sigma_y2 = perturbed.var()
    sigma_xy = ((original - mu_x) * (perturbed - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return (numerator / denominator).item()


def compute_max_pixel_delta(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """Return the maximum absolute pixel change (in 0-255 scale)."""
    return ((perturbed - original).abs().max().item() * 255.0)


def check_quality(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    psnr_threshold: float = 45.0,
    ssim_threshold: float = 0.99,
) -> dict:
    """
    Check if the perturbed image meets the quality gate requirements.

    Returns a dict with:
      - psnr: float (dB)
      - ssim: float
      - max_pixel_delta: float (0-255 scale)
      - passed: bool (True if both PSNR and SSIM meet thresholds)
    """
    psnr = compute_psnr(original, perturbed)
    ssim = compute_ssim(original, perturbed)
    max_delta = compute_max_pixel_delta(original, perturbed)

    passed = (psnr >= psnr_threshold) and (ssim >= ssim_threshold)

    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "max_pixel_delta": round(max_delta, 2),
        "passed": passed,
        "psnr_threshold": psnr_threshold,
        "ssim_threshold": ssim_threshold,
    }


def reduce_settings_for_retry(settings_dict: dict) -> dict:
    """
    Halve all epsilon/amplitude values in settings for a quality gate retry.
    Returns a new dict with reduced values.
    """
    reduced = settings_dict.copy()
    reduction_keys = [
        "fgsm_epsilon",
        "style_cloak_epsilon",
        "nightshade_epsilon",
        "noise_amplitude",
    ]
    for key in reduction_keys:
        if key in reduced:
            reduced[key] = reduced[key] * 0.5
    return reduced
