"""Adaptive per-region perturbation scaling (visual masking).

Exploits the human visual system's contrast sensitivity and texture masking
properties to allocate more perturbation budget to textured/busy regions
and less to smooth/uniform regions.

This acts as a force multiplier for all gradient-based techniques (FGSM, PGD,
Nightshade, Style Cloak) — it makes each one more effective per unit of
perceptual distortion.

Algorithm:
  1. Compute local texture complexity via Laplacian variance on small patches.
  2. Normalize to [min_weight, 1.0] to create a spatial weight map.
  3. Apply the weight map to scale perturbation gradients per-pixel.

The Laplacian operator measures local second-order derivatives (edges, texture).
High Laplacian variance = detailed/textured region = perturbation is invisible.
Low Laplacian variance = smooth region = perturbation may be visible.
"""
import torch
import torch.nn.functional as F


def compute_complexity_map(
    image_tensor: torch.Tensor,
    kernel_size: int = 16,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """
    Compute a spatial complexity weight map from an image tensor.

    Uses Laplacian variance on local patches to estimate texture density.
    Returns a weight map in [min_weight, 1.0] where 1.0 = highly textured
    (perturbation fully applied) and min_weight = very smooth (perturbation
    reduced to preserve imperceptibility).

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        kernel_size: Patch size for local Laplacian variance computation.
                     Larger = smoother transitions between regions.
        min_weight: Minimum weight for the smoothest regions (0.0-1.0).
                    0.3 means smooth areas get 30% of the full perturbation.

    Returns:
        (1, 1, H, W) weight map tensor in [min_weight, 1.0].
    """
    # Convert to grayscale for edge detection (luminance channel)
    # Standard luminance weights: 0.299R + 0.587G + 0.114B
    _, _, H, W = image_tensor.shape

    gray = (
        0.299 * image_tensor[:, 0:1, :, :]
        + 0.587 * image_tensor[:, 1:2, :, :]
        + 0.114 * image_tensor[:, 2:3, :, :]
    )  # (1, 1, H, W)

    # Laplacian kernel (second-order derivative, detects edges/texture)
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(1, 1, 3, 3)

    # Compute Laplacian response (3x3 kernel with padding=1 preserves size)
    laplacian = F.conv2d(gray, laplacian_kernel, padding=1)

    # Local mean of squared Laplacian ≈ local variance (texture indicator)
    # Use avg_pool2d which correctly handles both even and odd kernel sizes
    laplacian_sq = laplacian ** 2
    local_var = F.avg_pool2d(
        laplacian_sq,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )

    # Resize to original spatial dims (avg_pool can shift size by ±1 with even kernels)
    if local_var.shape[2] != H or local_var.shape[3] != W:
        local_var = F.interpolate(local_var, size=(H, W), mode="bilinear", align_corners=False)

    # Normalize to [0, 1] then scale to [min_weight, 1.0]
    var_max = local_var.max()
    if var_max > 1e-8:
        normalized = local_var / var_max
    else:
        # Uniform image — return uniform weight map
        normalized = torch.ones(1, 1, H, W, dtype=image_tensor.dtype, device=image_tensor.device) * 0.5

    # Smooth the weight map with a Gaussian-like blur to avoid sharp transitions
    blur_kernel_size = max(3, kernel_size // 2)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1  # ensure odd for exact size preservation
    blur_kernel = torch.ones(
        1, 1, blur_kernel_size, blur_kernel_size,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ) / (blur_kernel_size * blur_kernel_size)

    weight_map = F.conv2d(normalized, blur_kernel, padding=blur_kernel_size // 2)

    # Final safety resize to guarantee exact match with input dimensions
    if weight_map.shape[2] != H or weight_map.shape[3] != W:
        weight_map = F.interpolate(weight_map, size=(H, W), mode="bilinear", align_corners=False)

    # Re-normalize after blur (blur can reduce peak values)
    wm_max = weight_map.max()
    if wm_max > 1e-8:
        weight_map = weight_map / wm_max

    # Scale to [min_weight, 1.0]
    weight_map = min_weight + (1.0 - min_weight) * weight_map

    # Clamp to ensure bounds
    weight_map = torch.clamp(weight_map, min_weight, 1.0)

    return weight_map


def apply_visual_masking(
    perturbation: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    """
    Scale a perturbation by the visual complexity weight map.

    Args:
        perturbation: (1, 3, H, W) perturbation tensor (the noise to add).
        weight_map: (1, 1, H, W) weight map from compute_complexity_map().

    Returns:
        Scaled (1, 3, H, W) perturbation tensor.
    """
    # Broadcast weight map across 3 channels
    return perturbation * weight_map
