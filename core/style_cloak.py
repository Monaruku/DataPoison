"""Style cloaking engine (Glaze-inspired).

Shifts the image's representation in the AI model's feature space toward a
different perceived style, while keeping pixel-level changes below the
threshold of human perception.

Algorithm:
  - Hook an intermediate layer (e.g. ResNet layer3) to extract feature maps.
  - Optimize a perturbed copy of the image via Adam to minimize:
      loss = -MSE(f_perturbed, f_target) + lambda * MSE(perturbed, original)
  - Per-iteration clamp to original ± epsilon to prevent visible drift.
  - Early stop if PSNR drops below the quality gate threshold.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import IMAGENET_MEAN, IMAGENET_STD


def _normalize_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet normalization with mean/std explicitly on the target device."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _compute_psnr(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """Compute PSNR in dB between two [0,1] tensors."""
    mse = F.mse_loss(perturbed, original).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _get_feature_hook(model: nn.Module, layer_name: str = "layer3"):
    """Register a forward hook and return (extract_fn, remove_fn)."""
    features = {}

    def hook_fn(module, inp, output):
        features["out"] = output

    target = model
    for attr in layer_name.split("."):
        target = getattr(target, attr)

    handle = target.register_forward_hook(hook_fn)

    def extract(x: torch.Tensor) -> torch.Tensor:
        features.clear()
        model(_normalize_on_device(x, x.device))
        return features.get("out")

    return extract, handle.remove


def _generate_style_target(
    original_shape: tuple,
    style: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a synthetic target feature reference for style cloaking.
    In a full implementation this would load real style reference images;
    here we use deterministic noise patterns that push features away from
    the original while remaining well-defined.
    """
    # Use a fixed seed based on style name for reproducibility
    import hashlib
    seed = int(hashlib.md5(style.encode()).hexdigest()[:8], 16) % (2**31)
    gen = torch.Generator(device=device).manual_seed(seed)

    # Shape matches the feature map, not the image
    # We'll generate at image resolution and let the feature extractor handle it
    _, _, H, W = original_shape
    target_image = torch.rand(1, 3, H, W, generator=gen, device=device)
    return target_image


def apply_style_cloak(
    image_tensor: torch.Tensor,
    model: nn.Module,
    epsilon: float = 0.008,
    lam: float = 5.0,
    num_iterations: int = 40,
    style_target: str = "random",
    device: torch.device = None,
    min_psnr: float = 45.0,
) -> torch.Tensor:
    """
    Apply style cloaking to shift the image in feature space.

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        model: Pre-trained classifier (eval mode, frozen params).
        epsilon: Max per-pixel deviation from original (hard clamp).
        lam: Regularization strength. Higher = less pixel change.
        num_iterations: Adam optimizer steps.
        style_target: Target style name ("random", "abstract", "impressionist").
        device: torch device.
        min_psnr: Early stop if PSNR drops below this (dB).

    Returns:
        Cloaked (1, 3, H, W) tensor clamped to [0, 1].
    """
    if device is None:
        device = image_tensor.device

    original = image_tensor.clone().detach().to(device)

    # Set up feature extraction hook
    extract_features, remove_hook = _get_feature_hook(model, layer_name="layer3")

    try:
        # Get original features (target to move AWAY from)
        f_original = extract_features(original).detach()

        # Generate target style reference image and extract its features
        target_img = _generate_style_target(original.shape, style_target, device)
        f_target = extract_features(target_img).detach()

        # Direction: push features from f_original toward f_target
        # We want f_perturbed ≈ f_target, so optimize:
        #   loss = -MSE(f_perturbed, f_target) + lam * MSE(perturbed, original)
        # The negative sign on feature loss pushes perturbed features toward target.

        # Initialize perturbed as original with gradient tracking
        perturbed = original.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([perturbed], lr=0.005)

        best_perturbed = original.clone()
        best_feature_dist = float("inf")

        for iteration in range(num_iterations):
            f_perturbed = extract_features(perturbed)

            # Feature loss: negative MSE toward target (push features there)
            feature_loss = -F.mse_loss(f_perturbed, f_target)

            # Pixel regularization: penalize deviation from original
            pixel_loss = F.mse_loss(perturbed, original)

            total_loss = feature_loss + lam * pixel_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Hard clamp: perturbed must stay within epsilon of original
            with torch.no_grad():
                perturbed.data = torch.clamp(perturbed.data, original - epsilon, original + epsilon)
                perturbed.data = torch.clamp(perturbed.data, 0.0, 1.0)

            # Track best result (lowest feature distance to target)
            with torch.no_grad():
                f_curr = extract_features(perturbed)
                curr_dist = F.mse_loss(f_curr, f_target).item()
                if curr_dist < best_feature_dist:
                    best_feature_dist = curr_dist
                    best_perturbed = perturbed.detach().clone()

            # Early stop if PSNR drops below threshold
            psnr = _compute_psnr(original, perturbed.detach())
            if psnr < min_psnr:
                break

        return best_perturbed.detach()

    finally:
        remove_hook()
