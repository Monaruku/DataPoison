"""PGD (Projected Gradient Descent) iterative adversarial perturbation engine.

Performs multiple small gradient steps instead of a single FGSM step, following
the curvature of the loss landscape for significantly stronger and more
transferable adversarial perturbations at the same epsilon budget.

Algorithm:
  x_0 = original
  for t in 1..num_steps:
      grad = ∇_x Loss(model(x_t), y_pred)
      x_{t+1} = clamp(x_t + alpha * sign(grad), original - epsilon, original + epsilon)
      x_{t+1} = clamp(x_{t+1}, 0, 1)

Where alpha = epsilon / num_steps (step size).

PGD is widely regarded as the strongest first-order adversarial attack and
produces perturbations with 2-5x higher transferability to unseen models
compared to single-step FGSM at the same perturbation budget.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

from core.utils import IMAGENET_MEAN, IMAGENET_STD


def _normalize_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet normalization with mean/std explicitly on the target device."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def apply_pgd(
    image_tensor: torch.Tensor,
    model: Union[nn.Module, List[nn.Module]],
    epsilon: float = 0.01,
    num_steps: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Apply PGD (iterative FGSM) perturbation to an image tensor.

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        model: Pre-trained classifier in eval mode with frozen params.
               Can be a single model or a list of models for ensemble
               gradient averaging (improves cross-model transferability).
        epsilon: Total perturbation budget (L-infinity). Same interpretation
                 as FGSM epsilon — max per-pixel deviation from original.
                 Recommended: 0.003 (minimal), 0.01 (moderate), 0.025 (strong).
        num_steps: Number of iterative gradient steps. More steps follow
                   the loss curvature more closely for stronger perturbations.
                   Recommended: 5 (fast), 10 (moderate), 20 (strong).
        device: torch device.

    Returns:
        Perturbed (1, 3, H, W) tensor clamped to [0, 1].
    """
    if device is None:
        device = image_tensor.device

    # Normalize to list for uniform ensemble handling
    models_list = model if isinstance(model, list) else [model]

    original = image_tensor.clone().detach().to(device)

    # Compute step size: fraction of total budget per iteration
    alpha = epsilon / max(1, num_steps)

    # Get each model's prediction on the clean image (class to disrupt)
    pred_classes = []
    with torch.no_grad():
        for m in models_list:
            normalized_clean = _normalize_on_device(original, device)
            pred_classes.append(m(normalized_clean).argmax(dim=1))

    # Start from the original image
    current = original.clone()

    for _ in range(num_steps):
        # Accumulate gradient signs across all models (ensemble)
        ensemble_grad = torch.zeros_like(current)

        for m, pred_class in zip(models_list, pred_classes):
            data = current.clone().detach().requires_grad_(True)

            # Forward pass with ImageNet normalization
            normalized = _normalize_on_device(data, device)
            output = m(normalized)

            # Maximize loss for the model's original prediction (untargeted attack)
            loss = F.cross_entropy(output, pred_class)

            # Backward pass: gradient w.r.t. input pixels
            m.zero_grad()
            loss.backward()

            if data.grad is not None:
                ensemble_grad += data.grad.data.sign()

        # Average gradient signs across models
        ensemble_grad = ensemble_grad / len(models_list)

        # Take a small step in the averaged gradient sign direction
        current = current + alpha * ensemble_grad.sign()

        # Project back onto the epsilon-ball around the original
        current = torch.clamp(current, original - epsilon, original + epsilon)

        # Hard clamp to valid pixel range
        current = torch.clamp(current, 0.0, 1.0)

    return current.detach()
