"""FGSM (Fast Gradient Sign Method) adversarial perturbation engine.

Adds gradient-guided noise to an image that confuses AI classifiers and
diffusion models during training, while remaining visually identical to the
original for human viewers.

Algorithm:
  perturbed = original + epsilon * sign(grad(loss, input))

The gradient is computed w.r.t. the INPUT pixels (model weights are frozen).
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


def apply_fgsm(
    image_tensor: torch.Tensor,
    model: Union[nn.Module, List[nn.Module]],
    epsilon: float = 0.01,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Apply FGSM perturbation to an image tensor.

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        model: Pre-trained classifier in eval mode with frozen params.
               Can be a single model or a list of models for ensemble
               gradient averaging (improves cross-model transferability).
        epsilon: Perturbation magnitude. Lower = more invisible.
                 Recommended: 0.003 (minimal), 0.01 (moderate), 0.025 (strong).
        device: torch device.

    Returns:
        Perturbed (1, 3, H, W) tensor clamped to [0, 1].
    """
    if device is None:
        device = image_tensor.device

    # Normalize to list for uniform handling
    models_list = model if isinstance(model, list) else [model]

    # Work on a clone with gradient tracking enabled
    data = image_tensor.clone().detach().to(device).requires_grad_(True)

    # Accumulate gradient signs across all models (ensemble)
    ensemble_grad = torch.zeros_like(data)

    for m in models_list:
        data_i = data.clone().detach().requires_grad_(True)

        # Forward pass with ImageNet normalization (device-aware)
        normalized = _normalize_on_device(data_i, device)
        output = m(normalized)

        # Use the model's own top-1 prediction as the class to disrupt
        pred_class = output.argmax(dim=1)
        loss = F.cross_entropy(output, pred_class)

        # Backward pass: gradient w.r.t. input pixels
        m.zero_grad()
        loss.backward()

        # Accumulate normalized gradient sign
        if data_i.grad is not None:
            ensemble_grad += data_i.grad.data.sign()

    # Average the gradient signs across models
    ensemble_grad = ensemble_grad / len(models_list)

    # Apply signed perturbation using averaged gradient direction
    perturbed = data + epsilon * ensemble_grad.sign()

    # Hard clamp to valid pixel range
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    return perturbed.detach()
