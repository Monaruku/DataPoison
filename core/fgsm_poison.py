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
from torchvision import transforms

from core.utils import IMAGENET_MEAN, IMAGENET_STD

# ImageNet normalization transform
_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def apply_fgsm(
    image_tensor: torch.Tensor,
    model: nn.Module,
    epsilon: float = 0.01,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Apply FGSM perturbation to an image tensor.

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        model: Pre-trained classifier in eval mode with frozen params.
        epsilon: Perturbation magnitude. Lower = more invisible.
                 Recommended: 0.003 (minimal), 0.01 (moderate), 0.025 (strong).
        device: torch device.

    Returns:
        Perturbed (1, 3, H, W) tensor clamped to [0, 1].
    """
    if device is None:
        device = image_tensor.device

    # Work on a clone with gradient tracking enabled
    data = image_tensor.clone().detach().to(device).requires_grad_(True)

    # Forward pass with ImageNet normalization
    normalized = _normalize(data.squeeze(0)).unsqueeze(0)
    output = model(normalized)

    # Use the model's own top-1 prediction as the class to disrupt
    pred_class = output.argmax(dim=1)
    loss = F.cross_entropy(output, pred_class)

    # Backward pass: gradient w.r.t. input pixels
    model.zero_grad()
    loss.backward()

    # Collect the gradient and apply signed perturbation
    data_grad = data.grad.data
    perturbed = data + epsilon * data_grad.sign()

    # Hard clamp to valid pixel range
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    return perturbed.detach()
