"""Nightshade-inspired prompt-specific poisoning engine.

Creates perturbations that cause text-to-image models to learn incorrect
associations during training. Uses a targeted variant of FGSM: instead of
maximizing loss for the true class, it *minimizes* loss for a semantically
distant wrong class.

Algorithm:
  perturbed = original - epsilon * sign(grad(target_class_loss, input))

The minus sign pulls the image toward the wrong class in feature space.
Multiple semantically distant targets can be applied in sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import IMAGENET_MEAN, IMAGENET_STD


def _normalize_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet normalization with mean/std explicitly on the target device."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std

# Semantic clusters of ImageNet classes (indices).
# We pick representative classes from very different semantic groups
# so that poisoning targets are maximally distant from the true class.
# Groups: animals, vehicles, furniture, food, instruments, clothing, nature
SEMANTIC_GROUPS = {
    "animals":     [207, 208, 235, 263, 292, 340, 386],
    "vehicles":    [436, 468, 511, 609, 656, 705, 757],
    "furniture":   [422, 531, 551, 621, 726, 737, 831],
    "food":        [923, 926, 930, 934, 941, 945, 954],
    "instruments": [401, 486, 546, 593, 642, 697, 776],
    "clothing":    [411, 458, 502, 610, 661, 718, 824],
    "nature":      [980, 981, 982, 983, 984, 985, 987],
}

# Map each group to the groups most distant from it
DISTANT_PAIRS = {
    "animals":     ["vehicles", "instruments"],
    "vehicles":    ["animals", "nature"],
    "furniture":   ["food", "clothing"],
    "food":        ["furniture", "instruments"],
    "instruments": ["animals", "vehicles"],
    "clothing":    ["nature", "food"],
    "nature":      ["vehicles", "clothing"],
}


def _classify(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """Return the top-1 predicted class index for the image."""
    normalized = _normalize_on_device(image_tensor, device)
    with torch.no_grad():
        output = model(normalized)
    return output.argmax(dim=1).item()


def _pick_target_classes(true_class: int, num_targets: int, model: nn.Module,
                          image_tensor: torch.Tensor, device: torch.device) -> list:
    """
    Pick target classes that are semantically distant from the true class.
    Falls back to random distant classes if the true class group is unknown.
    """
    # Find which group the true class belongs to
    true_group = None
    for group, classes in SEMANTIC_GROUPS.items():
        if true_class in classes:
            true_group = group
            break

    # Get distant groups
    if true_group and true_group in DISTANT_PAIRS:
        distant_groups = DISTANT_PAIRS[true_group]
    else:
        # Fallback: pick from all groups
        distant_groups = list(SEMANTIC_GROUPS.keys())

    # Collect candidate target classes from distant groups
    candidates = []
    for g in distant_groups:
        candidates.extend(SEMANTIC_GROUPS[g])

    # Pick num_targets from candidates (cycle if needed)
    targets = []
    for i in range(num_targets):
        targets.append(candidates[i % len(candidates)])

    return targets


def _targeted_fgsm_step(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_class: int,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """Single targeted FGSM step: pull image toward target_class."""
    data = image_tensor.clone().detach().to(device).requires_grad_(True)
    normalized = _normalize_on_device(data, device)
    output = model(normalized)

    target_tensor = torch.tensor([target_class], device=device)
    loss = F.cross_entropy(output, target_tensor)

    model.zero_grad()
    loss.backward()

    # Subtract gradient: pull image TOWARD the target class
    data_grad = data.grad.data
    perturbed = data - epsilon * data_grad.sign()
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    return perturbed.detach()


def apply_nightshade(
    image_tensor: torch.Tensor,
    model: nn.Module,
    epsilon: float = 0.01,
    num_targets: int = 2,
    device: torch.device = None,
    multi_pass: bool = False,
) -> torch.Tensor:
    """
    Apply Nightshade-style prompt poisoning.

    Args:
        image_tensor: (1, 3, H, W) float tensor in [0, 1].
        model: Pre-trained classifier (eval mode, frozen params).
        epsilon: Perturbation magnitude per target.
        num_targets: Number of semantically distant target classes.
        device: torch device.
        multi_pass: If True, run multiple passes with decreasing epsilon.

    Returns:
        Poisoned (1, 3, H, W) tensor clamped to [0, 1].
    """
    if device is None:
        device = image_tensor.device

    current = image_tensor.clone().to(device)
    original = image_tensor.clone().to(device)

    # Classify original image
    true_class = _classify(model, current, device)
    target_classes = _pick_target_classes(true_class, num_targets, model, current, device)

    # Scale epsilon by sqrt(n_targets) to keep total perturbation bounded
    effective_epsilon = epsilon / max(1, num_targets ** 0.5)

    for target_class in target_classes:
        current = _targeted_fgsm_step(current, model, target_class, effective_epsilon, device)
        # Re-clamp to original ± epsilon to prevent visible drift
        current = torch.clamp(current, original - epsilon, original + epsilon)
        current = torch.clamp(current, 0.0, 1.0)

    return current
