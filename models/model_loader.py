"""Lazy-loading and caching for pre-trained torchvision models."""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

_model_cache: Dict[str, nn.Module] = {}


def get_device() -> torch.device:
    """Auto-detect CUDA availability, fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model(name: str = "resnet18", device: Optional[torch.device] = None) -> nn.Module:
    """
    Load and cache a pre-trained model in eval mode with frozen parameters.
    Only input gradients are needed (for FGSM), not weight gradients.
    """
    if device is None:
        device = get_device()

    cache_key = f"{name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {name}. Supported: resnet18, vgg16")

    model = model.to(device)
    model.eval()

    # Freeze all model parameters — we only compute gradients w.r.t. input pixels
    for param in model.parameters():
        param.requires_grad = False

    _model_cache[cache_key] = model
    return model


def get_feature_extractor(model_name: str = "resnet18", layer: str = "layer3",
                          device: Optional[torch.device] = None):
    """
    Return a callable that extracts intermediate features from a model layer.
    Uses a forward hook to capture output from the specified layer.
    """
    model = get_model(model_name, device)
    features: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        features["out"] = output

    # Attach hook to the specified layer
    target_layer = model
    for attr in layer.split("."):
        target_layer = getattr(target_layer, attr)
    target_layer.register_forward_hook(hook_fn)

    def extract(x: torch.Tensor) -> torch.Tensor:
        features.clear()
        with torch.no_grad():
            model(x)
        return features.get("out")

    return extract
