"""Image and tensor conversion utilities."""
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a float tensor with batch dim, shape (1, 3, H, W), values in [0, 1]."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = _to_tensor(image)
    return tensor.unsqueeze(0)  # add batch dim


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a float tensor (1, 3, H, W) or (3, H, W) in [0, 1] back to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0.0, 1.0).cpu()
    return _to_pil(tensor)


def validate_image(path: str) -> bool:
    """Check if a file is a valid, openable image."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(path: str) -> dict:
    """Return basic image metadata."""
    with Image.open(path) as img:
        return {
            "path": path,
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": os.path.getsize(path),
        }


def save_image(image: Image.Image, path: str, fmt: str = "png") -> None:
    """Save a PIL image with appropriate quality settings."""
    fmt_lower = fmt.lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt_lower in ("jpg", "jpeg"):
        image.save(path, format="JPEG", quality=95)
    elif fmt_lower == "webp":
        image.save(path, format="WEBP", quality=95)
    else:
        image.save(path, format="PNG")


def downscale_if_needed(image: Image.Image, max_dim: int = 4096) -> Image.Image:
    """Downscale image so longest edge <= max_dim, preserving aspect ratio."""
    longest = max(image.width, image.height)
    if longest <= max_dim:
        return image
    ratio = max_dim / longest
    new_w = int(image.width * ratio)
    new_h = int(image.height * ratio)
    return image.resize((new_w, new_h), Image.LANCZOS)
