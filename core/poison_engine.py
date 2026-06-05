"""Poison engine orchestrator.

Coordinates all poisoning techniques, applies them sequentially to an image,
and runs the quality gate to ensure visual imperceptibility.

The engine accepts a PoisonSettings dataclass (populated from a preset or
custom values) and dispatches to individual technique engines.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
from PIL import Image

from core.presets import PRESETS
from core.utils import pil_to_tensor, tensor_to_pil, downscale_if_needed
from core.fgsm_poison import apply_fgsm
from core.style_cloak import apply_style_cloak
from core.nightshade_poison import apply_nightshade
from core.noise_patterns import apply_noise_patterns
from core.metadata_poison import apply_metadata_poison
from core.quality_gate import (
    check_quality,
    reduce_settings_for_retry,
    MAX_RETRIES,
)
from models.model_loader import get_model, get_device


@dataclass
class PoisonSettings:
    """All configurable parameters for the poisoning pipeline."""
    preset: str = "moderate"

    # FGSM
    fgsm_enabled: bool = True
    fgsm_epsilon: float = 0.01

    # Style Cloak
    style_cloak_enabled: bool = True
    style_cloak_epsilon: float = 0.008
    style_cloak_lambda: float = 5.0
    style_cloak_iterations: int = 40
    style_cloak_target: str = "random"

    # Nightshade
    nightshade_enabled: bool = True
    nightshade_epsilon: float = 0.01
    nightshade_targets: int = 2

    # Noise Patterns
    noise_enabled: bool = True
    noise_amplitude: float = 0.008
    noise_patterns_enabled: list = field(default_factory=lambda: ["hf_gaussian", "moire"])

    # Metadata
    metadata_enabled: bool = True
    metadata_strip_exif: bool = True
    metadata_watermark: bool = True
    metadata_watermark_depth: int = 1

    # Output
    output_format: str = "png"
    model_name: str = "resnet18"
    multi_pass: bool = False

    # Quality gate
    quality_gate_enabled: bool = True
    quality_gate_psnr: float = 45.0
    quality_gate_ssim: float = 0.99

    @classmethod
    def from_preset(cls, preset_name: str) -> "PoisonSettings":
        """Create settings from a named preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        params = PRESETS[preset_name]
        # Filter only fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in params.items() if k in valid_fields}
        settings = cls(**filtered)
        settings.preset = preset_name
        return settings

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PoisonResult:
    """Result of processing a single image."""
    original: Image.Image
    poisoned: Image.Image
    techniques_applied: List[str]
    settings_used: PoisonSettings
    quality_report: dict
    processing_time_ms: float
    retries: int = 0
    warnings: List[str] = field(default_factory=list)


class PoisonEngine:
    """Main orchestrator that applies poisoning techniques to images."""

    def __init__(self):
        self._device = get_device()
        self._model_cache = {}

    def _get_model(self, name: str):
        """Get or load a model by name."""
        if name not in self._model_cache:
            self._model_cache[name] = get_model(name, self._device)
        return self._model_cache[name]

    def _apply_techniques(
        self,
        image_tensor: torch.Tensor,
        settings: PoisonSettings,
        model: torch.nn.Module,
    ) -> tuple:
        """Apply all enabled techniques sequentially. Returns (tensor, list_of_applied)."""
        current = image_tensor.clone()
        applied = []

        # Count enabled techniques for epsilon scaling
        n_enabled = sum([
            settings.fgsm_enabled,
            settings.style_cloak_enabled,
            settings.nightshade_enabled,
            settings.noise_enabled,
        ])
        scale = max(1, n_enabled ** 0.5)

        # 1. FGSM
        if settings.fgsm_enabled:
            eff_eps = settings.fgsm_epsilon / scale
            current = apply_fgsm(current, model, epsilon=eff_eps, device=self._device)
            applied.append("FGSM Adversarial Noise")

        # 2. Style Cloak
        if settings.style_cloak_enabled:
            eff_eps = settings.style_cloak_epsilon / scale
            current = apply_style_cloak(
                current, model,
                epsilon=eff_eps,
                lam=settings.style_cloak_lambda,
                num_iterations=settings.style_cloak_iterations,
                style_target=settings.style_cloak_target,
                device=self._device,
                min_psnr=settings.quality_gate_psnr if settings.quality_gate_enabled else 0.0,
            )
            applied.append("Style Cloak")

        # 3. Nightshade
        if settings.nightshade_enabled:
            eff_eps = settings.nightshade_epsilon / scale
            current = apply_nightshade(
                current, model,
                epsilon=eff_eps,
                num_targets=settings.nightshade_targets,
                device=self._device,
                multi_pass=settings.multi_pass,
            )
            applied.append("Prompt Poison")

        # 4. Noise Patterns (no model needed)
        if settings.noise_enabled:
            eff_amp = settings.noise_amplitude / scale
            current = apply_noise_patterns(
                current,
                amplitude=eff_amp,
                patterns=settings.noise_patterns_enabled,
            )
            applied.append("High-Frequency Noise")

        return current, applied

    def process_image(
        self,
        pil_image: Image.Image,
        settings: PoisonSettings,
        save_path: Optional[str] = None,
        progress_callback=None,
    ) -> PoisonResult:
        """
        Process a single image through the full poisoning pipeline.

        Args:
            pil_image: Source PIL Image.
            settings: PoisonSettings with all parameters.
            save_path: Optional output path (needed for EXIF metadata on JPEG).
            progress_callback: Optional callable(step, message) for progress updates.

        Returns:
            PoisonResult with original, poisoned image, quality report, and metadata.
        """
        start_time = time.time()
        warnings = []

        # Downscale if needed to prevent OOM
        original_image = downscale_if_needed(pil_image.copy(), max_dim=4096)
        if original_image.size != pil_image.size:
            warnings.append(f"Image downscaled from {pil_image.size} to {original_image.size} to prevent memory issues.")

        # Convert to tensor
        image_tensor = pil_to_tensor(original_image).to(self._device)
        original_tensor = image_tensor.clone()

        # Load model
        model = self._get_model(settings.model_name)

        # Apply techniques with optional quality gate retry
        retries = 0
        current_settings = settings
        perturbed_tensor = None
        applied = []

        if not settings.quality_gate_enabled:
            # Bypass quality gate — apply once, no retry
            if progress_callback:
                progress_callback(0.1, "Applying techniques (quality gate bypassed)...")
            perturbed_tensor, applied = self._apply_techniques(
                image_tensor, current_settings, model
            )
            warnings.append(
                "Quality gate was bypassed. Output may have visible artifacts."
            )
        else:
            for attempt in range(MAX_RETRIES + 1):
                if progress_callback:
                    progress_callback(0.1, f"Applying techniques (attempt {attempt + 1})...")

                perturbed_tensor, applied = self._apply_techniques(
                    image_tensor, current_settings, model
                )

                # Quality gate check
                quality = check_quality(
                    original_tensor, perturbed_tensor,
                    psnr_threshold=current_settings.quality_gate_psnr,
                    ssim_threshold=current_settings.quality_gate_ssim,
                )

                if quality["passed"]:
                    break

                if attempt < MAX_RETRIES:
                    retries += 1
                    warnings.append(
                        f"Quality gate failed (PSNR={quality['psnr']}dB, SSIM={quality['ssim']}). "
                        f"Reducing parameters by 50% and retrying..."
                    )
                    reduced = reduce_settings_for_retry(current_settings.to_dict())
                    current_settings = PoisonSettings(**reduced)
                else:
                    warnings.append(
                        f"Quality gate: PSNR={quality['psnr']}dB, SSIM={quality['ssim']} after "
                        f"{MAX_RETRIES} retries. Protection was weakened to maintain invisibility."
                    )

        # Convert back to PIL
        poisoned_pil = tensor_to_pil(perturbed_tensor)

        # Apply metadata poisoning (operates on PIL image)
        if settings.metadata_enabled:
            if progress_callback:
                progress_callback(0.8, "Applying metadata protection...")
            applied_metadata = "Metadata Scramble"

            poisoned_pil = apply_metadata_poison(
                poisoned_pil,
                strip_exif=settings.metadata_strip_exif,
                embed_watermark=settings.metadata_watermark,
                watermark_depth=settings.metadata_watermark_depth,
                output_format=settings.output_format,
                save_path=save_path,
            )
            applied.append(applied_metadata)

        # Final quality report — ensure both tensors are on the same device
        poisoned_tensor_final = pil_to_tensor(poisoned_pil).to(self._device)
        final_quality = check_quality(
            original_tensor, poisoned_tensor_final,
            psnr_threshold=settings.quality_gate_psnr,
            ssim_threshold=settings.quality_gate_ssim,
        )

        elapsed = (time.time() - start_time) * 1000

        return PoisonResult(
            original=original_image,
            poisoned=poisoned_pil,
            techniques_applied=applied,
            settings_used=current_settings,
            quality_report=final_quality,
            processing_time_ms=round(elapsed, 1),
            retries=retries,
            warnings=warnings,
        )
