"""Preset definitions and parameter metadata for the poisoning engine."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Parameter metadata: tooltips and descriptions for the GUI
# ─────────────────────────────────────────────────────────────────────────────

PARAM_INFO: Dict[str, Dict[str, Any]] = {
    "fgsm_epsilon": {
        "label": "FGSM Perturbation Strength",
        "description": "Controls the magnitude of gradient-based adversarial noise added to each pixel.",
        "protection_effect": (
            "Higher values create stronger perturbations that more effectively confuse AI classifiers "
            "and diffusion models during training."
        ),
        "imperceptibility_effect": (
            "Values below 0.008 are virtually invisible to human eyes. "
            "0.01–0.02 may produce a faint grain visible only on close inspection of flat areas. "
            "Above 0.025, noise texture becomes visible on smooth backgrounds."
        ),
        "recommended": "Minimal: 0.001–0.005  |  Moderate: 0.008–0.015  |  Strong: 0.02–0.03",
        "range": (0.001, 0.05),
        "step": 0.001,
    },
    "pgd_epsilon": {
        "label": "PGD Perturbation Strength",
        "description": "Total perturbation budget for the iterative PGD attack (L-infinity bound). Same scale as FGSM epsilon.",
        "protection_effect": (
            "Higher values allow a larger total perturbation across all PGD iterations, "
            "yielding stronger adversarial transferability to unseen AI models."
        ),
        "imperceptibility_effect": (
            "Same visual thresholds as FGSM — values below 0.008 are invisible. "
            "The per-step change is epsilon/num_steps, so each individual step is extremely subtle."
        ),
        "recommended": "Minimal: 0.001–0.005  |  Moderate: 0.008–0.015  |  Strong: 0.02–0.03",
        "range": (0.001, 0.05),
        "step": 0.001,
    },
    "pgd_steps": {
        "label": "PGD Iterations",
        "description": "Number of iterative gradient steps. Each step uses epsilon/steps as its step size.",
        "protection_effect": (
            "More iterations follow the loss landscape curvature more closely, producing stronger "
            "and more transferable perturbations. Diminishing returns above ~20 steps."
        ),
        "imperceptibility_effect": (
            "Each step is epsilon/num_steps in magnitude, so individual steps are imperceptible. "
            "The total perturbation is clamped to epsilon regardless of iteration count."
        ),
        "recommended": "Minimal: 5  |  Moderate: 10  |  Strong: 20",
        "range": (2, 30),
        "step": 1,
    },
    "style_cloak_epsilon": {
        "label": "Style Cloak Perturbation",
        "description": "Maximum per-pixel change allowed during style cloaking optimization.",
        "protection_effect": (
            "Larger epsilon lets the optimizer push the image further in feature space, "
            "making the AI perceive a completely different style."
        ),
        "imperceptibility_effect": (
            "Values below 0.008 are indistinguishable from the original. "
            "The optimizer is also constrained by the regularization strength (lambda)."
        ),
        "recommended": "Minimal: 0.001–0.004  |  Moderate: 0.005–0.012  |  Strong: 0.015–0.025",
        "range": (0.001, 0.04),
        "step": 0.001,
    },
    "style_cloak_lambda": {
        "label": "Style Cloak Regularization (λ)",
        "description": "Controls how strongly the optimizer resists changing pixels from the original.",
        "protection_effect": (
            "Lower lambda allows more aggressive feature-space movement (stronger protection) "
            "at the cost of slightly more pixel change."
        ),
        "imperceptibility_effect": (
            "Higher lambda (8–10) keeps pixels very close to the original. "
            "Lower lambda (2–3) allows more change for stronger protection."
        ),
        "recommended": "Minimal: 8–10  |  Moderate: 4–6  |  Strong: 2–4",
        "range": (1.0, 10.0),
        "step": 0.5,
    },
    "style_cloak_iterations": {
        "label": "Optimization Iterations",
        "description": "Number of Adam optimizer steps for style cloaking.",
        "protection_effect": (
            "More iterations allow the optimizer to find a better perturbation, "
            "increasing feature-space distance from the original style."
        ),
        "imperceptibility_effect": (
            "With strong regularization, even 100 iterations remain invisible. "
            "The per-iteration pixel clamp prevents drift regardless of iteration count."
        ),
        "recommended": "Minimal: 15–25  |  Moderate: 30–50  |  Strong: 60–100",
        "range": (5, 120),
        "step": 5,
    },
    "nightshade_epsilon": {
        "label": "Prompt Poison Strength",
        "description": "Magnitude of the targeted perturbation that causes wrong AI associations.",
        "protection_effect": (
            "Higher values cause stronger misclassification, making text-to-image models "
            "learn incorrect associations for the protected image."
        ),
        "imperceptibility_effect": (
            "Same as FGSM — values below 0.008 are invisible. "
            "The perturbation is gradient-guided but in the opposite direction."
        ),
        "recommended": "Minimal: 0.001–0.005  |  Moderate: 0.008–0.015  |  Strong: 0.02–0.03",
        "range": (0.001, 0.05),
        "step": 0.001,
    },
    "nightshade_targets": {
        "label": "Number of Poison Targets",
        "description": "How many semantically distant target classes to poison against per pass.",
        "protection_effect": (
            "More targets cause the model to learn multiple wrong associations, "
            "compounding the poisoning effect."
        ),
        "imperceptibility_effect": (
            "Each additional target adds a small amount of perturbation. "
            "Epsilon is automatically divided by sqrt(n_targets) to keep the total invisible."
        ),
        "recommended": "Minimal: 1  |  Moderate: 2  |  Strong: 3–5",
        "range": (1, 6),
        "step": 1,
    },
    "noise_amplitude": {
        "label": "Noise Amplitude",
        "description": "Peak amplitude of high-frequency structured noise added to the image.",
        "protection_effect": (
            "Higher amplitude creates stronger interference in the frequency bands "
            "that AI preprocessing pipelines rely on for feature extraction."
        ),
        "imperceptibility_effect": (
            "The noise is filtered to frequencies above human visual acuity. "
            "Amplitudes below 0.01 are completely invisible on typical displays. "
            "Above 0.02, a very faint shimmer may appear on solid color regions."
        ),
        "recommended": "Minimal: 0.001–0.004  |  Moderate: 0.005–0.012  |  Strong: 0.012–0.02",
        "range": (0.001, 0.03),
        "step": 0.001,
    },
    "metadata_watermark_depth": {
        "label": "Watermark Bit Depth",
        "description": "Number of least-significant bits used for the invisible watermark.",
        "protection_effect": (
            "Depth 1 (LSB only) is robust to lossless transforms. "
            "Depth 2 adds resilience to mild JPEG compression."
        ),
        "imperceptibility_effect": (
            "Depth 1 changes only bit 0 of each channel — mathematically imperceptible "
            "(max change of 1/255 per channel). Depth 2 adds bit 1 — still invisible "
            "to human vision (max change of 3/255)."
        ),
        "recommended": "Minimal: 1  |  Moderate: 1  |  Strong: 2",
        "range": (1, 2),
        "step": 1,
    },
    "quality_gate_psnr": {
        "label": "Quality Gate PSNR Threshold (dB)",
        "description": "Minimum Peak Signal-to-Noise Ratio required for the output to be accepted.",
        "protection_effect": (
            "A lower threshold allows stronger perturbations to pass through. "
            "A higher threshold forces the engine to use gentler perturbations."
        ),
        "imperceptibility_effect": (
            "48 dB ≈ 0.10% mean pixel change (extreme imperceptibility). "
            "45 dB ≈ 0.18% (professional grade). "
            "40 dB ≈ 0.32% (still very subtle). "
            "Below 40 dB, differences may become visible on close inspection."
        ),
        "recommended": "Minimal: 48  |  Moderate: 45  |  Strong: 40",
        "range": (35.0, 55.0),
        "step": 1.0,
    },
    "visual_masking_strength": {
        "label": "Visual Masking Strength",
        "description": "Minimum perturbation weight for smooth/uniform regions (0.0-1.0). Lower = more aggressive texture-based scaling.",
        "protection_effect": (
            "Lower values allocate more perturbation budget to textured regions where it's invisible, "
            "making the overall perturbation more effective per unit of distortion."
        ),
        "imperceptibility_effect": (
            "0.1–0.2: Very aggressive — smooth areas get minimal perturbation, textures get full strength. "
            "0.3: Balanced — smooth areas get 30% of full perturbation. "
            "0.5: Conservative — less differentiation between regions."
        ),
        "recommended": "Minimal: 0.5  |  Moderate: 0.3  |  Strong: 0.15",
        "range": (0.05, 0.8),
        "step": 0.05,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Preset definitions
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "label": "Minimal Visual Impact",
        "description": "Nearly invisible protection. Prioritizes visual fidelity over strength.",
        "fgsm_enabled": True,
        "fgsm_epsilon": 0.003,
        "pgd_enabled": True,
        "pgd_epsilon": 0.003,
        "pgd_steps": 5,
        "style_cloak_enabled": True,
        "style_cloak_epsilon": 0.002,
        "style_cloak_lambda": 8.0,
        "style_cloak_iterations": 20,
        "style_cloak_target": "random",
        "nightshade_enabled": True,
        "nightshade_epsilon": 0.003,
        "nightshade_targets": 1,
        "noise_enabled": True,
        "noise_amplitude": 0.002,
        "noise_patterns_enabled": ["hf_gaussian"],
        "metadata_enabled": True,
        "metadata_strip_exif": True,
        "metadata_watermark": False,
        "metadata_watermark_depth": 1,
        "output_format": "png",
        "model_name": "resnet18",
        "multi_pass": False,
        "ensemble_enabled": False,
        "ensemble_models": ["resnet18", "mobilenet"],
        "visual_masking_enabled": False,
        "visual_masking_strength": 0.5,
        "quality_gate_enabled": True,
        "quality_gate_psnr": 48.0,
        "quality_gate_ssim": 0.995,
    },
    "moderate": {
        "label": "Moderate Protection",
        "description": "Balanced protection with no visible impact. Recommended for most use cases.",
        "fgsm_enabled": True,
        "fgsm_epsilon": 0.01,
        "pgd_enabled": True,
        "pgd_epsilon": 0.01,
        "pgd_steps": 10,
        "style_cloak_enabled": True,
        "style_cloak_epsilon": 0.008,
        "style_cloak_lambda": 5.0,
        "style_cloak_iterations": 40,
        "style_cloak_target": "random",
        "nightshade_enabled": True,
        "nightshade_epsilon": 0.01,
        "nightshade_targets": 2,
        "noise_enabled": True,
        "noise_amplitude": 0.008,
        "noise_patterns_enabled": ["hf_gaussian", "moire"],
        "metadata_enabled": True,
        "metadata_strip_exif": True,
        "metadata_watermark": True,
        "metadata_watermark_depth": 1,
        "output_format": "png",
        "model_name": "resnet18",
        "multi_pass": False,
        "ensemble_enabled": False,
        "ensemble_models": ["resnet18", "mobilenet"],
        "visual_masking_enabled": True,
        "visual_masking_strength": 0.3,
        "quality_gate_enabled": True,
        "quality_gate_psnr": 45.0,
        "quality_gate_ssim": 0.99,
    },
    "strong": {
        "label": "Strong Protection",
        "description": "Maximum disruption to AI training. Very subtle grain may appear on close inspection.",
        "fgsm_enabled": True,
        "fgsm_epsilon": 0.025,
        "pgd_enabled": True,
        "pgd_epsilon": 0.025,
        "pgd_steps": 20,
        "style_cloak_enabled": True,
        "style_cloak_epsilon": 0.02,
        "style_cloak_lambda": 3.0,
        "style_cloak_iterations": 80,
        "style_cloak_target": "random",
        "nightshade_enabled": True,
        "nightshade_epsilon": 0.025,
        "nightshade_targets": 4,
        "noise_enabled": True,
        "noise_amplitude": 0.015,
        "noise_patterns_enabled": ["hf_gaussian", "moire"],
        "metadata_enabled": True,
        "metadata_strip_exif": True,
        "metadata_watermark": True,
        "metadata_watermark_depth": 2,
        "output_format": "png",
        "model_name": "resnet18",
        "multi_pass": True,
        "ensemble_enabled": True,
        "ensemble_models": ["resnet18", "mobilenet", "efficientnet"],
        "visual_masking_enabled": True,
        "visual_masking_strength": 0.15,
        "quality_gate_enabled": True,
        "quality_gate_psnr": 40.0,
        "quality_gate_ssim": 0.97,
    },
    "custom": {
        "label": "Custom",
        "description": "Full manual control over all parameters.",
        # Custom starts from moderate defaults; user overrides everything
        "fgsm_enabled": True,
        "fgsm_epsilon": 0.01,
        "pgd_enabled": True,
        "pgd_epsilon": 0.01,
        "pgd_steps": 10,
        "style_cloak_enabled": True,
        "style_cloak_epsilon": 0.008,
        "style_cloak_lambda": 5.0,
        "style_cloak_iterations": 40,
        "style_cloak_target": "random",
        "nightshade_enabled": True,
        "nightshade_epsilon": 0.01,
        "nightshade_targets": 2,
        "noise_enabled": True,
        "noise_amplitude": 0.008,
        "noise_patterns_enabled": ["hf_gaussian", "moire"],
        "metadata_enabled": True,
        "metadata_strip_exif": True,
        "metadata_watermark": True,
        "metadata_watermark_depth": 1,
        "output_format": "png",
        "model_name": "resnet18",
        "multi_pass": False,
        "ensemble_enabled": False,
        "ensemble_models": ["resnet18", "mobilenet"],
        "visual_masking_enabled": True,
        "visual_masking_strength": 0.3,
        "quality_gate_enabled": True,
        "quality_gate_psnr": 45.0,
        "quality_gate_ssim": 0.99,
    },
}


# Trade-off bar computation: parameter ranges for normalization
PARAM_RANGES: Dict[str, tuple] = {
    "fgsm_epsilon": (0.001, 0.05),
    "pgd_epsilon": (0.001, 0.05),
    "style_cloak_epsilon": (0.001, 0.04),
    "noise_amplitude": (0.001, 0.03),
    "nightshade_epsilon": (0.001, 0.05),
}


def compute_tradeoff_score(settings: Dict[str, Any]) -> float:
    """
    Compute a normalized trade-off score in [0.0, 1.0].
    0.0 = maximum invisibility, 1.0 = maximum protection.
    Used to position the marker on the trade-off indicator bar.
    """
    scores = []
    for key, (lo, hi) in PARAM_RANGES.items():
        if key in settings:
            val = settings[key]
            normalized = (val - lo) / (hi - lo) if hi > lo else 0.0
            scores.append(max(0.0, min(1.0, normalized)))

    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def tradeoff_color(score: float) -> str:
    """Return a hex color for the trade-off bar given score [0, 1]."""
    if score < 0.3:
        return "#4CAF50"  # green
    elif score < 0.6:
        return "#FF9800"  # orange
    else:
        return "#F44336"  # red
