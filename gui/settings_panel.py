"""Settings panel with preset selector, per-technique parameter controls, and trade-off bar."""
import customtkinter as ctk
from core.presets import PRESETS, PARAM_INFO, compute_tradeoff_score, tradeoff_color
from gui.tooltip import ToolTip, create_param_tooltip


class SettingsPanel(ctk.CTkScrollableFrame):
    """Preset selector and individual parameter controls with tooltips."""

    def __init__(self, master, on_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self._on_change = on_change
        self._controls = {}  # key -> widget
        self._tooltips = {}
        self._suppress_callback = False

        self._build_preset_selector()
        self._build_fgsm_section()
        self._build_pgd_section()
        self._build_style_cloak_section()
        self._build_nightshade_section()
        self._build_noise_section()
        self._build_visual_masking_section()
        self._build_ensemble_section()
        self._build_metadata_section()
        self._build_common_section()
        self._build_tradeoff_bar()

    # ── Preset Selector ──────────────────────────────────────────────────

    def _build_preset_selector(self):
        ctk.CTkLabel(
            self, text="Preset",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=5, pady=(5, 3))

        self.preset_var = ctk.StringVar(value="moderate")
        preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        preset_frame.pack(fill="x", padx=5, pady=(0, 8))

        presets = [
            ("minimal", "Minimal Visual Impact"),
            ("moderate", "Moderate Protection"),
            ("strong", "Strong Protection"),
            ("custom", "Custom"),
        ]
        for key, label in presets:
            rb = ctk.CTkRadioButton(
                preset_frame, text=label, variable=self.preset_var,
                value=key, command=self._on_preset_change,
                font=ctk.CTkFont(size=12),
            )
            rb.pack(anchor="w", padx=5, pady=1)

        self.preset_desc = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray65"), wraplength=300, justify="left"
        )
        self.preset_desc.pack(anchor="w", padx=10, pady=(0, 8))
        self._update_preset_desc()

    # ── FGSM Section ─────────────────────────────────────────────────────

    def _build_fgsm_section(self):
        self._section_header("FGSM Adversarial Noise")
        self._add_slider("fgsm_epsilon", "Perturbation Strength (epsilon)",
                         0.001, 0.05, 0.001, 0.01)

    # ── PGD Section ──────────────────────────────────────────────────────

    def _build_pgd_section(self):
        self._section_header("PGD Iterative Perturbation")
        self._add_slider("pgd_epsilon", "Perturbation Strength (epsilon)",
                         0.001, 0.05, 0.001, 0.01)
        self._add_slider("pgd_steps", "Gradient Steps",
                         2, 30, 1, 10)

    # ── Style Cloak Section ──────────────────────────────────────────────

    def _build_style_cloak_section(self):
        self._section_header("Style Cloak")
        self._add_slider("style_cloak_epsilon", "Perturbation Bound",
                         0.001, 0.04, 0.001, 0.008)
        self._add_slider("style_cloak_lambda", "Regularization (lambda)",
                         1.0, 10.0, 0.5, 5.0)
        self._add_slider("style_cloak_iterations", "Optimization Iterations",
                         5, 120, 5, 40)

        # Style target dropdown
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(frame, text="Style Target", font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        self._controls["style_cloak_target"] = ctk.CTkOptionMenu(
            frame, values=["random", "abstract", "impressionist"],
            command=lambda v: self._on_param_change("style_cloak_target", v),
            width=140, font=ctk.CTkFont(size=12)
        )
        self._controls["style_cloak_target"].pack(side="right")
        self._add_tooltip("style_cloak_target", self._controls["style_cloak_target"])

    # ── Nightshade Section ───────────────────────────────────────────────

    def _build_nightshade_section(self):
        self._section_header("Prompt Poison (Nightshade)")
        self._add_slider("nightshade_epsilon", "Poison Strength (epsilon)",
                         0.001, 0.05, 0.001, 0.01)
        self._add_slider("nightshade_targets", "Number of Targets",
                         1, 6, 1, 2)

        # Multi-pass toggle
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        self._controls["multi_pass"] = ctk.CTkSwitch(
            frame, text="Multi-pass", font=ctk.CTkFont(size=12),
            command=lambda: self._on_param_change("multi_pass", self._controls["multi_pass"].get())
        )
        self._controls["multi_pass"].pack(side="left", padx=5)

    # ── Noise Section ────────────────────────────────────────────────────

    def _build_noise_section(self):
        self._section_header("High-Frequency Noise")
        self._add_slider("noise_amplitude", "Noise Amplitude",
                         0.001, 0.03, 0.001, 0.008)

        # Sub-pattern checkboxes
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(frame, text="Patterns:", font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        self._controls["noise_hf"] = ctk.CTkCheckBox(
            frame, text="HF Gaussian", font=ctk.CTkFont(size=11),
            command=lambda: self._on_noise_pattern_change(), onvalue=True, offvalue=False
        )
        self._controls["noise_hf"].select()
        self._controls["noise_hf"].pack(side="left", padx=5)

        self._controls["noise_moire"] = ctk.CTkCheckBox(
            frame, text="Moire", font=ctk.CTkFont(size=11),
            command=lambda: self._on_noise_pattern_change(), onvalue=True, offvalue=False
        )
        self._controls["noise_moire"].select()
        self._controls["noise_moire"].pack(side="left", padx=5)

    # ── Visual Masking Section ──────────────────────────────────────────────

    def _build_visual_masking_section(self):
        self._section_header("Adaptive Region Scaling")
        self._add_slider("visual_masking_strength", "Min Region Weight",
                         0.05, 0.8, 0.05, 0.3)

    # ── Ensemble Section ──────────────────────────────────────────────────

    def _build_ensemble_section(self):
        self._section_header("Ensemble Models")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        self._controls["ensemble_enabled"] = ctk.CTkSwitch(
            frame, text="Enable multi-model ensemble",
            font=ctk.CTkFont(size=12),
            command=lambda: self._on_param_change(
                "ensemble_enabled",
                self._controls["ensemble_enabled"].get()
            )
        )
        self._controls["ensemble_enabled"].pack(anchor="w", padx=5, pady=1)

        # Ensemble model count selector
        ens_frame = ctk.CTkFrame(self, fg_color="transparent")
        ens_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(ens_frame, text="Models", font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        self._controls["ensemble_count"] = ctk.CTkOptionMenu(
            ens_frame, values=["2 models", "3 models"],
            command=lambda v: self._on_param_change("ensemble_count", v),
            width=120, font=ctk.CTkFont(size=12)
        )
        self._controls["ensemble_count"].pack(side="right")

    # ── Metadata Section ─────────────────────────────────────────────────

    def _build_metadata_section(self):
        self._section_header("Metadata Scramble")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        self._controls["metadata_strip_exif"] = ctk.CTkSwitch(
            frame, text="Strip original EXIF", font=ctk.CTkFont(size=12),
            command=lambda: self._on_param_change("metadata_strip_exif",
                                                   self._controls["metadata_strip_exif"].get())
        )
        self._controls["metadata_strip_exif"].select()
        self._controls["metadata_strip_exif"].pack(anchor="w", padx=5, pady=1)

        self._controls["metadata_watermark"] = ctk.CTkSwitch(
            frame, text="Embed invisible watermark", font=ctk.CTkFont(size=12),
            command=lambda: self._on_param_change("metadata_watermark",
                                                   self._controls["metadata_watermark"].get())
        )
        self._controls["metadata_watermark"].select()
        self._controls["metadata_watermark"].pack(anchor="w", padx=5, pady=1)

        wm_frame = ctk.CTkFrame(self, fg_color="transparent")
        wm_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(wm_frame, text="Watermark Depth", font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        self._controls["metadata_watermark_depth"] = ctk.CTkOptionMenu(
            wm_frame, values=["1 (LSB only)", "2 (LSB+1)"],
            command=lambda v: self._on_wm_depth_change(v),
            width=120, font=ctk.CTkFont(size=12)
        )
        self._controls["metadata_watermark_depth"].pack(side="right")
        self._add_tooltip("metadata_watermark_depth", self._controls["metadata_watermark_depth"])

    # ── Common Section ───────────────────────────────────────────────────

    def _build_common_section(self):
        self._section_header("Output Settings")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(frame, text="Format", font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        self._controls["output_format"] = ctk.CTkOptionMenu(
            frame, values=["png", "jpg", "webp"],
            command=lambda v: self._on_param_change("output_format", v),
            width=120, font=ctk.CTkFont(size=12)
        )
        self._controls["output_format"].pack(side="right")

        frame2 = ctk.CTkFrame(self, fg_color="transparent")
        frame2.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(frame2, text="Model", font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
        self._controls["model_name"] = ctk.CTkOptionMenu(
            frame2, values=["resnet18", "vgg16", "mobilenet", "efficientnet"],
            command=lambda v: self._on_param_change("model_name", v),
            width=120, font=ctk.CTkFont(size=12)
        )
        self._controls["model_name"].pack(side="right")

        # Quality gate toggle
        qg_frame = ctk.CTkFrame(self, fg_color="transparent")
        qg_frame.pack(fill="x", padx=5, pady=(4, 2))
        self._controls["quality_gate_enabled"] = ctk.CTkSwitch(
            qg_frame, text="Quality Gate (auto-reduce if visible)",
            font=ctk.CTkFont(size=12),
            command=lambda: self._on_param_change(
                "quality_gate_enabled",
                self._controls["quality_gate_enabled"].get()
            )
        )
        self._controls["quality_gate_enabled"].select()  # ON by default
        self._controls["quality_gate_enabled"].pack(anchor="w", padx=2)

    # ── Trade-off Bar ────────────────────────────────────────────────────

    def _build_tradeoff_bar(self):
        self._section_header("Protection vs Invisibility")

        self.tradeoff_frame = ctk.CTkFrame(self, corner_radius=6, height=30)
        self.tradeoff_frame.pack(fill="x", padx=5, pady=(2, 5))

        # Gradient bar (drawn with labels)
        self.bar_labels = ctk.CTkFrame(self.tradeoff_frame, fg_color="transparent")
        self.bar_labels.pack(fill="x", padx=2, pady=2)

        ctk.CTkLabel(self.bar_labels, text="Max Invisibility",
                      font=ctk.CTkFont(size=10), text_color="#4CAF50").pack(side="left")
        ctk.CTkLabel(self.bar_labels, text="Max Protection",
                      font=ctk.CTkFont(size=10), text_color="#F44336").pack(side="right")

        # Progress bar as indicator
        self.tradeoff_bar = ctk.CTkProgressBar(self.tradeoff_frame, width=280)
        self.tradeoff_bar.pack(padx=10, pady=(0, 5))
        self.tradeoff_bar.set(0.3)

        self.tradeoff_label = ctk.CTkLabel(
            self.tradeoff_frame, text="Balanced",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.tradeoff_label.pack(pady=(0, 5))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _section_header(self, title: str):
        lbl = ctk.CTkLabel(
            self, text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("gray30", "gray70"),
        )
        lbl.pack(anchor="w", padx=5, pady=(10, 3))
        sep = ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=5, pady=(0, 5))

    def _add_slider(self, key: str, label: str, from_: float, to: float,
                    step: float, default: float):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)

        lbl = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=12))
        lbl.pack(side="left", padx=2)

        val_label = ctk.CTkLabel(
            frame, text=f"{default:.3f}" if step < 1 else f"{int(default)}",
            font=ctk.CTkFont(size=12, weight="bold"), width=55
        )
        val_label.pack(side="right", padx=2)

        steps = int((to - from_) / step)
        slider = ctk.CTkSlider(
            frame, from_=from_, to=to, number_of_steps=max(steps, 1),
            command=lambda v, k=key, vl=val_label, s=step: self._on_slider_change(k, v, vl, s),
            width=120
        )
        slider.set(default)
        slider.pack(side="right", padx=2)

        self._controls[key] = slider
        self._controls[f"{key}_label"] = val_label
        self._add_tooltip(key, lbl)

    def _add_tooltip(self, param_key: str, widget):
        text = create_param_tooltip(param_key, PARAM_INFO)
        if text:
            self._tooltips[param_key] = ToolTip(widget, text)

    # ── Event Handlers ───────────────────────────────────────────────────

    def _on_slider_change(self, key: str, value, val_label, step: float):
        if step < 1:
            val_label.configure(text=f"{value:.3f}")
        else:
            val_label.configure(text=f"{int(value)}")
        self._switch_to_custom()
        self._update_tradeoff()
        self._fire_change()

    def _on_param_change(self, key: str, value):
        self._switch_to_custom()
        self._update_tradeoff()
        self._fire_change()

    def _on_noise_pattern_change(self):
        self._switch_to_custom()
        self._fire_change()

    def _on_wm_depth_change(self, value: str):
        self._switch_to_custom()
        self._fire_change()

    def _on_preset_change(self):
        preset_name = self.preset_var.get()
        self._update_preset_desc()
        self._apply_preset_to_controls(preset_name)
        self._update_tradeoff()
        self._fire_change()

    def _switch_to_custom(self):
        """Auto-switch to Custom preset when any parameter is manually changed."""
        if self.preset_var.get() != "custom":
            self._suppress_callback = True
            self.preset_var.set("custom")
            self._update_preset_desc()
            self._suppress_callback = False

    def _update_preset_desc(self):
        preset_name = self.preset_var.get()
        info = PRESETS.get(preset_name, {})
        self.preset_desc.configure(text=info.get("description", ""))

    def _apply_preset_to_controls(self, preset_name: str):
        """Update all controls to match a preset's values."""
        self._suppress_callback = True
        preset = PRESETS.get(preset_name, PRESETS["moderate"])

        slider_map = {
            "fgsm_epsilon": preset.get("fgsm_epsilon", 0.01),
            "pgd_epsilon": preset.get("pgd_epsilon", 0.01),
            "pgd_steps": preset.get("pgd_steps", 10),
            "style_cloak_epsilon": preset.get("style_cloak_epsilon", 0.008),
            "style_cloak_lambda": preset.get("style_cloak_lambda", 5.0),
            "style_cloak_iterations": preset.get("style_cloak_iterations", 40),
            "nightshade_epsilon": preset.get("nightshade_epsilon", 0.01),
            "nightshade_targets": preset.get("nightshade_targets", 2),
            "noise_amplitude": preset.get("noise_amplitude", 0.008),
            "visual_masking_strength": preset.get("visual_masking_strength", 0.3),
        }

        for key, value in slider_map.items():
            if key in self._controls:
                self._controls[key].set(value)
                lbl_key = f"{key}_label"
                if lbl_key in self._controls:
                    step = PARAM_INFO.get(key, {}).get("step", 1)
                    if step < 1:
                        self._controls[lbl_key].configure(text=f"{value:.3f}")
                    else:
                        self._controls[lbl_key].configure(text=f"{int(value)}")

        if "style_cloak_target" in self._controls:
            self._controls["style_cloak_target"].set(preset.get("style_cloak_target", "random"))

        if "multi_pass" in self._controls:
            if preset.get("multi_pass", False):
                self._controls["multi_pass"].select()
            else:
                self._controls["multi_pass"].deselect()

        patterns = preset.get("noise_patterns_enabled", ["hf_gaussian", "moire"])
        if "noise_hf" in self._controls:
            if "hf_gaussian" in patterns:
                self._controls["noise_hf"].select()
            else:
                self._controls["noise_hf"].deselect()
        if "noise_moire" in self._controls:
            if "moire" in patterns:
                self._controls["noise_moire"].select()
            else:
                self._controls["noise_moire"].deselect()

        if "metadata_strip_exif" in self._controls:
            if preset.get("metadata_strip_exif", True):
                self._controls["metadata_strip_exif"].select()
            else:
                self._controls["metadata_strip_exif"].deselect()

        if "metadata_watermark" in self._controls:
            if preset.get("metadata_watermark", True):
                self._controls["metadata_watermark"].select()
            else:
                self._controls["metadata_watermark"].deselect()

        depth = preset.get("metadata_watermark_depth", 1)
        if "metadata_watermark_depth" in self._controls:
            self._controls["metadata_watermark_depth"].set(
                "1 (LSB only)" if depth == 1 else "2 (LSB+1)"
            )

        if "output_format" in self._controls:
            self._controls["output_format"].set(preset.get("output_format", "png"))
        if "model_name" in self._controls:
            self._controls["model_name"].set(preset.get("model_name", "resnet18"))

        if "quality_gate_enabled" in self._controls:
            if preset.get("quality_gate_enabled", True):
                self._controls["quality_gate_enabled"].select()
            else:
                self._controls["quality_gate_enabled"].deselect()

        if "ensemble_enabled" in self._controls:
            if preset.get("ensemble_enabled", False):
                self._controls["ensemble_enabled"].select()
            else:
                self._controls["ensemble_enabled"].deselect()

        ens_models = preset.get("ensemble_models", ["resnet18", "mobilenet"])
        if "ensemble_count" in self._controls:
            self._controls["ensemble_count"].set(
                "3 models" if len(ens_models) >= 3 else "2 models"
            )

        self._suppress_callback = False

    def _update_tradeoff(self):
        """Update the trade-off indicator bar."""
        settings = self.get_settings_dict()
        score = compute_tradeoff_score(settings)
        self.tradeoff_bar.set(score)

        color = tradeoff_color(score)
        if score < 0.3:
            label = "Maximum Invisibility"
        elif score < 0.6:
            label = "Balanced"
        else:
            label = "Maximum Protection"
        self.tradeoff_label.configure(text=label, text_color=color)

    def _fire_change(self):
        if not self._suppress_callback and self._on_change:
            self._on_change()

    # ── Public API ───────────────────────────────────────────────────────

    def get_settings_dict(self) -> dict:
        """Collect all current parameter values as a dict."""
        d = {}
        slider_keys = [
            "fgsm_epsilon", "pgd_epsilon", "pgd_steps",
            "style_cloak_epsilon", "style_cloak_lambda",
            "style_cloak_iterations", "nightshade_epsilon", "nightshade_targets",
            "noise_amplitude", "visual_masking_strength",
        ]
        for key in slider_keys:
            if key in self._controls:
                d[key] = self._controls[key].get()

        if "style_cloak_target" in self._controls:
            d["style_cloak_target"] = self._controls["style_cloak_target"].get()
        if "multi_pass" in self._controls:
            d["multi_pass"] = bool(self._controls["multi_pass"].get())

        # Noise patterns
        patterns = []
        if self._controls.get("noise_hf") and self._controls["noise_hf"].get():
            patterns.append("hf_gaussian")
        if self._controls.get("noise_moire") and self._controls["noise_moire"].get():
            patterns.append("moire")
        d["noise_patterns_enabled"] = patterns

        # Metadata
        if "metadata_strip_exif" in self._controls:
            d["metadata_strip_exif"] = bool(self._controls["metadata_strip_exif"].get())
        if "metadata_watermark" in self._controls:
            d["metadata_watermark"] = bool(self._controls["metadata_watermark"].get())
        if "metadata_watermark_depth" in self._controls:
            depth_str = self._controls["metadata_watermark_depth"].get()
            d["metadata_watermark_depth"] = 1 if "LSB only" in depth_str else 2

        if "output_format" in self._controls:
            d["output_format"] = self._controls["output_format"].get()
        if "model_name" in self._controls:
            d["model_name"] = self._controls["model_name"].get()
        if "quality_gate_enabled" in self._controls:
            d["quality_gate_enabled"] = bool(self._controls["quality_gate_enabled"].get())

        # Ensemble settings
        if "ensemble_enabled" in self._controls:
            d["ensemble_enabled"] = bool(self._controls["ensemble_enabled"].get())
        if "ensemble_count" in self._controls:
            count_str = self._controls["ensemble_count"].get()
            count = 3 if "3" in count_str else 2
            if count >= 3:
                d["ensemble_models"] = ["resnet18", "mobilenet", "efficientnet"]
            else:
                d["ensemble_models"] = ["resnet18", "mobilenet"]

        # Visual masking is read from the technique panel toggle
        # (visual_masking_enabled is set from technique_panel.get_enabled())

        d["preset"] = self.preset_var.get()
        return d

    def get_preset_name(self) -> str:
        return self.preset_var.get()
