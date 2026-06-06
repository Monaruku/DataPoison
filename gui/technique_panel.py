"""Technique selection cards with descriptions."""
import customtkinter as ctk


TECHNIQUE_INFO = [
    {
        "key": "fgsm",
        "name": "Adversarial Noise (FGSM)",
        "description": (
            "Adds imperceptible gradient-based noise that confuses AI classifiers "
            "and diffusion models. Strong protection against CNNs."
        ),
    },
    {
        "key": "pgd",
        "name": "Iterative Perturbation (PGD)",
        "description": (
            "Multiple small gradient steps that follow the loss curvature for stronger, "
            "more transferable perturbations. Superior cross-model protection."
        ),
    },
    {
        "key": "style_cloak",
        "name": "Style Cloak (Glaze-style)",
        "description": (
            "Shifts the AI's perception of your art style while keeping it identical "
            "to human eyes. Protects your unique artistic style from mimicry."
        ),
    },
    {
        "key": "nightshade",
        "name": "Prompt Poison (Nightshade-style)",
        "description": (
            "Causes text-to-image models to learn wrong associations. Most disruptive "
            "to training pipelines that scrape images from the web."
        ),
    },
    {
        "key": "noise",
        "name": "High-Frequency Noise",
        "description": (
            "Ultra-low-amplitude structured noise at frequencies beyond human perception "
            "but disruptive to AI preprocessing. Lightweight and fast."
        ),
    },
    {
        "key": "visual_masking",
        "name": "Adaptive Region Scaling",
        "description": (
            "Allocates more perturbation to textured areas where it's invisible and less to "
            "smooth regions. Force multiplier for all gradient-based techniques."
        ),
    },
    {
        "key": "metadata",
        "name": "Metadata Scramble",
        "description": (
            "Replaces EXIF data and embeds invisible watermarks in pixel data. "
            "Confuses data scraping pipelines. No visual change."
        ),
    },
]


class TechniquePanel(ctk.CTkScrollableFrame):
    """Scrollable list of technique selection cards."""

    def __init__(self, master, on_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self._on_change = on_change
        self._checkboxes = {}

        ctk.CTkLabel(
            self, text="Protection Techniques",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w", padx=5, pady=(5, 8))

        for tech in TECHNIQUE_INFO:
            self._create_card(tech)

    def _create_card(self, tech: dict):
        """Create a single technique selection card."""
        card = ctk.CTkFrame(self, corner_radius=8, border_width=1,
                            border_color=("gray75", "gray35"))
        card.pack(fill="x", padx=2, pady=3)

        cb = ctk.CTkCheckBox(
            card,
            text=tech["name"],
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_checkbox_change,
            onvalue=True,
            offvalue=False,
        )
        cb.select()  # Enabled by default
        cb.pack(anchor="w", padx=10, pady=(8, 0))
        self._checkboxes[tech["key"]] = cb

        desc_label = ctk.CTkLabel(
            card,
            text=tech["description"],
            font=ctk.CTkFont(size=11),
            wraplength=280,
            justify="left",
            text_color=("gray40", "gray65"),
        )
        desc_label.pack(anchor="w", padx=10, pady=(2, 8))

    def _on_checkbox_change(self):
        if self._on_change:
            self._on_change()

    def get_enabled(self) -> dict:
        """Return dict of technique_key -> bool."""
        return {key: bool(cb.get()) for key, cb in self._checkboxes.items()}

    def set_enabled(self, enabled_dict: dict):
        """Set enabled state for techniques."""
        for key, value in enabled_dict.items():
            if key in self._checkboxes:
                if value:
                    self._checkboxes[key].select()
                else:
                    self._checkboxes[key].deselect()
