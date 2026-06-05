"""Side-by-side image comparison widget with zoom and amplified difference view."""
import customtkinter as ctk
from PIL import Image, ImageChops, ImageEnhance
import numpy as np


class ImageViewer(ctk.CTkFrame):
    """Displays original and poisoned images side by side with zoom and difference view."""

    def __init__(self, master, on_open_image=None, **kwargs):
        super().__init__(master, **kwargs)
        self._on_open_image = on_open_image  # callback when user opens an image from Preview

        self.original_image = None
        self.poisoned_image = None
        self._zoom_level = 1.0
        self._show_diff = False

        # Top toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=5, pady=(5, 0))

        # Open Image button
        self.open_btn = ctk.CTkButton(
            toolbar, text="Open Image",
            command=self._handle_open_image, width=110, height=28,
            font=ctk.CTkFont(size=12)
        )
        self.open_btn.pack(side="left", padx=(0, 10))

        self.zoom_label = ctk.CTkLabel(toolbar, text="Zoom: 100%", font=ctk.CTkFont(size=12))
        self.zoom_label.pack(side="left", padx=5)

        self.zoom_slider = ctk.CTkSlider(
            toolbar, from_=0.25, to=3.0, number_of_steps=55,
            command=self._on_zoom_change, width=150
        )
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side="left", padx=5)

        self.diff_btn = ctk.CTkButton(
            toolbar, text="Show Amplified Difference",
            command=self._toggle_diff, width=180, height=28,
            font=ctk.CTkFont(size=12)
        )
        self.diff_btn.pack(side="left", padx=5)

        # Quality label
        self.quality_label = ctk.CTkLabel(
            toolbar, text="", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.quality_label.pack(side="right", padx=5)

        # Image display area
        self.display_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_columnconfigure(1, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=0)
        self.display_frame.grid_rowconfigure(1, weight=1)

        # Labels
        ctk.CTkLabel(
            self.display_frame, text="Original",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, pady=(0, 2))

        ctk.CTkLabel(
            self.display_frame, text="Protected",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=1, pady=(0, 2))

        self.left_label = ctk.CTkLabel(self.display_frame, text="No image loaded")
        self.left_label.grid(row=1, column=0, sticky="nsew", padx=2)

        self.right_label = ctk.CTkLabel(self.display_frame, text="")
        self.right_label.grid(row=1, column=1, sticky="nsew", padx=2)

        # Placeholder images
        self._left_ctk_img = None
        self._right_ctk_img = None

    def set_images(self, original: Image.Image, poisoned: Image.Image = None,
                   quality_report: dict = None):
        """Update displayed images."""
        self.original_image = original
        self.poisoned_image = poisoned
        self._show_diff = False
        self.diff_btn.configure(text="Show Amplified Difference")
        self._refresh_display()

        if quality_report:
            psnr = quality_report.get("psnr", "N/A")
            ssim = quality_report.get("ssim", "N/A")
            status = "PASSED" if quality_report.get("passed") else "WEAKENED"
            color = "#4CAF50" if quality_report.get("passed") else "#FF9800"
            self.quality_label.configure(
                text=f"PSNR: {psnr} dB | SSIM: {ssim} | {status}",
                text_color=color
            )
        else:
            self.quality_label.configure(text="")

    def _on_zoom_change(self, value):
        self._zoom_level = value
        self.zoom_label.configure(text=f"Zoom: {int(value * 100)}%")
        self._refresh_display()

    def _toggle_diff(self):
        self._show_diff = not self._show_diff
        self.diff_btn.configure(
            text="Hide Difference" if self._show_diff else "Show Amplified Difference"
        )
        self._refresh_display()

    def _make_diff_image(self, gain: float = 50.0) -> Image.Image:
        """Create an amplified difference image (50x gain by default)."""
        if not self.original_image or not self.poisoned_image:
            return None

        orig = self.original_image.convert("RGB")
        pois = self.poisoned_image.convert("RGB")

        # Resize poisoned to match original if needed
        if orig.size != pois.size:
            pois = pois.resize(orig.size, Image.LANCZOS)

        diff = ImageChops.difference(orig, pois)
        enhancer = ImageEnhance.Brightness(diff)
        amplified = enhancer.enhance(gain)
        return amplified

    def _refresh_display(self):
        """Redraw images at current zoom level."""
        if not self.original_image:
            return

        # Calculate display size based on available space
        frame_w = max(self.display_frame.winfo_width(), 400)
        half_w = frame_w // 2 - 10
        frame_h = max(self.display_frame.winfo_height() - 30, 300)

        orig = self.original_image.copy()
        display_size = (
            max(int(half_w * self._zoom_level), 50),
            max(int(frame_h * self._zoom_level), 50),
        )
        # Maintain aspect ratio
        ratio = min(display_size[0] / orig.width, display_size[1] / orig.height)
        new_size = (int(orig.width * ratio), int(orig.height * ratio))

        orig_resized = orig.resize(new_size, Image.LANCZOS)

        if self._show_diff and self.poisoned_image:
            # Show amplified difference on the right
            right_img = self._make_diff_image(gain=50.0)
            if right_img:
                right_img = right_img.resize(new_size, Image.LANCZOS)
            else:
                right_img = orig_resized
        elif self.poisoned_image:
            right_img = self.poisoned_image.copy().resize(new_size, Image.LANCZOS)
        else:
            right_img = orig_resized

        self._left_ctk_img = ctk.CTkImage(
            light_image=orig_resized, dark_image=orig_resized,
            size=new_size
        )
        self._right_ctk_img = ctk.CTkImage(
            light_image=right_img, dark_image=right_img,
            size=new_size
        )

        self.left_label.configure(image=self._left_ctk_img, text="")
        self.right_label.configure(image=self._right_ctk_img, text="")

    def clear(self):
        """Clear displayed images."""
        self.original_image = None
        self.poisoned_image = None
        self._left_ctk_img = None
        self._right_ctk_img = None
        self.left_label.configure(image=None, text="No image loaded")
        self.right_label.configure(image=None, text="")
        self.quality_label.configure(text="")

    def _handle_open_image(self):
        """Open a file dialog and load an image directly into the viewer."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ]
        )
        if path and self._on_open_image:
            self._on_open_image(path)
