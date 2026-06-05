"""Main application window — orchestrates all GUI components and the poisoning engine."""
import os
import queue
import threading
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from core.poison_engine import PoisonEngine, PoisonSettings, PoisonResult
from core.presets import PRESETS
from core.utils import save_image, validate_image
from gui.image_viewer import ImageViewer
from gui.technique_panel import TechniquePanel
from gui.settings_panel import SettingsPanel
from gui.batch_panel import BatchPanel
from gui.progress_dialog import ProgressDialog


class App(ctk.CTk):
    """Main DataPoison application window."""

    def __init__(self):
        super().__init__()

        self.title("DataPoison — Protect Your Art From AI Training")
        self.geometry("1200x800")
        self.minsize(950, 620)

        # State
        self.engine = None  # Lazy-initialized
        self.results = {}   # path -> PoisonResult
        self._result_queue = queue.Queue()
        self._cancel_event = threading.Event()

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # Configure grid: left sidebar + right content
        self.grid_columnconfigure(0, weight=0, minsize=340)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content_area()
        self._build_status_bar()

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        sidebar.grid_propagate(False)

        # Title
        title_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            title_frame, text="DataPoison",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_frame, text="Protect your art from AI training",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray65")
        ).pack(anchor="w", pady=(0, 5))

        # Technique panel
        self.technique_panel = TechniquePanel(
            sidebar,
            on_change=self._on_technique_change,
            width=310,
        )
        self.technique_panel.pack(fill="x", padx=10, pady=(5, 2))

        # Settings panel
        self.settings_panel = SettingsPanel(
            sidebar,
            on_change=self._on_settings_change,
            width=310,
        )
        self.settings_panel.pack(fill="both", expand=True, padx=10, pady=(2, 2))

        # Action buttons
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(5, 10))

        self.poison_btn = ctk.CTkButton(
            btn_frame, text="Poison Selected Image",
            command=self._poison_current,
            height=38, font=ctk.CTkFont(size=13, weight="bold")
        )
        self.poison_btn.pack(fill="x", pady=(0, 5))

        self.poison_all_btn = ctk.CTkButton(
            btn_frame, text="Poison All Images",
            command=self._poison_all,
            height=38, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#28a745", hover_color="#218838"
        )
        self.poison_all_btn.pack(fill="x", pady=(0, 5))

        self.save_btn = ctk.CTkButton(
            btn_frame, text="Save Protected Images...",
            command=self._save_results,
            height=34, font=ctk.CTkFont(size=12),
            fg_color="gray50", hover_color="gray40"
        )
        self.save_btn.pack(fill="x")

    def _build_content_area(self):
        content = ctk.CTkFrame(self, corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        # Tab view
        self.tabview = ctk.CTkTabview(content)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Preview tab
        preview_tab = self.tabview.add("Preview")
        preview_tab.grid_columnconfigure(0, weight=1)
        preview_tab.grid_rowconfigure(0, weight=1)
        self.image_viewer = ImageViewer(preview_tab, on_open_image=self._on_preview_open_image)
        self.image_viewer.grid(row=0, column=0, sticky="nsew")

        # Batch tab
        batch_tab = self.tabview.add("Batch")
        batch_tab.grid_columnconfigure(0, weight=1)
        batch_tab.grid_rowconfigure(0, weight=1)
        self.batch_panel = BatchPanel(batch_tab, on_files_changed=self._on_files_changed)
        self.batch_panel.grid(row=0, column=0, sticky="nsew")

        # Report tab
        report_tab = self.tabview.add("Report")
        report_tab.grid_columnconfigure(0, weight=1)
        report_tab.grid_rowconfigure(0, weight=1)
        self.report_text = ctk.CTkTextbox(
            report_tab, font=ctk.CTkFont(size=13), wrap="word"
        )
        self.report_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.report_text.insert("1.0", "Protection report will appear here after processing.\n\n"
                                        "Process images to see details about applied techniques,\n"
                                        "quality metrics (PSNR, SSIM), and any warnings.")

    def _build_status_bar(self):
        # Detect compute device
        from models.model_loader import get_device
        device = get_device()
        if device.type == "cuda":
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            device_text = f"GPU: {gpu_name}"
            device_color = "#4CAF50"
        else:
            device_text = "CPU (install CUDA PyTorch for GPU acceleration)"
            device_color = "#FF9800"

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))

        self.status_label = ctk.CTkLabel(
            status_frame, text="Ready \u2014 Add images and select a preset to begin.",
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
            anchor="w"
        )
        self.status_label.pack(side="left", fill="x", expand=True)

        self.device_label = ctk.CTkLabel(
            status_frame, text=device_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=device_color,
            anchor="e"
        )
        self.device_label.pack(side="right", padx=(10, 0))

    # ── Settings Collection ──────────────────────────────────────────────

    def _collect_settings(self) -> PoisonSettings:
        """Build PoisonSettings from current GUI state."""
        preset_name = self.settings_panel.get_preset_name()
        if preset_name != "custom":
            settings = PoisonSettings.from_preset(preset_name)
            # Override with technique panel enable/disable
            enabled = self.technique_panel.get_enabled()
            settings.fgsm_enabled = enabled.get("fgsm", True)
            settings.style_cloak_enabled = enabled.get("style_cloak", True)
            settings.nightshade_enabled = enabled.get("nightshade", True)
            settings.noise_enabled = enabled.get("noise", True)
            settings.metadata_enabled = enabled.get("metadata", True)
            return settings

        # Custom: read from settings panel
        params = self.settings_panel.get_settings_dict()
        enabled = self.technique_panel.get_enabled()

        noise_patterns = params.get("noise_patterns_enabled", ["hf_gaussian", "moire"])

        settings = PoisonSettings(
            preset="custom",
            fgsm_enabled=enabled.get("fgsm", True),
            fgsm_epsilon=params.get("fgsm_epsilon", 0.01),
            style_cloak_enabled=enabled.get("style_cloak", True),
            style_cloak_epsilon=params.get("style_cloak_epsilon", 0.008),
            style_cloak_lambda=params.get("style_cloak_lambda", 5.0),
            style_cloak_iterations=int(params.get("style_cloak_iterations", 40)),
            style_cloak_target=params.get("style_cloak_target", "random"),
            nightshade_enabled=enabled.get("nightshade", True),
            nightshade_epsilon=params.get("nightshade_epsilon", 0.01),
            nightshade_targets=int(params.get("nightshade_targets", 2)),
            noise_enabled=enabled.get("noise", True),
            noise_amplitude=params.get("noise_amplitude", 0.008),
            noise_patterns_enabled=noise_patterns,
            metadata_enabled=enabled.get("metadata", True),
            metadata_strip_exif=params.get("metadata_strip_exif", True),
            metadata_watermark=params.get("metadata_watermark", True),
            metadata_watermark_depth=params.get("metadata_watermark_depth", 1),
            output_format=params.get("output_format", "png"),
            model_name=params.get("model_name", "resnet18"),
            multi_pass=params.get("multi_pass", False),
            quality_gate_enabled=params.get("quality_gate_enabled", True),
            quality_gate_psnr=PRESETS.get(preset_name, PRESETS["moderate"]).get("quality_gate_psnr", 45.0),
            quality_gate_ssim=PRESETS.get(preset_name, PRESETS["moderate"]).get("quality_gate_ssim", 0.99),
        )
        return settings

    # ── Event Handlers ───────────────────────────────────────────────────

    def _on_preview_open_image(self, path: str):
        """Handle opening an image directly from the Preview tab."""
        if not path:
            return
        from core.utils import validate_image
        if not validate_image(path):
            messagebox.showerror("Invalid Image", f"Cannot open image:\n{path}")
            return

        # Add to batch panel if not already there
        if path not in self.batch_panel.file_paths:
            self.batch_panel.file_paths.append(path)
            self.batch_panel._refresh_list()

        # Show the image in the preview
        try:
            original = Image.open(path).convert("RGB")
            self.image_viewer.set_images(original)
            self.status_label.configure(text=f"Loaded: {os.path.basename(path)} — click 'Poison Selected Image' to protect it.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def _on_technique_change(self):
        pass  # Techniques will be read at processing time

    def _on_settings_change(self):
        pass  # Settings will be read at processing time

    def _on_files_changed(self):
        count = len(self.batch_panel.file_paths)
        self.status_label.configure(text=f"{count} image(s) loaded. Select a preset and click Poison.")

    # ── Processing ───────────────────────────────────────────────────────

    def _ensure_engine(self):
        if self.engine is None:
            self.status_label.configure(text="Loading AI model (first time may download ~45 MB)...")
            self.update_idletasks()
            self.engine = PoisonEngine()

    def _poison_current(self):
        """Process the currently selected/first image."""
        path = self.batch_panel.get_current_selection()
        if not path:
            messagebox.showinfo("No Image", "Add images first using the Batch tab.")
            return

        if not validate_image(path):
            messagebox.showerror("Invalid Image", f"Cannot open image:\n{path}")
            return

        self._ensure_engine()
        settings = self._collect_settings()

        # Show the original image
        original = Image.open(path).convert("RGB")
        self.image_viewer.set_images(original)
        self.tabview.set("Preview")

        # Process in background
        self._cancel_event.clear()
        self.poison_btn.configure(state="disabled", text="Processing...")
        self.poison_all_btn.configure(state="disabled")

        self.status_label.configure(text=f"Processing: {os.path.basename(path)}...")

        thread = threading.Thread(
            target=self._process_single,
            args=(path, original, settings),
            daemon=True
        )
        thread.start()
        self._poll_result_queue()

    def _poison_all(self):
        """Process all images in the batch."""
        paths = self.batch_panel.file_paths
        if not paths:
            messagebox.showinfo("No Images", "Add images first using the Batch tab.")
            return

        self._ensure_engine()
        settings = self._collect_settings()
        self._cancel_event.clear()
        self.batch_panel.reset_all_statuses()

        self.poison_btn.configure(state="disabled")
        self.poison_all_btn.configure(state="disabled", text="Processing...")
        self.save_btn.configure(state="disabled")

        thread = threading.Thread(
            target=self._process_batch,
            args=(paths, settings),
            daemon=True
        )
        thread.start()
        self._poll_result_queue()

    def _process_single(self, path: str, original: Image.Image, settings: PoisonSettings):
        """Worker: process a single image."""
        try:
            result = self.engine.process_image(original, settings)
            self._result_queue.put(("single_done", path, result))
        except Exception as e:
            self._result_queue.put(("error", path, str(e)))

    def _process_batch(self, paths: list, settings: PoisonSettings):
        """Worker: process all images in batch."""
        total = len(paths)
        for i, path in enumerate(paths):
            if self._cancel_event.is_set():
                self._result_queue.put(("cancelled", path, None))
                break

            if not validate_image(path):
                self._result_queue.put(("skip_error", path, "Invalid image file"))
                continue

            # Mark file as "Processing" and update progress bar
            self._result_queue.put(("file_processing", path, None))
            self._result_queue.put(("progress", i, f"{os.path.basename(path)}"))

            try:
                original = Image.open(path).convert("RGB")
                result = self.engine.process_image(original, settings)
                self._result_queue.put(("batch_done", path, result))
                self._result_queue.put(("progress", i + 1, f"{os.path.basename(path)}"))
            except Exception as e:
                self._result_queue.put(("error", path, str(e)))

        self._result_queue.put(("batch_complete", total, None))

    def _poll_result_queue(self):
        """Poll the result queue and update GUI on the main thread."""
        try:
            while True:
                msg_type, data1, data2 = self._result_queue.get_nowait()

                if msg_type == "single_done":
                    path, result = data1, data2
                    self.results[path] = result
                    self.image_viewer.set_images(
                        result.original, result.poisoned, result.quality_report
                    )
                    self.status_label.configure(
                        text=f"Done! PSNR: {result.quality_report['psnr']} dB | "
                             f"SSIM: {result.quality_report['ssim']} | "
                             f"Time: {result.processing_time_ms:.0f}ms"
                    )
                    self.poison_btn.configure(state="normal", text="Poison Selected Image")
                    self.poison_all_btn.configure(state="normal")
                    self._update_report(path, result)

                elif msg_type == "file_processing":
                    path = data1
                    self.batch_panel.set_file_status(path, "Processing...", "#FF9800")

                elif msg_type == "batch_done":
                    path, result = data1, data2
                    self.results[path] = result
                    self.batch_panel.set_file_status(path, "Done", "#4CAF50")

                elif msg_type == "progress":
                    idx, filename = data1, data2
                    total = len(self.batch_panel.file_paths)
                    self.batch_panel.update_progress(idx, total, filename)

                elif msg_type == "error":
                    path, error_msg = data1, data2
                    self.batch_panel.set_file_status(path, "Error", "#F44336")
                    self.status_label.configure(text=f"Error processing {os.path.basename(path)}: {error_msg}")

                elif msg_type == "skip_error":
                    path, reason = data1, data2
                    self.batch_panel.set_file_status(path, "Skipped", "#FF9800")

                elif msg_type == "cancelled":
                    self.status_label.configure(text="Processing cancelled.")
                    self.batch_panel.set_file_status(data1, "Cancelled", "#FF9800")

                elif msg_type == "batch_complete":
                    total = data1 or len(self.batch_panel.file_paths)
                    self.status_label.configure(
                        text=f"Batch complete! {len(self.results)} image(s) processed."
                    )
                    self.batch_panel.update_progress(total, total, "Complete")
                    self.poison_btn.configure(state="normal")
                    self.poison_all_btn.configure(state="normal", text="Poison All Images")
                    self.save_btn.configure(state="normal")
                    self._update_batch_report()

        except queue.Empty:
            pass

        # Keep polling if there are pending results
        if not self._result_queue.empty() or (
            self.poison_btn.cget("state") == "disabled"
        ):
            self.after(100, self._poll_result_queue)

    # ── Save ─────────────────────────────────────────────────────────────

    def _save_results(self):
        """Save all processed images."""
        if not self.results:
            messagebox.showinfo("No Results", "Process images first before saving.")
            return

        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return

        settings = self._collect_settings()
        fmt = settings.output_format

        saved = 0
        for path, result in self.results.items():
            basename = os.path.splitext(os.path.basename(path))[0]
            ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}.get(fmt, ".png")
            out_path = os.path.join(output_dir, f"{basename}_protected{ext}")
            save_image(result.poisoned, out_path, fmt)
            saved += 1

        messagebox.showinfo("Saved", f"Successfully saved {saved} protected image(s) to:\n{output_dir}")
        self.status_label.configure(text=f"Saved {saved} image(s) to {output_dir}")

    # ── Report ───────────────────────────────────────────────────────────

    def _update_report(self, path: str, result: PoisonResult):
        """Update the report tab with details for a single image."""
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", f"Protection Report\n")
        self.report_text.insert("end", f"{'='*50}\n\n")
        self.report_text.insert("end", f"File: {os.path.basename(path)}\n")
        self.report_text.insert("end", f"Preset: {result.settings_used.preset}\n")
        self.report_text.insert("end", f"Processing time: {result.processing_time_ms:.0f} ms\n\n")

        self.report_text.insert("end", f"Techniques Applied:\n")
        for tech in result.techniques_applied:
            self.report_text.insert("end", f"  - {tech}\n")

        self.report_text.insert("end", f"\nQuality Metrics:\n")
        qr = result.quality_report
        self.report_text.insert("end", f"  PSNR: {qr['psnr']} dB (threshold: {qr['psnr_threshold']} dB)\n")
        self.report_text.insert("end", f"  SSIM: {qr['ssim']} (threshold: {qr['ssim_threshold']})\n")
        self.report_text.insert("end", f"  Max pixel delta: {qr['max_pixel_delta']}/255\n")
        self.report_text.insert("end", f"  Quality gate: {'PASSED' if qr['passed'] else 'WEAKENED'}\n")

        if result.retries > 0:
            self.report_text.insert("end", f"\n  Retries: {result.retries} (parameters auto-reduced)\n")

        if result.warnings:
            self.report_text.insert("end", f"\nWarnings:\n")
            for w in result.warnings:
                self.report_text.insert("end", f"  - {w}\n")

    def _update_batch_report(self):
        """Update the report tab with a summary of all batch results."""
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", f"Batch Protection Report\n")
        self.report_text.insert("end", f"{'='*50}\n\n")
        self.report_text.insert("end", f"Total images processed: {len(self.results)}\n\n")

        for path, result in self.results.items():
            qr = result.quality_report
            status = "PASSED" if qr["passed"] else "WEAKENED"
            self.report_text.insert("end", f"{os.path.basename(path)}:\n")
            self.report_text.insert("end", f"  PSNR: {qr['psnr']} dB | SSIM: {qr['ssim']} | {status}\n")
            self.report_text.insert("end", f"  Techniques: {', '.join(result.techniques_applied)}\n\n")

        self.tabview.set("Report")
