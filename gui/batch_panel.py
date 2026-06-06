"""Batch processing panel with file list and progress bar."""
import os
import customtkinter as ctk
from PIL import Image


class BatchPanel(ctk.CTkFrame):
    """File list with thumbnails, add/remove buttons, and progress bar."""

    def __init__(self, master, on_files_changed=None, **kwargs):
        super().__init__(master, **kwargs)
        self._on_files_changed = on_files_changed
        self.file_paths = []

        # Top: buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)

        self.add_btn = ctk.CTkButton(
            btn_frame, text="Add Files", command=self._add_files,
            width=100, height=30, font=ctk.CTkFont(size=12)
        )
        self.add_btn.pack(side="left", padx=3)

        self.remove_btn = ctk.CTkButton(
            btn_frame, text="Remove Selected", command=self._remove_selected,
            width=120, height=30, font=ctk.CTkFont(size=12),
            fg_color="#d9534f", hover_color="#c9302c"
        )
        self.remove_btn.pack(side="left", padx=3)

        self.clear_btn = ctk.CTkButton(
            btn_frame, text="Clear All", command=self._clear_all,
            width=80, height=30, font=ctk.CTkFont(size=12),
            fg_color="gray50", hover_color="gray40"
        )
        self.clear_btn.pack(side="left", padx=3)

        self.count_label = ctk.CTkLabel(
            btn_frame, text="0 files", font=ctk.CTkFont(size=12)
        )
        self.count_label.pack(side="right", padx=5)

        # File list (scrollable)
        self.file_list = ctk.CTkScrollableFrame(self)
        self.file_list.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Progress section
        prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        prog_frame.pack(fill="x", padx=5, pady=5)

        self.progress_label = ctk.CTkLabel(
            prog_frame, text="Ready", font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(anchor="w", padx=2)

        self.progress_bar = ctk.CTkProgressBar(prog_frame, width=400)
        self.progress_bar.pack(fill="x", padx=2, pady=(2, 0))
        self.progress_bar.set(0)

        # Row widgets for tracking
        self._row_frames = {}
        self._row_labels = {}
        self._status_labels = {}

    def _add_files(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ]
        )
        for p in paths:
            if p not in self.file_paths:
                self.file_paths.append(p)
        self._refresh_list()

    def _remove_selected(self):
        # Find selected rows
        to_remove = []
        for path, cb in self._row_labels.items():
            if isinstance(cb, ctk.CTkCheckBox) and cb.get():
                to_remove.append(path)
        for p in to_remove:
            if p in self.file_paths:
                self.file_paths.remove(p)
        self._refresh_list()

    def _clear_all(self):
        self.file_paths.clear()
        self._refresh_list()

    def _refresh_list(self):
        # Clear existing rows
        for widget in self.file_list.winfo_children():
            widget.destroy()
        self._row_frames.clear()
        self._row_labels.clear()
        self._status_labels.clear()

        for path in self.file_paths:
            self._create_row(path)

        self.count_label.configure(text=f"{len(self.file_paths)} file(s)")
        if self._on_files_changed:
            self._on_files_changed()

    def _create_row(self, path: str):
        row = ctk.CTkFrame(self.file_list, corner_radius=4, height=52)
        row.pack(fill="x", padx=2, pady=1)
        self._row_frames[path] = row

        # Checkbox for selection
        cb = ctk.CTkCheckBox(
            row, text="", width=20, onvalue=True, offvalue=False
        )
        cb.pack(side="left", padx=(5, 2))
        self._row_labels[path] = cb

        # Filename label
        basename = os.path.basename(path)
        name_label = ctk.CTkLabel(
            row, text=basename, font=ctk.CTkFont(size=12),
            anchor="w", width=250
        )
        name_label.pack(side="left", padx=2)

        # Status label
        status = ctk.CTkLabel(
            row, text="Pending", font=ctk.CTkFont(size=11),
            text_color="gray50", width=80
        )
        status.pack(side="right", padx=5)
        self._status_labels[path] = status

    def update_progress(self, current: int, total: int, filename: str = ""):
        """Update progress bar and label during batch processing."""
        if total > 0:
            # Show partial progress for the file currently being processed
            # current = files completed, so add 0.5 to show mid-processing
            is_processing = "Processing" in filename or "Applying" in filename
            if is_processing and current < total:
                fraction = (current + 0.5) / total
            else:
                fraction = current / total
            fraction = max(0.0, min(1.0, fraction))
            self.progress_bar.set(fraction)
            self.progress_label.configure(
                text=f"[{current}/{total}] {filename}"
            )
        else:
            self.progress_bar.set(0)
            self.progress_label.configure(text="Ready")

    def set_file_status(self, path: str, status: str, color: str = "gray50"):
        """Update the status label for a specific file."""
        if path in self._status_labels:
            self._status_labels[path].configure(text=status, text_color=color)

    def reset_all_statuses(self):
        """Reset all file statuses to 'Pending'."""
        for path in self._status_labels:
            self._status_labels[path].configure(text="Pending", text_color="gray50")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Ready")

    def get_current_selection(self) -> str:
        """Return the first selected file path, or None."""
        for path, cb in self._row_labels.items():
            if isinstance(cb, ctk.CTkCheckBox) and cb.get():
                return path
        return self.file_paths[0] if self.file_paths else None
