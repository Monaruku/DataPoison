"""Modal progress dialog for long-running operations."""
import customtkinter as ctk


class ProgressDialog(ctk.CTkToplevel):
    """Modal dialog showing progress during image processing."""

    def __init__(self, master, title: str = "Processing...", cancellable: bool = True):
        super().__init__(master)
        self.title(title)
        self.geometry("400x180")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self._cancelled = False

        # Center on parent
        self.update_idletasks()
        px = master.winfo_x() + (master.winfo_width() - 400) // 2
        py = master.winfo_y() + (master.winfo_height() - 180) // 2
        self.geometry(f"+{px}+{py}")

        # Content
        self.message_label = ctk.CTkLabel(
            self, text="Processing images...",
            font=ctk.CTkFont(size=14)
        )
        self.message_label.pack(pady=(20, 10))

        self.progress_bar = ctk.CTkProgressBar(self, width=350)
        self.progress_bar.pack(padx=25, pady=5)
        self.progress_bar.set(0)

        self.detail_label = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray65")
        )
        self.detail_label.pack(pady=5)

        if cancellable:
            self.cancel_btn = ctk.CTkButton(
                self, text="Cancel", command=self._cancel,
                width=100, height=30, fg_color="#d9534f", hover_color="#c9302c"
            )
            self.cancel_btn.pack(pady=10)

    def update_progress(self, fraction: float, message: str = "", detail: str = ""):
        """Update progress bar and messages. fraction in [0, 1]."""
        self.progress_bar.set(max(0.0, min(1.0, fraction)))
        if message:
            self.message_label.configure(text=message)
        if detail:
            self.detail_label.configure(text=detail)
        self.update_idletasks()

    def _cancel(self):
        self._cancelled = True
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.configure(text="Cancelling...", state="disabled")

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def close(self):
        """Close the dialog."""
        try:
            self.grab_release()
            self.destroy()
        except Exception:
            pass
