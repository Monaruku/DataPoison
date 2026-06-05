"""Reusable tooltip widget for customtkinter controls."""
import customtkinter as ctk


class ToolTip:
    """
    A tooltip that appears when hovering over a widget.
    Displays multi-line text with formatting.
    """

    def __init__(self, widget, text: str = "", delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip_window = None
        self._after_id = None

        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def set_text(self, text: str):
        """Update tooltip text."""
        self.text = text

    def _on_enter(self, event=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip_window or not self.text:
            return

        # Position tooltip below the widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self._tip_window = ctk.CTkToplevel(self.widget)
        self._tip_window.wm_overrideredirect(True)
        self._tip_window.wm_geometry(f"+{x}+{y}")
        self._tip_window.attributes("-topmost", True)

        # Make semi-transparent on Windows
        try:
            self._tip_window.attributes("-alpha", 0.95)
        except Exception:
            pass

        frame = ctk.CTkFrame(
            self._tip_window,
            corner_radius=6,
            fg_color=("gray85", "gray20"),
            border_width=1,
            border_color=("gray70", "gray40"),
        )
        frame.pack(padx=1, pady=1)

        label = ctk.CTkLabel(
            frame,
            text=self.text,
            wraplength=350,
            justify="left",
            font=ctk.CTkFont(size=12),
            padx=10,
            pady=6,
        )
        label.pack()

    def _hide(self):
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


def create_param_tooltip(param_key: str, param_info: dict) -> str:
    """Build a formatted tooltip string from parameter metadata."""
    info = param_info.get(param_key, {})
    if not info:
        return ""

    parts = [
        f"  {info.get('label', param_key)}",
        "",
        f"  {info.get('description', '')}",
        "",
        f"  Protection: {info.get('protection_effect', '')}",
        "",
        f"  Visual impact: {info.get('imperceptibility_effect', '')}",
        "",
        f"  {info.get('recommended', '')}",
    ]
    return "\n".join(parts)
