import flet as ft


class AudioProgressBar(ft.Container):
    def __init__(self, on_seek=None):
        self.progress_slider = ft.Slider(
            min=0.0,
            max=1.0,
            value=0.0,
            divisions=1000,
            on_change_end=on_seek,
            disabled=True,
            active_color=ft.Colors.BLUE,
            label="{value}%",
        )

        self.time_display = ft.Text(
            "0:00 / 0:00",
            size=12,
            color=ft.Colors.GREY_400,
            text_align=ft.TextAlign.CENTER,
        )

        super().__init__(
            content=ft.Column(
                [
                    self.progress_slider,
                    ft.Row(
                        [
                            ft.Container(),
                            self.time_display,
                            ft.Container(),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                ],
                spacing=5,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )

    def _safe_update(self):
        try:
            self.update()
        except RuntimeError:
            pass

    def update_progress(self, progress: float, current: float, total: float):
        self.progress_slider.value = progress
        self.time_display.value = (
            f"{self._format_time(current)} / {self._format_time(total)}"
        )
        self._safe_update()

    def set_duration(self, duration: float):
        self.progress_slider.disabled = False
        self.update_progress(0.0, 0.0, duration)

    def reset(self):
        self.progress_slider.value = 0.0
        self.progress_slider.disabled = True
        self.time_display.value = "0:00 / 0:00"
        self._safe_update()

    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
