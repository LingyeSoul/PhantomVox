import flet as ft
from ui.components.audio_progress_bar import AudioProgressBar


class AudioControlPanel(ft.Container):
    def __init__(
        self, on_play=None, on_stop=None, on_save=None, on_seek=None, has_audio=False
    ):
        self.has_audio = has_audio

        self.play_button = ft.Button(
            "播放",
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_play,
        )

        self.stop_button = ft.Button(
            "停止",
            icon=ft.Icons.STOP,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_stop,
        )

        self.save_button = ft.Button(
            "保存音频",
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_save,
        )

        self.progress_bar = AudioProgressBar(on_seek=on_seek)

        super().__init__(
            content=ft.Column(
                [
                    ft.Row(
                        [self.play_button, self.stop_button, self.save_button],
                        spacing=10,
                    ),
                    ft.Divider(height=10),
                    self.progress_bar,
                ],
                spacing=10,
            )
        )

    def _safe_update(self):
        try:
            self.update()
        except RuntimeError:
            pass

    def update_audio_state(self, has_audio: bool):
        self.has_audio = has_audio
        self.play_button.disabled = not has_audio
        self.save_button.disabled = not has_audio
        self._safe_update()

    def update_progress(self, progress: float, current: float, total: float):
        self.progress_bar.update_progress(progress, current, total)

    def set_duration(self, duration: float):
        self.progress_bar.set_duration(duration)

    def reset_progress(self):
        self.progress_bar.reset()
