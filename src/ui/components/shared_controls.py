"""
共享 UI 组件

提供通用的文本输入、音频控制等组件
"""

import flet as ft
import logging

from ui.components.audio_progress_bar import AudioProgressBar

logger = logging.getLogger(__name__)


class TextPanel(ft.Container):
    """通用文本输入面板"""

    def __init__(
        self,
        placeholder="请输入文本...",
        min_lines=12,
        max_lines=20,
        on_change=None,
        on_clear=None
    ):
        # 注意：placeholder 参数不再使用，但保留以保持向后兼容
        self.text_input = ft.TextField(
            multiline=True,
            min_lines=min_lines,
            max_lines=max_lines,
            border_radius=8,
            autofocus=False,
            expand=True,
            text_style=ft.TextStyle(
                font_family="Microsoft YaHei",
                size=14
            ),
            on_change=on_change
        )

        self.clear_button = ft.IconButton(
            icon=ft.Icons.CLEAR,
            icon_color=ft.Colors.GREY_400,
            tooltip="清空文本",
            on_click=on_clear or self._on_clear_default
        )

        super().__init__(
            content=ft.Column(
                [
                    ft.Container(
                        content=self.text_input,
                        padding=10,
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                        border_radius=8,
                        height=150,
                    )
                ],
                spacing=10
            ),
            expand=True
        )

    def _on_clear_default(self, e):
        """默认清空处理"""
        self.text_input.value = ""
        self.text_input.update()

    def get_text(self) -> str:
        """获取输入的文本"""
        return self.text_input.value or ""

    def set_text(self, text: str):
        """设置文本"""
        self.text_input.value = text
        self.text_input.update()

    def clear(self):
        """清空文本"""
        self._on_clear_default(None)


class AudioControlPanel(ft.Container):
    """通用音频控制面板"""

    def __init__(
        self,
        on_play=None,
        on_stop=None,
        on_save=None,
        on_seek=None,
        has_audio=False
    ):
        self.has_audio = has_audio

        # 播放按钮
        self.play_button = ft.Button(
            "播放",
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_play
        )

        # 停止按钮
        self.stop_button = ft.Button(
            "停止",
            icon=ft.Icons.STOP,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_stop
        )

        # 保存按钮
        self.save_button = ft.Button(
            "保存音频",
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_save
        )

        # 进度条
        self.progress_bar = AudioProgressBar(on_seek=on_seek)

        super().__init__(
            content=ft.Column(
                [
                    ft.Row([self.play_button, self.stop_button, self.save_button], spacing=10),
                    ft.Divider(height=10),
                    self.progress_bar
                ],
                spacing=10
            )
        )

    def update_audio_state(self, has_audio: bool):
        """更新音频状态"""
        self.has_audio = has_audio
        self.play_button.disabled = not has_audio
        self.save_button.disabled = not has_audio
        self.update()

    def update_progress(self, progress: float, current: float, total: float):
        """更新播放进度"""
        self.progress_bar.update_progress(progress, current, total)

    def set_duration(self, duration: float):
        """设置音频时长并启用进度条"""
        self.progress_bar.set_duration(duration)

    def reset_progress(self):
        """重置进度条"""
        self.progress_bar.reset()


class ParameterSliders(ft.Container):
    """通用参数滑块面板"""

    def __init__(self, on_speed_change=None, on_pitch_change=None):
        # 语速滑块
        self.speed_slider = ft.Slider(
            min=0.5,
            max=2.0,
            value=1.0,
            divisions=15,
            label="语速: {value}x",
            on_change=on_speed_change
        )

        # 音调滑块
        self.pitch_slider = ft.Slider(
            min=0.5,
            max=2.0,
            value=1.0,
            divisions=15,
            label="音调: {value}x",
            on_change=on_pitch_change
        )

        super().__init__(
            content=ft.Column(
                [
                    ft.Text("参数调节", size=14, weight=ft.FontWeight.BOLD),
                    ft.Column(
                        [
                            ft.Text("语速", size=12),
                            self.speed_slider,
                        ],
                        spacing=5
                    ),
                    ft.Column(
                        [
                            ft.Text("音调", size=12),
                            self.pitch_slider,
                        ],
                        spacing=5
                    )
                ],
                spacing=15
            )
        )

    def get_speed(self) -> float:
        """获取语速"""
        return self.speed_slider.value

    def get_pitch(self) -> float:
        """获取音调"""
        return self.pitch_slider.value

    def set_speed(self, value: float):
        """设置语速"""
        self.speed_slider.value = value
        self.speed_slider.update()

    def set_pitch(self, value: float):
        """设置音调"""
        self.pitch_slider.value = value
        self.pitch_slider.update()
