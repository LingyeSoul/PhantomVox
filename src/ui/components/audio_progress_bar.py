"""
音频播放进度条组件

提供可拖动的进度条和时间显示
"""

import flet as ft


class AudioProgressBar(ft.Container):
    """可拖动的音频播放进度条"""

    def __init__(self, on_seek=None):
        """
        初始化进度条

        Args:
            on_seek: 拖动进度条时的回调函数，接收进度值 (0.0-1.0)
        """
        # 进度条滑块
        self.progress_slider = ft.Slider(
            min=0.0,
            max=1.0,
            value=0.0,
            divisions=1000,  # 提供更平滑的拖动
            on_change_end=on_seek,  # 拖动结束时触发跳转
            disabled=True,
            active_color=ft.Colors.BLUE,
            label="{value}%",  # 显示百分比
        )

        # 时间显示
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
                            ft.Container(),  # 左侧占位
                            self.time_display,
                            ft.Container(),  # 右侧占位
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                ],
                spacing=5,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )

    def update_progress(self, progress: float, current: float, total: float):
        """
        更新进度条和时间显示

        Args:
            progress: 播放进度 (0.0-1.0)
            current: 当前播放位置（秒）
            total: 总时长（秒）
        """
        self.progress_slider.value = progress
        self.time_display.value = f"{self._format_time(current)} / {self._format_time(total)}"
        self.update()

    def set_duration(self, duration: float):
        """
        设置音频时长并启用进度条

        Args:
            duration: 音频总时长（秒）
        """
        self.progress_slider.disabled = False
        self.update_progress(0.0, 0.0, duration)

    def reset(self):
        """重置进度条到初始状态"""
        self.progress_slider.value = 0.0
        self.progress_slider.disabled = True
        self.time_display.value = "0:00 / 0:00"
        self.update()

    def _format_time(self, seconds: float) -> str:
        """
        格式化时间为 M:SS 格式

        Args:
            seconds: 秒数

        Returns:
            str: 格式化后的时间字符串
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
