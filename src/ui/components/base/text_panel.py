"""
通用文本输入面板组件

可在多个语音生成页面之间复用
"""

import flet as ft


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
