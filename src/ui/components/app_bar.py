"""
PhantomVox AppBar 组件

自定义顶部应用栏，包含窗口控制和主题切换功能
参考 SillyTavernLauncher 的 AppBar 设计
"""

import flet as ft
from typing import Callable, Optional


class PhantomAppBar:
    """自定义 AppBar，包含窗口控制和主题切换功能"""

    def __init__(
        self,
        page: ft.Page,
        version: str,
        on_theme_toggle: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        初始化 AppBar

        Args:
            page: Flet Page 对象
            version: 应用版本号
            on_theme_toggle: 主题切换回调函数
            on_close: 关闭窗口回调函数
        """
        self.page = page
        self.version = version
        self.on_theme_toggle = on_theme_toggle
        self.on_close = on_close

    def _minimize_window(self, e):
        """最小化窗口处理"""
        try:
            self.page.window.minimized = True
            self.page.update()
        except Exception as ex:
            print(f"最小化窗口失败: {ex}")

    def _toggle_theme(self, e):
        """切换主题处理"""
        try:
            # 获取当前主题模式
            current_mode = self.page.theme_mode
            new_mode = ft.ThemeMode.LIGHT if current_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK

            # 更新主题模式
            self.page.theme_mode = new_mode

            # 重建 AppBar 以更新主题图标
            self.page.appbar = self.build()
            self.page.update()

            # 调用自定义主题切换回调
            if self.on_theme_toggle:
                self.on_theme_toggle(new_mode)
        except Exception as ex:
            print(f"切换主题失败: {ex}")

    def _close_window(self, e):
        """关闭窗口处理"""
        try:
            if self.on_close:
                self.on_close(e)
            else:
                # 如果没有提供关闭回调，直接关闭
                self.page.window.destroy()
        except Exception as ex:
            print(f"关闭窗口失败: {ex}")

    def _get_theme_icon(self):
        """根据当前主题模式获取对应的图标"""
        current_mode = self.page.theme_mode if hasattr(self.page, 'theme_mode') else ft.ThemeMode.DARK
        # 在深色主题下显示亮色图标（提示切换到亮色），反之亦然
        return ft.Icons.LIGHT_MODE if current_mode == ft.ThemeMode.DARK else ft.Icons.DARK_MODE

    def _get_theme_tooltip(self):
        """根据当前主题模式获取工具提示文本"""
        current_mode = self.page.theme_mode if hasattr(self.page, 'theme_mode') else ft.ThemeMode.DARK
        return "切换到亮色主题" if current_mode == ft.ThemeMode.DARK else "切换到深色主题"

    def build(self) -> ft.AppBar:
        """
        构建 AppBar 组件

        Returns:
            ft.AppBar: Flet AppBar 控件
        """
        return ft.AppBar(
            leading_width=40,
            title=ft.WindowDragArea(
                content=ft.Text(f"PhantomVox V{self.version}"),
                width=800,
            ),
            center_title=False,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            actions=[
                # 最小化按钮
                ft.IconButton(
                    ft.Icons.MINIMIZE,
                    on_click=self._minimize_window,
                    icon_size=30,
                    tooltip="最小化"
                ),
                # 主题切换按钮
                ft.IconButton(
                    icon=self._get_theme_icon(),
                    on_click=self._toggle_theme,
                    icon_size=30,
                    tooltip=self._get_theme_tooltip()
                ),
                # 关闭按钮
                ft.IconButton(
                    ft.Icons.CLOSE,
                    on_click=self._close_window,
                    icon_size=30,
                    tooltip="关闭"
                ),
            ],
        )
