"""
PhantomVox UI 事件处理模块

提供简化的 UI 事件处理功能
"""

import flet as ft
from config.config_manager import ConfigManager
from utils.logger import app_logger


class UiEvent:
    """UI 事件处理器"""

    def __init__(self, page, terminal, ui_controller=None):
        self.page = page
        self.terminal = terminal
        self.ui_controller = ui_controller

        # 初始化配置管理器
        try:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config if self.config_manager and hasattr(self.config_manager, 'config') else {}
            if not isinstance(self.config, dict):
                app_logger.warning(f"config 不是字典类型: {type(self.config)}，使用空字典")
                self.config = {}
        except Exception as e:
            app_logger.error(f"ConfigManager初始化异常: {e}", exc_info=True)
            self.config_manager = None
            self.config = {}

    def show_error_dialog(self, title, message):
        """
        显示错误对话框

        Args:
            title (str): 对话框标题
            message (str): 错误消息
        """
        page = self.page

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, size=18, color=ft.Colors.RED_600),
            content=ft.Column([
                ft.Text(message, size=13),
            ], tight=True, horizontal_alignment=ft.CrossAxisAlignment.START),
            actions=[
                ft.TextButton(
                    "复制错误信息",
                    icon=ft.Icons.COPY,
                    on_click=lambda e: self._copy_to_clipboard_wrapper(message, page)
                ),
                ft.Button(
                    "确定",
                    on_click=lambda e: page.pop_dialog()
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # 使用 Flet 的标准 API 显示对话框
        page.show_dialog(dialog)

    def _copy_to_clipboard_wrapper(self, text, page):
        """复制到剪贴板的包装函数（用于同步 event handler）"""
        async def copy_async():
            await ft.Clipboard().set(text)
            page.show_dialog(ft.SnackBar(ft.Text("已复制到剪贴板")))
        page.run_task(copy_async)

    def showMsg(self, msg):
        """显示 SnackBar 消息"""
        # 使用正确的 API 显示 SnackBar（适配 Flet 0.80.1+）
        self.page.show_dialog(ft.SnackBar(ft.Text(msg), show_close_icon=True, duration=3000))

    def cleanup(self):
        """清理所有资源引用"""
        try:
            app_logger.info("UIEvent cleaning up resources...")

            # 断开与 UI 控制器的引用
            if self.ui_controller is not None:
                self.ui_controller = None

            # 清理 page 引用
            if hasattr(self, 'page') and self.page is not None:
                self.page = None

            # 清理 terminal 引用
            if hasattr(self, 'terminal') and self.terminal is not None:
                self.terminal = None

            # 清理配置管理器引用
            if hasattr(self, 'config_manager') and self.config_manager is not None:
                self.config_manager = None

            app_logger.info("UIEvent resources cleaned up")
        except Exception:
            app_logger.exception("UIEvent cleanup error")

    def __del__(self):
        """析构函数 - 确保资源释放"""
        try:
            self.cleanup()
        except Exception:
            # __del__ 中避免使用 logger，防止模块已卸载导致错误
            pass
