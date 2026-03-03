"""
对话框工具函数

提供通用的对话框辅助功能
"""

import flet as ft
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


async def show_confirm_dialog(
    page: ft.Page,
    title: str,
    message: str,
    on_confirm: Optional[Callable] = None,
    on_cancel: Optional[Callable] = None,
    confirm_text: str = "确认",
    cancel_text: str = "取消",
    icon: Optional[str] = None,
    icon_color: Optional[str] = None,
) -> bool:
    """
    显示确认对话框

    Args:
        page: Flet Page 实例
        title: 对话框标题
        message: 对话框消息
        on_confirm: 确认回调（可选）
        on_cancel: 取消回调（可选）
        confirm_text: 确认按钮文本
        cancel_text: 取消按钮文本
        icon: 图标名称（可选）
        icon_color: 图标颜色（可选）

    Returns:
        bool: 用户是否确认
    """
    result = {"confirmed": False}

    def handle_confirm(_):
        result["confirmed"] = True
        page.pop_dialog()
        if on_confirm:
            try:
                on_confirm()
            except Exception as e:
                logger.error(f"Confirm callback failed: {e}", exc_info=True)
                pass

    def handle_cancel(_):
        result["confirmed"] = False
        page.pop_dialog()
        if on_cancel:
            try:
                on_cancel()
            except Exception as e:
                logger.error(f"Cancel callback failed: {e}", exc_info=True)
                pass

    # 构建标题行
    title_controls = []
    if icon:
        title_controls.append(
            ft.Icon(icon, color=icon_color or ft.Colors.WARNING, size=24)
        )
    title_controls.append(ft.Text(title, size=16, weight=ft.FontWeight.BOLD))

    dialog = ft.AlertDialog(
        title=ft.Row(title_controls, spacing=10),
        content=ft.Text(message, size=14),
        actions=[
            ft.TextButton(
                cancel_text,
                on_click=handle_cancel,
            ),
            ft.TextButton(
                confirm_text,
                on_click=handle_confirm,
                style=ft.ButtonStyle(
                    color=ft.Colors.RED
                    if "覆盖" in confirm_text or "删除" in confirm_text
                    else ft.Colors.BLUE
                ),
            ),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    page.show_dialog(dialog)
    return result["confirmed"]


async def show_overwrite_dialog(
    page: ft.Page,
    filename: str,
    on_confirm: Optional[Callable] = None,
    on_cancel: Optional[Callable] = None,
) -> bool:
    """
    显示文件覆盖确认对话框

    Args:
        page: Flet Page 实例
        filename: 文件名
        on_confirm: 确认覆盖回调
        on_cancel: 取消回调

    Returns:
        bool: 用户是否确认覆盖
    """
    return await show_confirm_dialog(
        page=page,
        title="文件已存在",
        message=f'文件 "{filename}" 已存在，是否覆盖？',
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text="覆盖",
        cancel_text="取消",
        icon=ft.Icons.WARNING_ROUNDED,
        icon_color=ft.Colors.WARNING,
    )


class ConfirmDialogHelper:
    """
    确认对话框辅助类（用于异步回调场景）

    用法:
        helper = ConfirmDialogHelper(page)

        def on_save_confirmed():
            # 执行保存
            ...

        # 检查并显示确认对话框
        if file_exists:
            helper.show_overwrite_dialog(
                filename="audio.wav",
                on_confirm=on_save_confirmed
            )
        else:
            on_save_confirmed()
    """

    def __init__(self, page: ft.Page):
        self.page = page

    def show_confirm_dialog(
        self,
        title: str,
        message: str,
        on_confirm: Optional[Callable] = None,
        on_cancel: Optional[Callable] = None,
        confirm_text: str = "确认",
        cancel_text: str = "取消",
        icon: Optional[str] = None,
        icon_color: Optional[str] = None,
    ):
        """显示确认对话框"""

        def handle_confirm(_):
            self.page.pop_dialog()
            if on_confirm:
                try:
                    on_confirm()
                except Exception as e:
                    logger.error(f"Confirm callback failed: {e}", exc_info=True)

        def handle_cancel(_):
            self.page.pop_dialog()
            if on_cancel:
                try:
                    on_cancel()
                except Exception as e:
                    logger.error(f"Cancel callback failed: {e}", exc_info=True)

        title_controls = []
        if icon:
            title_controls.append(
                ft.Icon(icon, color=icon_color or ft.Colors.WARNING, size=24)
            )
        title_controls.append(ft.Text(title, size=16, weight=ft.FontWeight.BOLD))

        dialog = ft.AlertDialog(
            title=ft.Row(title_controls, spacing=10),
            content=ft.Text(message, size=14),
            actions=[
                ft.TextButton(cancel_text, on_click=handle_cancel),
                ft.TextButton(
                    confirm_text,
                    on_click=handle_confirm,
                    style=ft.ButtonStyle(
                        color=ft.Colors.RED
                        if "覆盖" in confirm_text or "删除" in confirm_text
                        else ft.Colors.BLUE
                    ),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.show_dialog(dialog)

    def show_overwrite_dialog(
        self,
        filename: str,
        on_confirm: Optional[Callable] = None,
        on_cancel: Optional[Callable] = None,
    ):
        """显示文件覆盖确认对话框"""
        self.show_confirm_dialog(
            title="文件已存在",
            message=f'文件 "{filename}" 已存在，是否覆盖？',
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="覆盖",
            cancel_text="取消",
            icon=ft.Icons.WARNING_ROUNDED,
            icon_color=ft.Colors.WARNING,
        )
