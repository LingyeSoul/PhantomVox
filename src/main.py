"""
PhantomVox - 基于 Qwen3-TTS 的文本转语音应用

主程序入口
"""

import flet as ft
import asyncio
import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.main_ui import PhantomUI
from utils.logger import app_logger

# 版本号
VERSION = "0.2.0"


async def main(page: ft.Page):
    """主函数"""

    # 窗口配置
    await page.window.center()

    # 自定义标题将通过 AppBar 设置，这里清空标准标题
    page.title = "PhantomVox"
    page.theme = ft.Theme(
        color_scheme_seed=ft.Colors.BLUE,
        font_family="Microsoft YaHei"
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed=ft.Colors.BLUE,
        font_family="Microsoft YaHei"
    )
    page.theme_mode = ft.ThemeMode.DARK  # 默认深色主题

    # 窗口尺寸
    page.window.width = 950
    page.window.height = 750
    page.window.resizable = False
    page.window.min_height=750
    page.window.min_width=950
    page.window.maximizable = False
    page.window.title_bar_hidden = True


    # 清空页面
    page.clean()

    app_logger.info(f"启动 PhantomVox v{VERSION}")
    app_logger.info(f"Python 版本: {sys.version}")
    app_logger.info(f"Flet 版本: {ft.__version__}")

    # 检查是否首次运行
    from config.config_manager import ConfigManager
    config_manager = ConfigManager()
    first_run = config_manager.get("first_run", True)

    # 加载保存的主题偏好
    saved_theme = config_manager.get("theme_mode", "dark")
    page.theme_mode = ft.ThemeMode.DARK if saved_theme == "dark" else ft.ThemeMode.LIGHT

    if first_run:
        # 显示欢迎界面
        await show_welcome_dialog(page)
        config_manager.set("first_run", False)
        config_manager.save_config()

    try:
        # 初始化 UI
        phantom_ui = PhantomUI(page, VERSION)

        # 设置自定义 AppBar
        page.appbar = phantom_ui.app_bar.build()

        # 构建主界面
        main_view = phantom_ui.build_main_view()

        # 添加到页面
        page.add(main_view)

        # 显示窗口
        page.window.visible = True
        await page.window.center()
        page.update()

        app_logger.info("PhantomVox 启动成功")

        # 显示提示
        if first_run:
            page.show_dialog(ft.SnackBar(
                ft.Text("欢迎使用 PhantomVox！请先在「模型管理」中下载 TTS 模型。"),
                duration=5000
            ))
        page.update()

    except Exception as e:
        app_logger.error(f"启动失败: {str(e)}", exc_info=True)
        show_error_dialog(page, str(e))


async def show_welcome_dialog(page: ft.Page):
    """显示欢迎对话框"""
    dialog = ft.AlertDialog(
        title=ft.Row([
            ft.Icon(ft.Icons.ROCKET_LAUNCH, color=ft.Colors.AMBER, size=30),
            ft.Text("欢迎使用 PhantomVox", size=20),
        ]),
        content=ft.Column([
            ft.Text(
                "PhantomVox 是一个基于 Qwen3-TTS 的文本转语音应用",
                size=14
            ),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
            ft.Text("快速开始：", size=14, weight=ft.FontWeight.BOLD),
            ft.Text("1. 在「模型管理」页面下载 TTS 模型", size=13),
            ft.Text("2. 在「语音合成」页面输入文本", size=13),
            ft.Text("3. 点击「播放」按钮生成语音", size=13),
        ], spacing=5, tight=True),
        actions=[
            ft.TextButton(
                "开始使用",
                on_click=lambda _: page.pop_dialog()
            ),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    page.show_dialog(dialog)


def show_error_dialog(page: ft.Page, error_message: str):
    """显示错误对话框"""

    def copy_error(_):
        page.clipboard.set_text(error_message)
        page.show_dialog(ft.SnackBar(ft.Text("错误信息已复制到剪贴板")))

    page.add(ft.Column([
        ft.Row([
            ft.Icon(ft.Icons.ERROR_OUTLINE, size=40, color=ft.Colors.RED),
            ft.Text("启动失败", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.RED),
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
        ft.Divider(height=30, color=ft.Colors.TRANSPARENT),
        ft.Container(
            content=ft.Column([
                ft.Text("错误信息:", size=14, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Text(error_message, size=13),
                    padding=15,
                    bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.RED),
                    border_radius=8,
                ),
            ], spacing=10),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
            border_radius=12,
        ),
        ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
        ft.Text("请检查：", size=14, weight=ft.FontWeight.BOLD),
        ft.Text("• 是否已正确安装所有依赖", size=13),
        ft.Text("• 是否已运行 pip install -r requirements.txt", size=13),
        ft.Text("• qwen-tts 是否正确安装", size=13),
        ft.Text("• 查看终端日志获取更多信息", size=13),
        ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
        ft.Row([
            ft.Button(
                "复制错误信息",
                icon=ft.Icons.COPY,
                on_click=copy_error
            ),
            ft.Button(
                "查看日志",
                icon=ft.Icons.TERMINAL,
                on_click=lambda _: page.show_dialog(ft.SnackBar(ft.Text("请查看控制台终端")))
            ),
        ], spacing=10)
    ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER))


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行 Flet 应用
    ft.run(main,view=ft.AppView.FLET_APP_HIDDEN)
