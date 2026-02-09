"""
PhantomVox 共享UI控件

提供可在多个视图之间复用的UI组件工厂函数
"""

import flet as ft
from typing import Any


def create_generate_button(on_click_handler) -> ft.Button:
    """
    创建标准化的生成语音按钮

    此函数提供统一的按钮样式，确保整个应用中生成按钮的一致性。
    如果需要修改按钮样式，只需在此处修改即可。

    Args:
        on_click_handler: 按钮点击事件的回调函数

    Returns:
        ft.Button: 配置好的生成语音按钮
    """
    return ft.Button(
        "生成语音",
        icon=ft.Icons.SEND,
        style=ft.ButtonStyle(
            text_style=ft.TextStyle(
                font_family="Microsoft YaHei",
                weight=ft.FontWeight.BOLD
            )
        ),
        on_click=on_click_handler
    )


def create_header_with_button(title: str, on_click_handler, button_text: str = "生成语音") -> ft.Row:
    """
    创建带标题和按钮的标准标题栏

    此函数创建一个包含标题和生成按钮的标题栏，标题在左侧，按钮在右侧。
    确保整个应用中标题栏的一致性。

    Args:
        title: 标题文本（如"文本输入"）
        on_click_handler: 按钮点击事件的回调函数
        button_text: 按钮文本，默认为"生成语音"

    Returns:
        ft.Row: 配置好的标题栏组件
    """
    return ft.Row([
        ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
        ft.Container(expand=True),
        ft.Button(
            button_text,
            icon=ft.Icons.SEND,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(
                    font_family="Microsoft YaHei",
                    weight=ft.FontWeight.BOLD
                )
            ),
            on_click=on_click_handler
        ),
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)


def create_labeled_control(label: str, control: Any, spacing: int = 5) -> ft.Column:
    """
    创建带标题的控件组

    此函数创建一个包含标题和控件的垂直布局，常用于表单分组。
    标题使用粗体14号字，控件紧随其后。

    Args:
        label: 控件组的标题文本（如"说话人选择"）
        control: Flet控件对象（如Dropdown、TextField等）
        spacing: 标题和控件之间的间距，默认为5

    Returns:
        ft.Column: 包含标题和控件的垂直布局
    """
    return ft.Column([
        ft.Text(label, size=14, weight=ft.FontWeight.BOLD),
        control,
    ], spacing=spacing)
