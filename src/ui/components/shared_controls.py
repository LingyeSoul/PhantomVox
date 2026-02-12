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


def create_model_dropdown(
    model_manager,
    model_type: str,
    on_change=None,
    width: int = 200
) -> ft.Dropdown:
    """
    创建模型选择下拉框

    此函数提供统一的模型选择下拉框样式和逻辑。

    Args:
        model_manager: 模型管理器实例
        model_type: 模型类型（如 'customvoice', 'base', 'voicedesign'）
        on_change: 选择变更回调函数
        width: 控件宽度，默认200

    Returns:
        ft.Dropdown: 配置好的下拉框
    """
    usable_models = model_manager.list_usable_models_by_type(model_type)
    model_options = []
    for model_id in usable_models:
        model_info = model_manager.get_model_info(model_id)
        if model_info:
            model_options.append(ft.dropdown.Option(model_id, model_info.name))

    default_model = usable_models[0] if usable_models else None
    return ft.Dropdown(
        label="选择模型",
        options=model_options,
        value=default_model,
        width=width,
        text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        disabled=len(usable_models) == 0,
        on_select=on_change
    )


def create_batch_controls(
    on_toggle=None,
    default_batch_size: str = "16"
) -> dict:
    """
    创建批量推理控件组

    此函数创建批量推理所需的所有控件，包括开关、批次大小输入框和分割模式下拉框。

    Args:
        on_toggle: 批量推理开关切换回调
        default_batch_size: 默认批次大小，默认"16"

    Returns:
        dict: 包含所有批量控件的字典
        - switch: 批量推理开关
        - size_input: 批次大小输入框
        - mode_dropdown: 分割模式下拉框
        - progress_text: 进度文本
        - progress_bar: 进度条
    """
    batch_streaming_switch = ft.Switch(
        label="",
        value=False,
        on_change=on_toggle
    )

    batch_size_input = ft.TextField(
        label="分批大小",
        value=default_batch_size,
        width=100,
        keyboard_type=ft.KeyboardType.NUMBER,
        text_style=ft.TextStyle(font_family="Microsoft YaHei", size=12),
    )

    split_mode_dropdown = ft.Dropdown(
        label="分割模式",
        options=[
            ft.dropdown.Option("multiline", "按行分割"),
            ft.dropdown.Option("sentence", "按句分割"),
        ],
        value="multiline",
        width=120,
        text_style=ft.TextStyle(font_family="Microsoft YaHei", size=12),
    )

    batch_progress_text = ft.Text("", size=12, visible=False)
    batch_progress_bar = ft.ProgressBar(value=0, visible=False, bar_height=4)

    return {
        "switch": batch_streaming_switch,
        "size_input": batch_size_input,
        "mode_dropdown": split_mode_dropdown,
        "progress_text": batch_progress_text,
        "progress_bar": batch_progress_bar,
    }


def create_audio_filename_input() -> ft.TextField:
    """
    创建音频文件名输入框

    此函数提供统一的音频文件名输入框样式。

    Returns:
        ft.TextField: 配置好的输入框
    """
    return ft.TextField(
        label="音频文件名（可选，留空则自动生成）",
        hint_text="例如: 我的语音",
        text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        expand=True
    )


def create_advanced_options_tile(controls: list = None) -> ft.ExpansionTile:
    """
    创建高级选项 ExpansionTile

    Args:
        controls: 内部控件列表，默认包含采样率选项

    Returns:
        ft.ExpansionTile: 配置好的高级选项面板
    """
    if controls is None:
        controls = [
            ft.ListTile(
                title=ft.Text("采样率", size=13),
                trailing=ft.Dropdown(
                    options=[
                        ft.dropdown.Option("24000", "24000 Hz"),
                    ],
                    value="24000",
                    width=120,
                    text_style=ft.TextStyle(font_family="Microsoft YaHei", size=12),
                ),
            )
        ]

    return ft.ExpansionTile(
        title=ft.Text("高级选项", size=14, weight=ft.FontWeight.BOLD),
        subtitle=ft.Text("配置生成参数", size=12),
        collapsed_bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
        bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
        controls_padding=ft.Padding.all(10),
        controls=controls,
    )

