"""
声音设计 (Voice Design) 页面

通过自然语言描述设计声音
"""

import flet as ft
import logging
import asyncio
import os
import time
import numpy as np
import torch

from ui.components.shared_controls import create_labeled_control
from ui.components.audio_progress_bar import AudioProgressBar
from ui.components.voice_library import VoiceLibrary
from tts.audio_temp_manager import AudioTempManager
from tts.text_splitter import smart_split
from utils.time_utils import format_elapsed_time

logger = logging.getLogger(__name__)

# 收藏相关常量
DEFAULT_DESCRIPTION_MAX_LENGTH = 10
MAX_NAME_LENGTH = 100
MAX_CONTENT_LENGTH = 5000


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

        self.play_button = ft.Button(
            "播放",
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_play
        )

        self.stop_button = ft.Button(
            "停止",
            icon=ft.Icons.STOP,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_stop
        )

        self.save_button = ft.Button(
            "保存音频",
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                text_style=ft.TextStyle(font_family="Microsoft YaHei")
            ),
            on_click=on_save
        )

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


class VoiceDesignView(ft.Container):
    """声音设计页面"""

    def __init__(
        self,
        page: ft.Page,
        tts_engine_getter,
        audio_manager_getter,
        terminal,
        voice_library: VoiceLibrary,
        config_manager,
        model_manager,
        on_clear_engine_cache=None
    ):
        # 使用私有变量存储 page，避免与 ft.Container 的 page 属性冲突
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.audio_manager_getter = audio_manager_getter
        self.terminal = terminal
        self.voice_library = voice_library
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.on_clear_engine_cache = on_clear_engine_cache

        # 当前生成的音频和临时文件路径
        self._last_audio = None
        self._temp_audio_file = None
        self._is_generating = False

        # 音频临时文件管理器
        self._audio_temp_manager = AudioTempManager()

        # 创建 FloatingActionButton（由 main_ui 集中管理）
        self._fab = ft.FloatingActionButton(
            icon=ft.Icons.SEND,
            bgcolor=ft.Colors.BLUE,
            on_click=self._on_generate,
            tooltip="生成语音",
        )
        # PDCA 循环 #1 修复: 不在 __init__ 中设置 page.floating_action_button
        # 原因: 视图是延迟初始化并缓存的，切换视图时 __init__ 不会再次调用
        # 修复: 由 main_ui.on_navigation_change 集中管理 FAB 切换

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

    def _build_ui(self):
        """构建UI界面"""
        # 模型选择下拉框 - 只显示 VoiceDesign 模型
        usable_models = self.model_manager.list_usable_models_by_type("voicedesign")
        model_options = []
        for model_id in usable_models:
            model_info = self.model_manager.get_model_info(model_id)
            if model_info:
                model_options.append(ft.dropdown.Option(model_id, model_info.name))

        default_model = usable_models[0] if usable_models else None
        self.model_dropdown = ft.Dropdown(
            label="选择模型",
            options=model_options,
            value=default_model,
            width=200,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            disabled=len(usable_models) == 0,
            on_select=self._on_model_changed
        )

        # 声音描述输入框
        self.design_input = ft.TextField(
            label="声音描述",
            multiline=True,
            min_lines=4,
            max_lines=6,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 字符计数
        self.char_count = ft.Text(
            "字符数: 0 / 推荐 30-80",
            size=12
        )
        self.design_input.on_change = self._on_design_change

        # 预设声音卡片
        preset_cards = []
        presets = self.voice_library.get_all_design_presets()

        for name, desc in presets.items():
            # 内置预设使用特殊标识
            is_builtin = name in ["温柔女声", "活泼少女", "磁性大叔",
                                  "正太少年", "知性御姐", "沉稳长者"]

            card = ft.Container(
                content=ft.Text(name, size=12, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                padding=8,
                bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                border_radius=8,
                width=100,
                on_click=lambda _, n=name, d=desc: self._on_preset_click(_, n, d),
                tooltip=desc
            )
            preset_cards.append(card)

        # 我的收藏 (Chips，支持滚动)
        self.fav_chips = ft.Row(
            [ft.Text("暂无收藏", size=12)],
            spacing=5,
            wrap=False,
            scroll=ft.ScrollMode.AUTO,
            height=40
        )
        self._refresh_favorites()  # 初始化时加载收藏

        # 设计历史
        self.history_list = ft.ListView(
            expand=1,
            spacing=5,
            item_extent=40
        )
        self._refresh_history()

        # 文本输入面板
        self.text_panel = TextPanel(
            placeholder="请输入要转换的文本...",
            min_lines=6,
            max_lines=10,
            on_clear=self._on_clear_text
        )

        # 批量推理控件
        self.batch_streaming_switch = ft.Switch(
            label="",
            value=False,
            on_change=self._on_batch_streaming_toggle
        )

        self.batch_size_input = ft.TextField(
            label="分批大小",
            value="16",
            width=100,
            keyboard_type=ft.KeyboardType.NUMBER,
            text_style=ft.TextStyle(font_family="Microsoft YaHei", size=12),
        )

        self.split_mode_dropdown = ft.Dropdown(
            label="分割模式",
            options=[
                ft.dropdown.Option("multiline", "按行分割"),
                ft.dropdown.Option("sentence", "按句分割"),
            ],
            value="multiline",
            width=120,
            text_style=ft.TextStyle(font_family="Microsoft YaHei", size=12),
        )

        self.batch_progress_text = ft.Text("", size=12, visible=False)
        self.batch_progress_bar = ft.ProgressBar(value=0, visible=False, bar_height=4)

        # 高级选项 ExpansionTile
        self.advanced_options_tile = ft.ExpansionTile(
            title=ft.Text("高级选项", size=14, weight=ft.FontWeight.BOLD),
            subtitle=ft.Text("配置生成参数", size=12),
            collapsed_bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
            bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
            controls_padding=ft.Padding.all(10),
            controls=[
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
            ],
        )

        # 批量推理 ExpansionTile
        self.batch_inference_tile = ft.ExpansionTile(
            title=ft.Text("批量推理", size=14, weight=ft.FontWeight.BOLD),
            subtitle=ft.Text("批量生成多个语音", size=12),
            collapsed_bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
            bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
            controls_padding=ft.Padding.all(10),
            controls=[
                ft.Column([
                    ft.Row([
                        ft.Text("启用", size=13),
                        self.batch_streaming_switch,
                        ft.Text("分批大小:", size=13),
                        self.batch_size_input,
                    ], alignment=ft.MainAxisAlignment.START, spacing=10),
                    create_labeled_control("分割模式", self.split_mode_dropdown),
                    ft.Text("按行分割: 每行一个文本\n按句分割: 自动识别句子边界", size=11,
                           color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                    self.batch_progress_text,
                    self.batch_progress_bar,
                ], spacing=5),
            ],
        )

        # 音频控制面板
        self.audio_control = AudioControlPanel(
            on_play=self._on_play,
            on_stop=self._on_stop,
            on_save=self._on_save,
            on_seek=self._on_seek,
            has_audio=False
        )

        # 音频文件名自定义输入框
        self.audio_filename_input = ft.TextField(
            label="音频文件名（可选，留空则自动生成）",
            hint_text="例如: 我的语音",
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            expand=True
        )

        # 右侧控制面板
        control_panel = ft.Container(
            content=ft.Column(
                [
                    # 声音描述
                    ft.Column([
                        ft.Row([
                            ft.Text("声音描述", size=14, weight=ft.FontWeight.BOLD),
                            ft.Icon(
                                ft.Icons.INFO_OUTLINE,
                                size=16,
                                tooltip="用自然语言描述你想要的声音特征"
                            )
                        ]),
                        self.design_input,
                        self.char_count,
                    ], spacing=5),

                    ft.Divider(),

                    # 预设声音
                    ft.Column([
                        ft.Text("预设声音 📚", size=14, weight=ft.FontWeight.BOLD),
                        ft.GridView(
                            runs_count=2,
                            max_extent=50,
                            spacing=10,
                            run_spacing=10,
                            controls=preset_cards
                        ),
                    ], spacing=5),

                    ft.Divider(),

                    # 我的收藏
                    ft.Column([
                        ft.Row([
                            ft.Text("我的收藏 ⭐", size=14, weight=ft.FontWeight.BOLD),
                            ft.IconButton(
                                icon=ft.Icons.ADD,
                                icon_size=18,
                                tooltip="保存当前描述为收藏",
                                on_click=self._on_save_favorite
                            )
                        ]),
                        self.fav_chips,
                    ], spacing=5),

                    ft.Divider(),

                    # 音频控制

                    # 音频控制
                    self.audio_control,

                    ft.Divider(),

                    # 音频文件名设置
                    create_labeled_control("保存设置", self.audio_filename_input),

                    ft.Divider(),

                    # 设计历史
                    ft.Column([
                        ft.Text("设计历史", size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=self.history_list,
                            height=150,
                            bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
                            border_radius=8,
                            padding=10
                        ),
                    ], spacing=5),
                ],
                spacing=10,
                scroll=ft.ScrollMode.AUTO
            ),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
            border_radius=12,
            width=380
        )

        # 左侧面板卡片
        left_panel = ft.Container(
            content=ft.Column([
                # 模型选择
                create_labeled_control("模型选择", self.model_dropdown),

                ft.Divider(),

                ft.Text("文本输入", size=16, weight=ft.FontWeight.BOLD),
                self.text_panel,

                ft.Divider(),

                # 高级选项
                self.advanced_options_tile,

                # 批量推理
                self.batch_inference_tile,
            ], spacing=10, scroll=ft.ScrollMode.AUTO),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
            border_radius=12,
            expand=True
        )

        # 主布局
        return ft.Row(
            [
                # 左侧文本输入区
                left_panel,

                # 右侧控制面板
                control_panel
            ],
            spacing=20,
            expand=True
        )

    def _on_model_changed(self, e):
        """模型选择改变事件"""
        if self.on_clear_engine_cache:
            self.on_clear_engine_cache(self.model_dropdown.value)
        self.terminal.add_log(f"模型已切换: {self.model_dropdown.value}")

    def refresh_model_dropdown(self):
        """刷新模型下拉框选项"""
        try:
            # 获取当前选中的模型
            current_value = self.model_dropdown.value

            # 重新获取 VoiceDesign 模型列表
            usable_models = self.model_manager.list_usable_models_by_type("voicedesign")
            model_options = []
            for model_id in usable_models:
                model_info = self.model_manager.get_model_info(model_id)
                if model_info:
                    model_options.append(ft.dropdown.Option(model_id, model_info.name))

            # 更新下拉框选项
            self.model_dropdown.options = model_options

            # 如果当前选中的模型仍然可用，保持选中；否则选择第一个
            if current_value in usable_models:
                self.model_dropdown.value = current_value
            elif usable_models:
                self.model_dropdown.value = usable_models[0]
            else:
                self.model_dropdown.value = None

            # 更新禁用状态
            self.model_dropdown.disabled = len(usable_models) == 0

            # 刷新显示
            self.model_dropdown.update()
        except Exception as e:
            logger.error(f"刷新模型下拉框失败: {str(e)}", exc_info=True)

    def _on_design_change(self, e):
        """声音描述输入变化事件"""
        text = self.design_input.value or ""
        char_count = len(text)
        self.char_count.value = f"字符数: {char_count} / 推荐 30-80"
        self.char_count.update()

    def _on_preset_click(self, _, name: str, desc: str):
        """预设声音卡片点击事件"""
        self.design_input.value = desc
        self.design_input.update()
        self._on_design_change(None)
        self.terminal.add_log(f"已选择预设: {name}")

    def _on_save_favorite(self, _):
        """保存当前描述为收藏"""
        desc = self.design_input.value or ""
        if not desc or not desc.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请先输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

        # 生成默认名称（使用描述的前N个字符）
        default_name = desc.strip()[:DEFAULT_DESCRIPTION_MAX_LENGTH] + ("..." if len(desc) > DEFAULT_DESCRIPTION_MAX_LENGTH else "")

        # 创建名称输入框
        name_input = ft.TextField(
            label="收藏名称",
            value=default_name,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            autofocus=True
        )

        # 显示对话框
        def save_dialog(_):
            name = name_input.value.strip() or default_name

            # 输入验证
            if len(name) > 100:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"名称过长（最多100字符，当前{len(name)}字符）"),
                    bgcolor=ft.Colors.RED
                ))
                return

            if len(desc) > 5000:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"内容过长（最多5000字符，当前{len(desc)}字符）"),
                    bgcolor=ft.Colors.RED
                ))
                return

            # 检查名称唯一性
            existing_names = [f["name"] for f in self.voice_library.get_favorite_designs()]
            if name in existing_names:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"收藏名称 \"{name}\" 已存在，请使用其他名称"),
                    bgcolor=ft.Colors.RED
                ))
                return

            success = self.voice_library.add_favorite_design(name, desc)

            if success:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"已收藏: {name}"),
                    bgcolor=ft.Colors.GREEN
                ))
                self.terminal.add_log(f"已保存收藏: {name}")
                self._refresh_favorites()  # 刷新收藏显示
            else:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("保存失败或已存在"),
                    bgcolor=ft.Colors.RED
                ))
            dialog.open = False
            self._page.update()

        def close_dialog(_):
            dialog.open = False
            self._page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("保存收藏", weight=ft.FontWeight.BOLD),
            content=name_input,
            actions=[
                ft.TextButton("取消", on_click=close_dialog),
                ft.TextButton("保存", on_click=save_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self._page.show_dialog(dialog)

    def _refresh_favorites(self):
        """刷新收藏列表"""
        favorites = self.voice_library.get_favorite_designs()

        if not favorites:
            self.fav_chips.controls = [ft.Text("暂无收藏", size=12)]
        else:
            self.fav_chips.controls.clear()
            for fav in favorites:
                # 创建带右键菜单的 Chip
                menu_button = ft.PopupMenuButton(
                    icon=ft.Icons.MORE_VERT,
                    items=[
                        ft.PopupMenuItem(
                            content=ft.Text("编辑名称"),
                            icon=ft.Icons.EDIT,
                            on_click=lambda _, f=fav: self._on_edit_favorite_name(_, f)
                        ),
                        ft.PopupMenuItem(
                            content=ft.Text("删除"),
                            icon=ft.Icons.DELETE,
                            on_click=lambda _, f=fav: self._on_delete_favorite(_, f)
                        ),
                    ]
                )

                chip_row = ft.Row([
                    ft.Chip(
                        label=ft.Text(fav["name"], size=11),
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                        on_click=lambda _, d=fav["description"]: self._on_favorite_click(_, d),
                    ),
                    menu_button
                ], spacing=5)

                self.fav_chips.controls.append(chip_row)

        # 只有在控件已添加到页面时才调用 update()
        try:
            self.fav_chips.update()
        except RuntimeError:
            pass

    def _on_favorite_click(self, _, desc: str):
        """收藏项点击事件"""
        self.design_input.value = desc
        self.design_input.update()
        self._on_design_change(None)
        self.terminal.add_log("已加载收藏的设计")

    def _on_edit_favorite_name(self, _, fav: dict):
        """编辑收藏"""
        old_name = fav["name"]
        old_desc = fav["description"]

        # 创建输入框
        name_input = ft.TextField(
            label="收藏名称",
            value=old_name,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )

        desc_input = ft.TextField(
            label="声音描述",
            value=old_desc,
            multiline=True,
            min_lines=3,
            max_lines=5,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            autofocus=True
        )

        # 显示对话框
        def save_dialog(_):
            new_name = name_input.value.strip()
            new_desc = desc_input.value.strip()

            if not new_name or not new_desc:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("名称和描述不能为空"),
                    bgcolor=ft.Colors.RED
                ))
                return

            success = self.voice_library.update_favorite_design(old_desc, new_name, new_desc)

            if success:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("已修改"),
                    bgcolor=ft.Colors.GREEN
                ))
                self._refresh_favorites()
                # 如果当前输入框是旧描述，更新它
                if self.design_input.value == old_desc:
                    self.design_input.value = new_desc
                    self.design_input.update()
                    self._on_design_change(None)
            else:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("修改失败"),
                    bgcolor=ft.Colors.RED
                ))

            dialog.open = False
            self._page.update()

        def close_dialog(_):
            dialog.open = False
            self._page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("编辑收藏", weight=ft.FontWeight.BOLD),
            content=ft.Column([name_input, desc_input], spacing=10, tight=True),
            actions=[
                ft.TextButton("取消", on_click=close_dialog),
                ft.TextButton("保存", on_click=save_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self._page.show_dialog(dialog)

    def _on_delete_favorite(self, _, fav: dict):
        """删除收藏"""
        desc = fav["description"]
        name = fav["name"]

        # 确认对话框
        def confirm_delete(_):
            success = self.voice_library.remove_favorite_design(desc)
            if success:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"已删除: {name}"),
                    bgcolor=ft.Colors.GREEN
                ))
                self._refresh_favorites()
            else:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("删除失败"),
                    bgcolor=ft.Colors.RED
                ))
            confirm_dialog.open = False
            self._page.update()

        def close_dialog(_):
            confirm_dialog.open = False
            self._page.update()

        confirm_dialog = ft.AlertDialog(
            title=ft.Text("确认删除", weight=ft.FontWeight.BOLD),
            content=ft.Text(f"确定要删除收藏 \"{name}\" 吗？"),
            actions=[
                ft.TextButton("取消", on_click=close_dialog),
                ft.TextButton("删除", on_click=confirm_delete),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self._page.show_dialog(confirm_dialog)

    def _refresh_history(self):
        """刷新设计历史"""
        self.history_list.controls.clear()

        history = self.voice_library.get_design_history(limit=10)
        for item in history:
            timestamp = item["timestamp"][:10]  # 只显示日期部分
            desc = item["description"][:40] + "..." if len(item["description"]) > 40 else item["description"]

            control = ft.Container(
                content=ft.Row([
                    ft.Text(f"[{timestamp}] {item['name']}", size=12, weight=ft.FontWeight.W_500),
                    ft.Text(desc, size=11, expand=True),
                ], spacing=5),
                padding=5,
                on_click=lambda _, d=item["description"]: self._on_history_click(_, d),
                tooltip=item["description"]
            )
            self.history_list.controls.append(control)

        # 只有在控件已添加到页面时才调用 update()
        try:
            self.history_list.update()
        except RuntimeError:
            pass

    def _on_history_click(self, _, desc: str):
        """历史记录点击事件"""
        self.design_input.value = desc
        self.design_input.update()
        self._on_design_change(None)
        self.terminal.add_log("已加载历史设计")

    def _on_clear_text(self, _):
        """清空文本"""
        self.text_panel.clear()

    def _on_batch_streaming_toggle(self, e):
        """批量推理开关切换"""
        pass

    async def _on_generate(self, e):
        """生成语音按钮点击事件"""
        if self._is_generating:
            return

        # 获取文本
        text = self.text_panel.get_text()
        if not text or not text.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请输入要转换的文本"),
                bgcolor=ft.Colors.RED
            ))
            return

        # 获取声音描述
        design_prompt = self.design_input.value or ""
        if not design_prompt or not design_prompt.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

        # 检查是否启用批量模式
        if self.batch_streaming_switch.value:
            await self._on_generate_with_batch(text, design_prompt)
        else:
            await self._on_generate_single(text, design_prompt)

    async def _on_generate_single(self, text: str, design_prompt: str):
        """单个文本生成"""
        self._is_generating = True
        start_time = time.perf_counter()  # 开始计时
        self.terminal.add_log("正在生成语音...")

        # 强制UI更新，让第一条日志立即显示
        try:
            self._page.update()
        except:
            pass

        try:
            # 获取TTS引擎（异步）
            tts_engine = await self.tts_engine_getter()

            # 生成语音
            self.terminal.add_log(f"声音描述: {design_prompt[:50]}...")

            # 强制UI更新，让参数日志立即显示
            try:
                self._page.update()
            except:
                pass

            # 在后台线程中执行TTS生成（使用异步API）
            audio, sr = await tts_engine.voice_design_synthesize_async(
                text=text,
                design_prompt=design_prompt,
                language="Chinese",
                timeout=300.0
            )

            self.terminal.add_log("✓ 语音生成成功")

            # 使用临时文件管理器保存音频
            if self._temp_audio_file:
                self._audio_temp_manager.cleanup_file(self._temp_audio_file)

            self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sr, prefix="design")

            # 保存音频数据用于计算时长
            self._last_audio = (audio, sr)

            # 保存到设计历史
            self.voice_library.save_design_history("自定义设计", design_prompt)
            self._refresh_history()

            # 更新音频控制状态
            self.audio_control.update_audio_state(True)

            # 播放音频
            await self._on_play(None)

        except Exception as e:
            logger.error(f"生成语音失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 生成失败: {str(e)}")
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"生成失败: {str(e)}"),
                bgcolor=ft.Colors.RED
            ))

        finally:
            self._is_generating = False
            elapsed_time = time.perf_counter() - start_time
            time_str = format_elapsed_time(elapsed_time)
            self.terminal.add_log(f"✓ 语音生成完成 (用时: {time_str})")

    async def _on_generate_with_batch(self, text: str, design_prompt: str):
        """批量模式生成语音"""
        if self._is_generating:
            return

        self._is_generating = True
        start_time = time.perf_counter()

        try:
            # 获取分割模式
            split_mode = self.split_mode_dropdown.value

            # 分割文本
            texts = smart_split(text, mode=split_mode, language="chinese")
            if not texts:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("没有有效的文本可生成"),
                    bgcolor=ft.Colors.RED
                ))
                return

            # 获取分批大小（限制范围 1-64，防止显存溢出）
            try:
                batch_size = int(self.batch_size_input.value)
                if batch_size < 1 or batch_size > 64:
                    batch_size = 16
            except ValueError:
                batch_size = 16

            # 获取 TTS 引擎
            tts_engine = await self.tts_engine_getter()

            # 执行批量生成
            await self._on_generate_batch(texts, design_prompt, tts_engine, batch_size)

            # 保存到设计历史
            self.voice_library.save_design_history("批量设计", design_prompt)
            self._refresh_history()

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 批量生成失败: {str(e)}")
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"批量生成失败: {str(e)}"),
                bgcolor=ft.Colors.RED
            ))
        finally:
            self._is_generating = False
            elapsed_time = time.perf_counter() - start_time
            time_str = format_elapsed_time(elapsed_time)
            self.terminal.add_log(f"✓ 批量生成完成 (用时: {time_str})")

    async def _on_generate_batch(
        self,
        texts: list,
        design_prompt: str,
        tts_engine,
        batch_size: int = 16
    ):
        """
        批量流式生成语音（支持分批处理以控制显存占用）

        Args:
            texts: 文本列表
            design_prompt: 声音设计描述
            tts_engine: TTS 引擎
            batch_size: 每批最大文本数（控制显存占用）
        """
        total = len(texts)
        self.terminal.add_log(f"开始批量生成 {total} 个文本（每批最多 {batch_size} 个）...")
        self.terminal.add_log(f"声音描述: {design_prompt[:50]}...")

        # 显示进度
        self.batch_progress_text.visible = True
        self.batch_progress_bar.visible = True
        self.batch_progress_text.value = f"准备生成 {total} 个文本..."
        self.batch_progress_bar.value = 0
        self._page.update()

        # 存储每个文本的音频块
        item_chunks = [[] for _ in range(len(texts))]
        sample_rate = 24000

        # 分批处理
        num_batches = (total + batch_size - 1) // batch_size
        global_completed = 0

        try:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total)
                batch_texts = texts[batch_start:batch_end]
                batch_num = batch_idx + 1

                self.terminal.add_log(f"处理第 {batch_num}/{num_batches} 批 (文本 {batch_start+1}-{batch_end})...")

                # 追踪当前批次每个文本的状态
                item_started = [False] * len(batch_texts)
                item_completed = [False] * len(batch_texts)

                chunk_count = 0
                async for chunks_list, sr in tts_engine.voice_design_batch_stream_synthesize_async(
                    texts=batch_texts,
                    design_prompt=design_prompt,
                    language="Chinese",
                ):
                    sample_rate = sr
                    chunk_count += 1

                    # 累积每个文本的音频块，并追踪完成状态
                    for i, chunk in enumerate(chunks_list):
                        global_idx = batch_start + i
                        if chunk.size > 0:
                            item_chunks[global_idx].append(chunk)
                            item_started[i] = True
                        elif item_started[i] and not item_completed[i]:
                            item_completed[i] = True

                    # 计算当前批次进度
                    batch_completed = sum(item_completed)

                    # 计算全局进度
                    global_completed = batch_start + batch_completed
                    progress = global_completed / total

                    self.batch_progress_text.value = f"批次 {batch_num}/{num_batches} - 已完成 {global_completed}/{total}"
                    self.batch_progress_bar.value = progress
                    self._page.update()

            # 合并每个文本的完整音频
            self.terminal.add_log("正在合并音频...")
            combined_audios = []
            for i, chunks in enumerate(item_chunks):
                if chunks:
                    # 过滤掉空数组
                    non_empty_chunks = [c for c in chunks if c.size > 0]
                    if non_empty_chunks:
                        combined = np.concatenate(non_empty_chunks)
                        combined_audios.append(combined)
                        self.terminal.add_log(f"  文本 {i+1}: {len(combined)/sample_rate:.2f}s")

            # 合并所有音频为一个文件
            if combined_audios:
                final_audio = np.concatenate(combined_audios)
                self._last_audio = (final_audio, sample_rate)

                if self._temp_audio_file:
                    self._audio_temp_manager.cleanup_file(self._temp_audio_file)

                self._temp_audio_file = self._audio_temp_manager.save_audio(
                    final_audio, sample_rate, prefix="batch"
                )

                self.batch_progress_text.value = f"✓ 批量生成完成: {total} 个文本, 总时长 {len(final_audio)/sample_rate:.2f}s"
                self.batch_progress_bar.value = 1.0

                self.terminal.add_log(f"✓ 批量语音生成成功: {total} 个文本")

                # 更新音频控制状态并播放
                self.audio_control.update_audio_state(True)
                await self._on_play(None)
            else:
                self.batch_progress_text.value = "✗ 生成失败: 没有有效的音频"
                self.terminal.add_log("✗ 批量生成失败: 没有生成任何音频")

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}", exc_info=True)
            self.batch_progress_text.value = f"✗ 生成失败: {str(e)}"
            self.terminal.add_log(f"✗ 批量生成失败: {str(e)}")
            raise
        finally:
            # 清理显存
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.terminal.add_log("已清理 GPU 显存")
            except Exception as cleanup_error:
                logger.warning(f"清理 GPU 显存失败: {cleanup_error}")

    async def _on_play(self, e):
        """播放音频"""
        if not self._temp_audio_file or not self._audio_temp_manager.file_exists(self._temp_audio_file):
            self.terminal.add_log("✗ 没有可播放的音频")
            return

        try:
            audio_manager = self.audio_manager_getter()

            # 设置进度回调
            async def progress_callback(p, c, t):
                self.audio_control.update_progress(p, c, t)
            audio_manager.set_progress_callback(progress_callback)

            # 设置播放完成回调
            async def completion_callback():
                self.audio_control.reset_progress()
            audio_manager.set_completion_callback(completion_callback)

            # 获取并设置时长
            if self._last_audio:
                audio_data, sr = self._last_audio
                duration = audio_manager.get_audio_duration(audio_data)
                self.audio_control.set_duration(duration)

            # 从文件播放
            await audio_manager.play_from_file(self._temp_audio_file)
            self.terminal.add_log("正在播放音频...")
        except Exception as e:
            logger.error(f"播放音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 播放失败: {str(e)}")

    async def _on_seek(self, e):
        """处理进度条拖动"""
        try:
            audio_manager = self.audio_manager_getter()
            progress = e.control.value

            if hasattr(audio_manager, '_audio_data') and audio_manager._audio_data is not None:
                duration = len(audio_manager._audio_data) / audio_manager.sample_rate
                position = progress * duration
                await audio_manager.seek(position)
        except Exception as e:
            logger.error(f"跳转失败: {str(e)}", exc_info=True)

    async def _on_stop(self, e):
        """停止播放"""
        try:
            audio_manager = self.audio_manager_getter()
            await audio_manager.stop()
            self.terminal.add_log("已停止播放")
        except Exception as e:
            logger.error(f"停止播放失败: {str(e)}", exc_info=True)

    async def _on_save(self, e):
        """保存音频"""
        if not self._temp_audio_file:
            self.terminal.add_log("✗ 没有可保存的音频")
            return

        try:
            # 获取保存路径
            save_dir = self.config_manager.get("audio.save_directory", "./output")

            # 获取自定义文件名（如果用户输入了）
            custom_filename = self.audio_filename_input.value.strip() if self.audio_filename_input.value else None

            # 使用临时文件管理器保存到持久化目录
            save_path = self._audio_temp_manager.save_to_persistent(
                self._temp_audio_file,
                save_dir,
                prefix="design",
                custom_filename=custom_filename
            )

            self.terminal.add_log(f"✓ 音频已保存: {save_path}")

            # 显示成功提示
            filename = os.path.basename(save_path)
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"音频已保存: {filename}"),
                bgcolor=ft.Colors.GREEN
            ))

        except Exception as e:
            logger.error(f"保存音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 保存失败: {str(e)}")
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"保存失败: {str(e)}"),
                bgcolor=ft.Colors.RED
            ))
