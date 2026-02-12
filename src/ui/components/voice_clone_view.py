"""
声音克隆 (Voice Clone) 页面

使用参考音频克隆声音
"""

import flet as ft
import logging
import asyncio
import os
import time
import numpy as np
from typing import List, Optional

from ui.components.shared_controls import create_labeled_control
from ui.components.audio_progress_bar import AudioProgressBar
from ui.components.voice_library import VoiceLibrary
from tts.audio_temp_manager import AudioTempManager
from tts.text_splitter import smart_split
from utils.time_utils import format_elapsed_time

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


class VoiceCloneView(ft.Container):
    """声音克隆页面"""

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
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.audio_manager_getter = audio_manager_getter
        self.terminal = terminal
        self.voice_library = voice_library
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.on_clear_engine_cache = on_clear_engine_cache

        # 当前生成的音频、克隆数据和临时文件路径
        self._last_audio = None
        self._temp_audio_file = None
        self._ref_audio_path = None
        self._is_generating = False

        # 上次保存的克隆信息（用于避免重复保存）
        self._last_saved_clone_ref_audio = None
        self._last_saved_clone_ref_text = None

        # 音频临时文件管理器
        self._audio_temp_manager = AudioTempManager()

        # 文件选择器 (FilePicker 不是控件，不需要添加到 overlay)
        self.file_picker = ft.FilePicker()

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
        # 模型选择下拉框 - 只显示 Base 模型（只有 Base 模型支持声音克隆）
        usable_models = self.model_manager.list_usable_models_by_type("base")
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

        # 参考音频选择
        self.ref_audio_button = ft.Button(
            "选择文件",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self._on_pick_file
        )

        self.ref_audio_status = ft.Text(
            "未选择文件",
            size=12
        )

        # 参考文本输入框
        self.ref_text_input = ft.TextField(
            label="参考文本",
            multiline=True,
            min_lines=3,
            max_lines=5,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            on_change=self._on_ref_text_change
        )

        # 克隆选项
        self.save_clone_checkbox = ft.Checkbox(
            label="保存为可重用克隆",
            value=False,
            on_change=self._on_save_clone_checkbox_change
        )

        self.clone_name_input = ft.TextField(
            label="克隆名称",
            hint_text="例如: 我的声音克隆",
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            visible=False,
            expand=True
        )

        # 音频文件名自定义输入框
        self.audio_filename_input = ft.TextField(
            label="音频文件名（可选，留空则自动生成）",
            hint_text="例如: 我的语音",
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            expand=True
        )

        self.x_vector_only_checkbox = ft.Checkbox(
            label="仅使用 x_vector (快速模式，质量可能降低)",
            value=False
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

        # 批量推理 ExpansionTile (左侧)
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


        # 克隆声音库
        self.clone_library_grid = ft.GridView(
            runs_count=3,
            max_extent=150,
            spacing=10,
            run_spacing=10,
            expand=False
        )
        self._refresh_clone_library()

        # 使用已保存克隆选择
        self.use_saved_clone_radio = ft.RadioGroup(
            content=ft.Column([
                ft.Radio(value="new", label="使用新音频（每次提取特征）"),
                ft.Radio(value="saved", label="使用已保存的克隆"),
            ]),
            value="new"
        )

        self.saved_clone_dropdown = ft.Dropdown(
            label="选择克隆",
            options=[],
            width=200,
            visible=False,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )
        # 将 on_change 作为属性设置，而不是构造函数参数
        self.saved_clone_dropdown.on_change = self._on_saved_clone_changed

        self.use_saved_clone_radio.on_change = self._on_clone_mode_change

        # 文本输入面板
        self.text_panel = TextPanel(
            placeholder="请输入要转换的文本...",
            min_lines=6,
            max_lines=10,
            on_clear=self._on_clear_text
        )

        # 音频控制面板
        self.audio_control = AudioControlPanel(
            on_play=self._on_play,
            on_stop=self._on_stop,
            on_save=self._on_save_audio,
            on_seek=self._on_seek,
            has_audio=False
        )

        # 控制面板
        control_panel = ft.Container(
            content=ft.Column(
                [
                    # 克隆选项
                    ft.Column([
                        ft.Text("克隆选项", size=14, weight=ft.FontWeight.BOLD),
                        ft.Column([
                            self.save_clone_checkbox,
                            self.clone_name_input,
                            self.x_vector_only_checkbox,
                        ], spacing=5),
                    ], spacing=5),

                    ft.Divider(),

                    # 克隆声音库
                    create_labeled_control("克隆声音库", self.clone_library_grid),

                    ft.Divider(),

                    # 克隆模式
                    ft.Column([
                        ft.Text("克隆模式", size=14, weight=ft.FontWeight.BOLD),
                        self.use_saved_clone_radio,
                        self.saved_clone_dropdown,
                    ], spacing=5),

                    ft.Divider(),

                    # 参考音频
                    ft.Column([
                        ft.Row([
                            ft.Text("参考音频", size=14, weight=ft.FontWeight.BOLD),
                            ft.Icon(
                                ft.Icons.INFO_OUTLINE,
                                size=16,
                                tooltip="选择参考音频文件并输入对应的文本"
                            )
                        ]),
                        ft.Row([
                            self.ref_audio_button,
                            self.ref_audio_status,
                        ], spacing=10),
                        self.ref_text_input,
                    ], spacing=5),

                    ft.Divider(),

                    # 音频控制

                    # 音频控制
                    self.audio_control,

                    ft.Divider(),

                    # 音频文件名设置
                    create_labeled_control("保存设置", self.audio_filename_input),
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

    def _on_saved_clone_changed(self, e):
        """已保存克隆选择改变事件"""
        self.terminal.add_log(f"已选择克隆: {self.saved_clone_dropdown.value}")

    def _on_clone_mode_change(self, e):
        """克隆模式切换事件"""
        mode = self.use_saved_clone_radio.value

        if mode == "saved":
            # 显示已保存克隆下拉框
            self._update_saved_clone_dropdown()
            self.saved_clone_dropdown.visible = True

            # 隐藏新音频相关控件
            self.ref_audio_button.visible = False
            self.ref_audio_status.visible = False
            self.ref_text_input.visible = False
        else:
            # 隐藏下拉框，显示音频选择
            self.saved_clone_dropdown.visible = False
            self.ref_audio_button.visible = True
            self.ref_audio_status.visible = True
            self.ref_text_input.visible = True

        self._page.update()

    def _on_save_clone_checkbox_change(self, e):
        """保存为克隆复选框变化事件"""
        self.clone_name_input.visible = e.control.value
        self._page.update()

    def _on_ref_text_change(self, _):
        """参考文本变化事件"""
        # 重置上次保存的克隆信息
        self._last_saved_clone_ref_audio = None
        self._last_saved_clone_ref_text = None

    def _on_batch_streaming_toggle(self, e):
        """批量推理开关切换事件"""
        enabled = e.control.value

        # 显示/隐藏批量进度
        self.batch_progress_text.visible = enabled
        self.batch_progress_bar.visible = enabled

        if enabled:
            self.batch_progress_text.value = "准备就绪"
            self.batch_progress_bar.value = 0

        self._page.update()
        self.terminal.add_log(f"批量推理: {'已启用' if enabled else '已禁用'}")

    def _update_saved_clone_dropdown(self):
        """更新已保存克隆下拉框"""
        clones = self.voice_library.get_all_clones()
        options = [
            ft.DropdownOption(
                text=c["name"],
                key=c["id"]
            ) for c in clones
        ]
        self.saved_clone_dropdown.options = options
        self.saved_clone_dropdown.update()

    def _refresh_clone_library(self):
        """刷新克隆声音库"""
        self.clone_library_grid.controls.clear()

        clones = self.voice_library.get_all_clones()

        if not clones:
            self.clone_library_grid.controls.append(
                ft.Text("暂无克隆", size=12)
            )
        else:
            for clone in clones:
                card = ft.Container(
                    content=ft.Column([
                        ft.Text(clone["name"], size=13, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            clone["created_at"][:10],
                            size=11
                        ),
                        ft.Row([
                            ft.IconButton(
                                icon=ft.Icons.PLAY_ARROW,
                                icon_size=18,
                                tooltip="试听",
                                on_click=lambda e, c=clone: self._on_preview_clone(e, c)
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE,
                                icon_size=18,
                                tooltip="删除",
                                on_click=lambda e, c=clone: self._on_delete_clone(e, c)
                            ),
                        ], spacing=5)
                    ], spacing=5),
                    padding=10,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                    border_radius=8,
                    width=140,
                    height=100
                )
                self.clone_library_grid.controls.append(card)

        # 只有在控件已添加到页面时才调用 update()
        try:
            self.clone_library_grid.update()
        except RuntimeError:
            pass

    def _on_preview_clone(self, e, clone: dict):
        """预览克隆声音"""
        self.terminal.add_log(f"试听克隆: {clone['name']}")
        # TODO: 实现试听功能

    def _on_delete_clone(self, e, clone: dict):
        """删除克隆声音"""
        success = self.voice_library.remove_clone(clone["id"])

        if success:
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"已删除: {clone['name']}"),
                bgcolor=ft.Colors.GREEN
            ))
            self.terminal.add_log(f"已删除克隆: {clone['name']}")
            self._refresh_clone_library()
        else:
            self._page.show_dialog(ft.SnackBar(
                ft.Text("删除失败"),
                bgcolor=ft.Colors.RED
            ))

    def _on_model_changed(self, _):
        """模型选择改变事件"""
        if self.on_clear_engine_cache:
            self.on_clear_engine_cache(self.model_dropdown.value)
        self.terminal.add_log(f"模型已切换: {self.model_dropdown.value}")

    def refresh_model_dropdown(self):
        """刷新模型下拉框选项"""
        try:
            # 获取当前选中的模型
            current_value = self.model_dropdown.value

            # 重新获取 Base 模型列表
            usable_models = self.model_manager.list_usable_models_by_type("base")
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

    def _on_clear_text(self, e):
        """清空文本"""
        self.text_panel.clear()

    async def _on_generate_batch(
        self,
        texts: List[str],
        clone_prompt,
        tts_engine,
        batch_size: int = 12
    ):
        """
        批量流式生成语音（支持分批处理以控制显存占用）

        Args:
            texts: 文本列表
            clone_prompt: 声音克隆提示
            tts_engine: TTS 引擎
            batch_size: 每批最大文本数（控制显存占用）
        """
        import torch

        total = len(texts)
        self.terminal.add_log(f"开始批量生成 {total} 个文本（每批最多 {batch_size} 个）...")

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
                async for chunks_list, sr in tts_engine.voice_clone_batch_stream_synthesize_async(
                    texts=batch_texts,
                    clone_prompt=clone_prompt,
                    language="Auto",
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
            # 无论成功还是异常，都清理显存（PDCA 循环 #2 修复）
            # 修复: finally 块中的异常处理，防止掩盖原始异常
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.terminal.add_log("已清理 GPU 显存")
            except Exception as cleanup_error:
                # 清理失败只记录日志，不掩盖原始异常
                logger.warning(f"清理 GPU 显存失败: {cleanup_error}")

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

        # 检查是否启用批量模式
        if self.batch_streaming_switch.value:
            await self._on_generate_with_batch(text)
        else:
            await self._on_generate_single(text)

    async def _on_generate_with_batch(self, text: str):
        """批量模式生成语音"""
        if self._is_generating:
            return

        self._is_generating = True
        start_time = time.perf_counter()  # 开始计时

        try:
            # 获取 TTS 引擎
            tts_engine = await self.tts_engine_getter()

            # 获取 clone_prompt
            clone_prompt = None
            clone_mode = self.use_saved_clone_radio.value

            if clone_mode == "saved":
                clone_id = self.saved_clone_dropdown.value
                if not clone_id:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("请选择要使用的克隆"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                clone = self.voice_library.get_clone(clone_id)
                if not clone:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("克隆不存在"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                # 使用预计算特征或提取新特征
                if "prompt_features" in clone and clone["prompt_features"]:
                    clone_prompt = clone["prompt_features"]
                    self.terminal.add_log(f"使用预计算特征: {clone['name']}")
                else:
                    self.terminal.add_log(f"首次使用克隆 '{clone['name']}'，正在提取特征...")
                    clone_prompt = await tts_engine.create_voice_clone_prompt_async(
                        ref_audio=clone["ref_audio"],
                        ref_text=clone["ref_text"],
                        x_vector_only=False
                    )
                    self.voice_library.update_clone_features(clone_id, clone_prompt)
                    self.terminal.add_log(f"✓ 特征已保存，下次可直接使用")
            else:
                # 使用新音频
                if not self._ref_audio_path:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("请选择参考音频"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                ref_text = self.ref_text_input.value or ""
                if not ref_text or not ref_text.strip():
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("请输入参考文本"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                x_vector_only = self.x_vector_only_checkbox.value
                self.terminal.add_log(f"正在提取声音特征: {os.path.basename(self._ref_audio_path)}")
                self._page.update()

                clone_prompt = await tts_engine.create_voice_clone_prompt_async(
                    ref_audio=self._ref_audio_path,
                    ref_text=ref_text.strip(),
                    x_vector_only=x_vector_only
                )

            # 分割文本
            split_mode = self.split_mode_dropdown.value
            texts = smart_split(text, mode=split_mode, language="chinese")

            if len(texts) <= 1:
                self.terminal.add_log("仅检测到单个文本，使用普通模式生成")
                # 单个文本，使用普通模式
                if len(texts) == 1:
                    audio, sr = await tts_engine.voice_clone_synthesize_async(
                        text=texts[0],
                        clone_prompt=clone_prompt,
                        timeout=300.0
                    )
                    self._last_audio = (audio, sr)
                    if self._temp_audio_file:
                        self._audio_temp_manager.cleanup_file(self._temp_audio_file)
                    self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sr, prefix="clone")
                    self.terminal.add_log("✓ 语音生成成功")
                    self.audio_control.update_audio_state(True)
                    await self._on_play(None)
            else:
                # 多个文本，使用批量模式（限制范围 1-64，防止显存溢出）
                try:
                    batch_size = int(self.batch_size_input.value)
                    if batch_size < 1 or batch_size > 64:
                        batch_size = 16
                except ValueError:
                    batch_size = 16
                await self._on_generate_batch(texts, clone_prompt, tts_engine, batch_size)

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}", exc_info=True)
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

    async def _on_generate_single(self, text: str):
        """单个文本生成（原有逻辑）"""
        if self._is_generating:
            return

        self._is_generating = True
        start_time = time.perf_counter()  # 开始计时

        try:
            clone_mode = self.use_saved_clone_radio.value

            if clone_mode == "saved":
                # 使用已保存的克隆
                clone_id = self.saved_clone_dropdown.value
                if not clone_id:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("请选择要使用的克隆"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                clone = self.voice_library.get_clone(clone_id)
                if not clone:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("克隆不存在"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                # 获取TTS引擎（需要在克隆分支中单独获取）
                tts_engine = await self.tts_engine_getter()

                # 检查是否有预计算的特征
                if "prompt_features" in clone and clone["prompt_features"]:
                    # 使用预计算的特征（快速）
                    self.terminal.add_log(f"使用预计算特征: {clone['name']}")

                    # 使用预计算特征生成语音（不需要 ref_audio 和 ref_text）
                    audio, sr = await tts_engine.voice_clone_synthesize_async(
                        text=text,
                        clone_prompt=clone["prompt_features"],
                        timeout=300.0
                    )

                    # 保存结果并返回
                    self._last_audio = (audio, sr)

                    if self._temp_audio_file:
                        self._audio_temp_manager.cleanup_file(self._temp_audio_file)

                    self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sr, prefix="clone")

                    self.terminal.add_log("✓ 语音生成成功（使用缓存特征）")

                    # 更新音频控制状态并播放
                    self.audio_control.update_audio_state(True)
                    await self._on_play(None)

                    return
                else:
                    # 首次使用或特征丢失，需要重新计算
                    self.terminal.add_log(f"首次使用克隆 '{clone['name']}'，正在提取特征...")

                    ref_audio = clone["ref_audio"]
                    ref_text = clone["ref_text"]

                    # 提取特征
                    prompt_features = await tts_engine.create_voice_clone_prompt_async(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        x_vector_only=False
                    )

                    # 保存特征以备下次使用
                    self.voice_library.update_clone_features(clone_id, prompt_features)
                    self.terminal.add_log(f"✓ 特征已保存，下次可直接使用")

                    # 使用特征生成语音（不需要 ref_audio 和 ref_text）
                    audio, sr = await tts_engine.voice_clone_synthesize_async(
                        text=text,
                        clone_prompt=prompt_features,
                        timeout=300.0
                    )

                    # 保存结果并返回
                    self._last_audio = (audio, sr)

                    if self._temp_audio_file:
                        self._audio_temp_manager.cleanup_file(self._temp_audio_file)

                    self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sr, prefix="clone")

                    self.terminal.add_log("✓ 语音生成成功")

                    # 更新音频控制状态并播放
                    self.audio_control.update_audio_state(True)
                    await self._on_play(None)

                    return
            else:
                # 使用新音频
                if not self._ref_audio_path:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("请选择参考音频"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                ref_text = self.ref_text_input.value or ""
                if not ref_text or not ref_text.strip():
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text("请输入参考文本"),
                        bgcolor=ft.Colors.RED
                    ))
                    return

                ref_audio = self._ref_audio_path
                ref_text = ref_text.strip()

            self.terminal.add_log("正在生成语音...")

            # 强制UI更新，让第一条日志立即显示
            try:
                self._page.update()
            except:
                pass

            # 获取TTS引擎（异步）
            tts_engine = await self.tts_engine_getter()

            # 获取参数
            x_vector_only = self.x_vector_only_checkbox.value

            # 生成语音
            self.terminal.add_log(f"参考音频: {os.path.basename(ref_audio)}")

            # 强制UI更新，让参数日志立即显示
            try:
                self._page.update()
            except:
                pass

            # 在后台线程中执行TTS生成（使用异步API）
            audio, sr = await tts_engine.voice_clone_synthesize_async(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only=x_vector_only,
                timeout=300.0
            )

            self.terminal.add_log("✓ 语音生成成功")

            # 使用临时文件管理器保存音频
            if self._temp_audio_file:
                self._audio_temp_manager.cleanup_file(self._temp_audio_file)

            self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sr, prefix="clone")

            # 保存音频数据用于计算时长
            self._last_audio = (audio, sr)

            # 检查是否需要保存为可重用克隆
            if self.save_clone_checkbox.value and clone_mode == "new":
                clone_name = self.clone_name_input.value or "未命名克隆"

                # 检查是否与上次保存的克隆相同
                if (self._last_saved_clone_ref_audio == ref_audio and
                    self._last_saved_clone_ref_text == ref_text):
                    self.terminal.add_log(f"ℹ 该克隆已存在，跳过保存")
                    logger.info(f"克隆已存在，跳过保存: ref_audio={ref_audio}, ref_text={ref_text}")
                else:
                    # 创建新克隆（包含特征计算）
                    self.terminal.add_log("正在提取声音特征并保存克隆...")

                    # 计算特征
                    prompt_features = await tts_engine.create_voice_clone_prompt_async(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        x_vector_only=x_vector_only
                    )

                    # 保存克隆（包含特征）
                    clone_id = self.voice_library.add_clone(
                        name=clone_name,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        prompt_features=prompt_features,  # 保存特征
                        x_vector_only=x_vector_only
                    )

                    if clone_id:
                        # 更新上次保存的克隆信息
                        self._last_saved_clone_ref_audio = ref_audio
                        self._last_saved_clone_ref_text = ref_text

                        self._page.show_dialog(ft.SnackBar(
                            ft.Text(f"已保存克隆: {clone_name}"),
                            bgcolor=ft.Colors.GREEN
                        ))
                        self.terminal.add_log(f"✓ 已保存克隆: {clone_name} ({clone_id})")
                        self._refresh_clone_library()

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

    async def _on_save_audio(self, e):
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
                prefix="clone",
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

    async def _on_pick_file(self, e):
        """文件选择按钮点击事件"""
        try:
            # 使用新的异步 API
            result = await self.file_picker.pick_files(
                allowed_extensions=["wav", "mp3", "flac"]
            )
            if result and len(result) > 0:
                self._ref_audio_path = result[0].path
                filename = os.path.basename(self._ref_audio_path)
                self.ref_audio_status.value = f"已选择: {filename}"
                self.ref_audio_status.update()
                self.terminal.add_log(f"已选择参考音频: {filename}")

                # 重置上次保存的克隆信息
                self._last_saved_clone_ref_audio = None
                self._last_saved_clone_ref_text = None
        except Exception as ex:
            logger.error(f"选择文件失败: {str(ex)}", exc_info=True)
            self.terminal.add_log(f"✗ 选择文件失败: {str(ex)}")





