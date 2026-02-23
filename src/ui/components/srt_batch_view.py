"""
SRT批量推理视图

参考TTS服务页面设计，支持三种TTS模式的批量推理
"""

import flet as ft
import logging
import os
from typing import Optional, Callable
from pathlib import Path

from tts.srt_batch_engine import SRTBatchEngine, SRTBatchResult
from tts.srt_config_models import (
    CustomVoiceConfig,
    VoiceDesignConfig,
    VoiceCloneConfig,
)
from ui.components.base.audio_control_panel import AudioControlPanel
from core.task_engine import get_task_engine, TaskType

logger = logging.getLogger(__name__)

TTS_MODE_CUSTOM_VOICE = "custom_voice"
TTS_MODE_VOICE_DESIGN = "voice_design"
TTS_MODE_VOICE_CLONE = "voice_clone"

MODE_LABELS = {
    TTS_MODE_CUSTOM_VOICE: "自定义语音",
    TTS_MODE_VOICE_DESIGN: "声音设计",
    TTS_MODE_VOICE_CLONE: "声音克隆",
}


class SRTBatchView(ft.Container):
    """SRT批量推理视图 - 参考TTS服务页面设计"""

    def __init__(
        self,
        page: ft.Page,
        tts_engine_getter: Callable,
        audio_manager_getter: Callable,
        terminal,
        voice_library,
        config_manager,
        model_manager,
        on_clear_engine_cache: Optional[Callable] = None,
        on_load_model: Optional[Callable] = None,
    ):
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.audio_manager_getter = audio_manager_getter
        self.terminal = terminal
        self.voice_library = voice_library
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.on_clear_engine_cache = on_clear_engine_cache
        self.on_load_model = on_load_model
        self.task_engine = get_task_engine()

        self.current_mode = TTS_MODE_VOICE_CLONE
        self.batch_engine: Optional[SRTBatchEngine] = None
        self._is_processing = False
        self._srt_file_path: Optional[str] = None
        self._last_output_path: Optional[str] = None

        # UI组件引用 - 初始化时创建实际控件
        self.result_text = ft.Text(
            "未开始", size=16, color=ft.Colors.GREY, weight=ft.FontWeight.BOLD
        )
        self.subtitle_list = ft.ListView(
            spacing=5,
            padding=10,
            height=180,
        )
        self.srt_file_text = ft.Text(
            "未选择文件", size=14, color=ft.Colors.GREY, italic=True
        )
        self.progress_ring = ft.ProgressRing(visible=False, width=40, height=40)
        self.progress_text = ft.Text("准备就绪", size=14)
        self.start_button = ft.ElevatedButton(
            "开始批量推理",
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.GREEN,
                color=ft.Colors.WHITE,
                padding=ft.padding.symmetric(horizontal=30, vertical=15),
            ),
            on_click=self._on_start_batch,
        )
        self.audio_filename_input = ft.TextField(
            hint_text="输入文件名（可选，留空则使用SRT文件名）",
            expand=True,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            content_padding=ft.padding.symmetric(horizontal=10, vertical=5),
        )

        # FAB (供main_ui使用)
        self._fab = ft.FloatingActionButton(
            icon=ft.Icons.PLAY_ARROW,
            bgcolor=ft.Colors.GREEN,
            on_click=self._on_start_batch,
            tooltip="开始批量推理",
            visible=False,  # SRT批量推理页面不使用FAB，使用自己的开始按钮
        )

        # 模型下拉框
        self.model_dropdown = ft.Dropdown(
            label="选择模型",
            options=[],
            width=250,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )
        self.design_model_dropdown = ft.Dropdown(
            label="选择模型",
            options=[],
            width=250,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )
        self.clone_model_dropdown = ft.Dropdown(
            label="选择模型",
            options=[],
            width=250,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )

        # 模式特定控件 - 初始化时创建
        speakers = self.voice_library.get_custom_voice_speakers()
        self.speaker_dropdown = ft.Dropdown(
            label="说话人",
            options=[ft.dropdown.Option(s) for s in speakers],
            value=speakers[0] if speakers else "Vivian",
            width=200,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )

        design_presets = (
            list(self.voice_library.get_all_design_presets().keys())
            if self.voice_library
            else []
        )
        self.design_preset_dropdown = ft.Dropdown(
            label="预设",
            options=[ft.dropdown.Option(p) for p in design_presets]
            if design_presets
            else [ft.dropdown.Option("无预设")],
            value=design_presets[0] if design_presets else None,
            width=200,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )

        clones = self.voice_library.get_all_clones()
        self.saved_clone_dropdown = ft.Dropdown(
            label="选择克隆",
            options=[ft.dropdown.Option(c["id"], c["name"]) for c in clones],
            width=250,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            visible=True,
        )

        self.clone_mode_radio = ft.RadioGroup(
            content=ft.Row(
                [
                    ft.Radio(value="new", label="新音频克隆"),
                    ft.Radio(value="saved", label="已保存克隆"),
                ],
                spacing=20,
            ),
            value="saved",
            on_change=self._on_clone_mode_change,
        )

        self.ref_audio_button = ft.ElevatedButton(
            "选择参考音频",
            icon=ft.Icons.AUDIO_FILE,
            on_click=self._on_pick_ref_audio,
            visible=False,
        )
        self.ref_audio_text = ft.Text(
            "未选择", size=12, color=ft.Colors.GREY, visible=False
        )
        self.ref_text_input = ft.TextField(
            label="参考文本",
            hint_text="输入参考音频对应的文本内容",
            multiline=True,
            min_lines=2,
            max_lines=3,
            visible=False,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )
        self.x_vector_checkbox = ft.Checkbox(
            label="仅使用x_vector（快速模式）",
            value=False,
            visible=False,
        )

        self.srt_file_picker = ft.FilePicker()
        self.ref_audio_picker = ft.FilePicker()

        # 存储参考音频路径
        self._ref_audio_path: Optional[str] = None

        # 刷新模型列表
        self._refresh_all_model_dropdowns()

        super().__init__(content=self.build(), expand=True)

    def build(self):
        """构建UI"""
        # 顶部标题
        header = ft.Row(
            [
                ft.Icon(ft.Icons.SUBTITLES, size=40, color=ft.Colors.BLUE),
                ft.Text("SRT字幕批量推理", size=24, weight=ft.FontWeight.BOLD),
            ],
            spacing=15,
        )

        # 状态卡片
        status_card = self._build_status_card()

        # 文件选择区域
        file_section = self._build_file_section()

        # 模式选择 - 使用TabBarView
        mode_section = self._build_mode_section()

        # 进度显示
        progress_section = self._build_progress_section()

        # 音频控制
        self.audio_control = AudioControlPanel(
            on_play=self._on_play,
            on_stop=self._on_stop,
            on_save=self._on_save,
            on_seek=self._on_seek,
            has_audio=False,
        )

        return ft.Column(
            [
                header,
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                file_section,
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                mode_section,
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                progress_section,
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                self.audio_control,
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                status_card,
            ],
            expand=True,
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
        )

    def _build_status_card(self) -> ft.Container:
        """构建状态卡片"""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.INFO_OUTLINE, size=30, color=ft.Colors.BLUE
                            ),
                            ft.Text("处理状态", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            ft.Text("状态:", size=14, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                        ]
                    ),
                    self.result_text,
                    ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
                    ft.Text("字幕列表:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=self.subtitle_list,
                        bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                        border_radius=8,
                        padding=10,
                        expand=True,
                    ),
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY),
            height=320,
        )

    def _build_file_section(self) -> ft.Container:
        """构建文件选择区域"""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.FOLDER_OPEN, size=30, color=ft.Colors.PURPLE
                            ),
                            ft.Text("文件选择", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    # SRT文件选择
                    ft.Column(
                        [
                            ft.Text("SRT字幕文件:", size=14, weight=ft.FontWeight.BOLD),
                            ft.Row(
                                [
                                    ft.ElevatedButton(
                                        "选择文件",
                                        icon=ft.Icons.UPLOAD_FILE,
                                        on_click=self._on_pick_srt_file,
                                    ),
                                    self.srt_file_text,
                                ],
                                spacing=10,
                            ),
                        ],
                        spacing=5,
                    ),
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

    def _build_mode_section(self) -> ft.Container:
        """构建模式选择区域 - 参考TTS服务页面的Tabs"""
        mode_tabs = ft.Tabs(
            selected_index=2,  # 默认VoiceClone
            length=3,
            animation_duration=300,
            content=ft.Column(
                [
                    ft.TabBar(
                        tab_alignment=ft.TabAlignment.START,
                        indicator_color=ft.Colors.BLUE,
                        tabs=[
                            ft.Tab(icon=ft.Icons.MIC, label="自定义语音"),
                            ft.Tab(icon=ft.Icons.PALETTE, label="声音设计"),
                            ft.Tab(icon=ft.Icons.CONTENT_COPY, label="声音克隆"),
                        ],
                    ),
                    ft.Container(
                        content=ft.TabBarView(
                            [
                                ft.Container(
                                    content=self._build_custom_voice_panel(),
                                    padding=15,
                                ),
                                ft.Container(
                                    content=self._build_voice_design_panel(),
                                    padding=15,
                                ),
                                ft.Container(
                                    content=self._build_voice_clone_panel(),
                                    padding=15,
                                ),
                            ],
                        ),
                        bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                        border_radius=8,
                        height=200,
                    ),
                ]
            ),
            on_change=self._on_mode_change,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.TUNE, size=30, color=ft.Colors.PURPLE),
                            ft.Text("推理模式", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    mode_tabs,
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

    def _build_custom_voice_panel(self) -> ft.Column:
        """构建自定义语音配置面板"""
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.STORAGE, size=20, color=ft.Colors.TEAL),
                        self.model_dropdown,
                    ],
                    spacing=10,
                ),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                ft.Row(
                    [
                        ft.Icon(ft.Icons.PERSON, size=20, color=ft.Colors.BLUE),
                        self.speaker_dropdown,
                    ],
                    spacing=10,
                ),
            ],
            spacing=10,
        )

    def _build_voice_design_panel(self) -> ft.Column:
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.STORAGE, size=20, color=ft.Colors.TEAL),
                        self.design_model_dropdown,
                    ],
                    spacing=10,
                ),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                ft.Row(
                    [
                        ft.Icon(ft.Icons.PALETTE, size=20, color=ft.Colors.PURPLE),
                        self.design_preset_dropdown,
                    ],
                    spacing=10,
                ),
            ],
            spacing=10,
        )

    def _build_voice_clone_panel(self) -> ft.Column:
        """构建声音克隆配置面板"""
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.STORAGE, size=20, color=ft.Colors.TEAL),
                        self.clone_model_dropdown,
                    ],
                    spacing=10,
                ),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                self.clone_mode_radio,
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                self.saved_clone_dropdown,
                ft.Row(
                    [self.ref_audio_button, self.ref_audio_text],
                    spacing=10,
                ),
                self.ref_text_input,
                self.x_vector_checkbox,
            ],
            spacing=5,
        )

    def _build_progress_section(self) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.AUTORENEW, size=30, color=ft.Colors.ORANGE
                            ),
                            ft.Text("批处理控制", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            self.progress_ring,
                            self.progress_text,
                            ft.Container(expand=True),
                            self.start_button,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        spacing=15,
                    ),
                    ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.SAVE, size=20, color=ft.Colors.TEAL),
                            self.audio_filename_input,
                        ],
                        spacing=10,
                    ),
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

    # ==================== 事件处理 ====================

    def _on_mode_change(self, e: ft.ControlEvent):
        """模式切换事件"""
        selected = (
            e.control.selected_index if hasattr(e.control, "selected_index") else 2
        )
        mode_map = {
            0: TTS_MODE_CUSTOM_VOICE,
            1: TTS_MODE_VOICE_DESIGN,
            2: TTS_MODE_VOICE_CLONE,
        }
        self.current_mode = mode_map.get(selected, TTS_MODE_VOICE_CLONE)
        self.terminal.add_log(f"切换到 {MODE_LABELS.get(self.current_mode)} 模式")

    def _on_clone_mode_change(self, e: ft.ControlEvent):
        """克隆模式切换"""
        is_new = e.control.value == "new" if hasattr(e.control, "value") else False
        self.saved_clone_dropdown.visible = not is_new
        self.ref_audio_button.visible = is_new
        self.ref_audio_text.visible = is_new
        self.ref_text_input.visible = is_new
        self.x_vector_checkbox.visible = is_new
        try:
            self._page.update()
        except RuntimeError:
            pass
        except Exception as ex:
            logger.warning(f"UI更新失败: {ex}")

    async def _on_pick_srt_file(self, e):
        """选择SRT文件"""
        try:
            result = await self.srt_file_picker.pick_files(
                dialog_title="选择SRT字幕文件",
                allowed_extensions=["srt"],
            )
            if result and len(result) > 0:
                self._srt_file_path = result[0].path
                filename = os.path.basename(self._srt_file_path)
                self.srt_file_text.value = filename
                self.srt_file_text.color = ft.Colors.GREEN
                self.terminal.add_log(f"已选择SRT文件: {filename}")
                self._load_subtitle_preview()
        except Exception as ex:
            logger.error(f"选择SRT文件失败: {ex}")

    def _load_subtitle_preview(self):
        """加载字幕预览"""
        if not self._srt_file_path:
            return

        try:
            from tts.srt_parser import SRTParser

            parser = SRTParser()
            entries = parser.parse_file(self._srt_file_path)

            self.subtitle_list.controls.clear()
            for entry in entries[:20]:
                text_display = (
                    entry.text[:50] + "..." if len(entry.text) > 50 else entry.text
                )
                item = ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                f"{entry.index}. {entry.start_time:.2f}s - {entry.end_time:.2f}s",
                                size=11,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Text(
                                text_display,
                                size=11,
                            ),
                        ],
                        spacing=2,
                    ),
                    padding=5,
                    bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                    border_radius=4,
                )
                self.subtitle_list.controls.append(item)

            if len(entries) > 20:
                self.subtitle_list.controls.append(
                    ft.Text(
                        f"... 还有 {len(entries) - 20} 条字幕", size=11, italic=True
                    )
                )

            self.result_text.value = f"共 {len(entries)} 条字幕"
            self.result_text.color = ft.Colors.BLUE

            try:
                self.subtitle_list.update()
                self.result_text.update()
            except RuntimeError:
                pass

        except Exception as e:
            self.terminal.add_log(f"加载字幕预览失败: {e}")

    async def _on_pick_ref_audio(self, e):
        """选择参考音频"""
        try:
            result = await self.ref_audio_picker.pick_files(
                dialog_title="选择参考音频",
                allowed_extensions=["wav", "mp3", "flac"],
            )
            if result and len(result) > 0:
                self._ref_audio_path = result[0].path
                filename = os.path.basename(self._ref_audio_path)
                self.ref_audio_text.value = filename
                self.ref_audio_text.color = ft.Colors.GREEN
                self.terminal.add_log(f"已选择参考音频: {filename}")
                try:
                    self.ref_audio_text.update()
                except RuntimeError:
                    pass
        except Exception as ex:
            logger.error(f"选择参考音频失败: {ex}")

    def _get_selected_model_id(self) -> Optional[str]:
        if self.current_mode == TTS_MODE_CUSTOM_VOICE:
            return self.model_dropdown.value
        elif self.current_mode == TTS_MODE_VOICE_DESIGN:
            return self.design_model_dropdown.value
        elif self.current_mode == TTS_MODE_VOICE_CLONE:
            return self.clone_model_dropdown.value
        return None

    def _get_loaded_model_id(self) -> Optional[str]:
        return self.task_engine.get_loaded_model_id()

    async def _ensure_model_loaded(self) -> bool:
        selected_model = self._get_selected_model_id()
        if not selected_model:
            self._show_snack_bar("请先选择模型", ft.Colors.RED)
            return False

        loaded_model = self._get_loaded_model_id()

        if loaded_model and loaded_model != selected_model:
            self.terminal.add_log(f"切换模型: {loaded_model} -> {selected_model}")
            self.progress_text.value = "正在卸载旧模型..."
            try:
                self._page.update()
            except:
                pass

            engine = self.tts_engine_getter()
            if engine:
                if hasattr(engine, "_engine"):
                    inner_engine = engine._engine
                else:
                    inner_engine = engine

                if hasattr(inner_engine, "unload"):

                    def sync_unload():
                        inner_engine.unload()

                    await self.task_engine.submit(
                        task_type=TaskType.UNLOAD,
                        func=sync_unload,
                        description=f"卸载模型: {loaded_model}",
                        priority=15,
                    )

            # 注意：卸载已通过 task_engine 完成，无需再调用 on_clear_engine_cache
            # 否则会触发异步卸载，可能把刚加载的新模型也卸载掉
            # 但需要清除 task_engine 中的模型记录，确保 on_load_model 会重新加载
            self.task_engine._loaded_model_id = None

            self.progress_text.value = "正在加载新模型..."
            try:
                self._page.update()
            except:
                pass

            if self.on_load_model:
                success = await self.on_load_model(self.current_mode)
                if not success:
                    self._show_snack_bar("模型加载失败", ft.Colors.RED)
                    return False
            else:
                engine = await self.tts_engine_getter()
                if engine is None:
                    self._show_snack_bar("模型加载失败", ft.Colors.RED)
                    return False

        elif loaded_model is None:
            self.progress_text.value = "正在加载模型..."
            try:
                self._page.update()
            except:
                pass

            if self.on_load_model:
                success = await self.on_load_model(self.current_mode)
                if not success:
                    self._show_snack_bar("模型加载失败", ft.Colors.RED)
                    return False
            else:
                engine = await self.tts_engine_getter()
                if engine is None:
                    self._show_snack_bar("模型加载失败", ft.Colors.RED)
                    return False

        return True

    async def _on_start_batch(self, e: ft.ControlEvent):
        if self._is_processing:
            return

        if not self._srt_file_path:
            self._show_snack_bar("请先选择SRT文件", ft.Colors.RED)
            return

        config = self._get_config_for_mode()
        if config is None:
            return

        self._is_processing = True
        self.start_button.disabled = True
        self.progress_ring.visible = True
        self.progress_text.value = "检查模型..."

        try:
            self._page.update()
        except:
            pass

        try:
            if not await self._ensure_model_loaded():
                self._is_processing = False
                self.start_button.disabled = False
                self.progress_ring.visible = False
                self.progress_text.value = "模型加载失败"
                try:
                    self._page.update()
                except:
                    pass
                return

            self.progress_text.value = "初始化引擎..."
            try:
                self._page.update()
            except:
                pass

            tts_engine = await self.tts_engine_getter()
            if tts_engine is None:
                raise RuntimeError("TTS引擎未初始化")

            # 创建批量推理引擎
            self.batch_engine = SRTBatchEngine(tts_engine)
            self.batch_engine.set_mode(self.current_mode, config)

            temp_dir = Path.cwd() / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            srt_name = Path(self._srt_file_path).stem
            output_path = str(temp_dir / f"{srt_name}_dubbed.wav")

            self.progress_text.value = "开始批量推理..."
            try:
                self._page.update()
            except:
                pass

            # 进度回调
            def progress_callback(current: int, total: int, text: str):
                self.progress_text.value = (
                    f"生成中... {current}/{total}: {text[:30]}..."
                )
                try:
                    self._page.update()
                except:
                    pass

            # 执行批量推理
            result = await self.batch_engine.process_srt(
                srt_file_path=self._srt_file_path,
                output_path=output_path,
                progress_callback=progress_callback,
                auto_adjust=True,
            )

            # 处理结果
            if result.success:
                self._last_output_path = result.output_path
                self.result_text.value = (
                    f"完成! 生成 {result.generated_count}/{result.total_entries} 条, "
                    f"总时长 {result.total_duration:.2f}s"
                )
                self.result_text.color = ft.Colors.GREEN
                self.progress_text.value = (
                    f"输出: {os.path.basename(result.output_path)}"
                )
                self.terminal.add_log(f"批量推理完成: {result.output_path}")
                self.terminal.add_log(f"  总条目: {result.total_entries}")
                self.terminal.add_log(f"  成功: {result.generated_count}")
                self.terminal.add_log(f"  失败: {result.failed_count}")
                self.terminal.add_log(f"  总时长: {result.total_duration:.2f}s")

                if self.audio_control:
                    self.audio_control.has_audio = True
                    self.audio_control.play_button.disabled = False
                    self.audio_control.save_button.disabled = False

                # 显示调整摘要
                adjusted_count = result.adjustment_summary.get("adjusted_count", 0)
                if adjusted_count > 0:
                    max_delay = result.adjustment_summary.get("max_delay", 0)
                    self.terminal.add_log(
                        f"  时间调整: {adjusted_count} 条, 最大延后 {max_delay:.2f}s"
                    )
            else:
                self.result_text.value = f"失败: {result.error_message}"
                self.result_text.color = ft.Colors.RED
                self.progress_text.value = "推理失败"
                self.terminal.add_log(f"批量推理失败: {result.error_message}")

        except Exception as e:
            logger.exception("批量推理失败")
            self.result_text.value = f"错误: {str(e)}"
            self.result_text.color = ft.Colors.RED
            self.progress_text.value = "发生错误"
            self.terminal.add_log(f"批量推理错误: {e}")

        finally:
            self._is_processing = False
            self.start_button.disabled = False
            self.progress_ring.visible = False

            # 清理 batch_engine 和音频数据，释放显存
            if self.batch_engine:
                # 清空音频数据列表，释放 numpy array 内存
                if hasattr(self.batch_engine, 'generated_audios'):
                    self.batch_engine.generated_audios.clear()
                if hasattr(self.batch_engine, 'audio_durations'):
                    self.batch_engine.audio_durations.clear()
                if hasattr(self.batch_engine, 'scheduler'):
                    # 清理 scheduler 中的音频数据
                    scheduler = self.batch_engine.scheduler
                    if hasattr(scheduler, 'scheduled') and scheduler.scheduled:
                        for entry in scheduler.scheduled:
                            if hasattr(entry, 'audio_data') and entry.audio_data is not None:
                                entry.audio_data = None
                self.batch_engine = None

            # 强制垃圾回收，释放显存
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

            try:
                self._page.update()
            except:
                pass

    def _get_config_for_mode(self):
        """获取当前模式的配置"""
        if self.current_mode == TTS_MODE_CUSTOM_VOICE:
            speaker_val = self.speaker_dropdown.value
            return CustomVoiceConfig(
                speaker=speaker_val if speaker_val else "Vivian",
            )

        elif self.current_mode == TTS_MODE_VOICE_DESIGN:
            preset = self.design_preset_dropdown.value
            if not preset or preset == "无预设":
                self._show_snack_bar("请选择预设", ft.Colors.RED)
                return None

            design_prompt = self.voice_library.get_design_preset(preset) or preset
            return VoiceDesignConfig(design_prompt=design_prompt)

        elif self.current_mode == TTS_MODE_VOICE_CLONE:
            clone_mode = self.clone_mode_radio.value
            if clone_mode == "saved":
                clone_id = self.saved_clone_dropdown.value
                if not clone_id:
                    self._show_snack_bar("请选择已保存的克隆", ft.Colors.RED)
                    return None

                clone_data = self.voice_library.get_clone(clone_id)
                if not clone_data:
                    self._show_snack_bar("克隆数据不存在", ft.Colors.RED)
                    return None

                return VoiceCloneConfig(
                    mode="saved",
                    clone_id=clone_id,
                    clone_prompt=clone_data.get("prompt_features"),
                )
            else:
                if not self._ref_audio_path:
                    self._show_snack_bar("请选择参考音频", ft.Colors.RED)
                    return None

                ref_text = self.ref_text_input.value or ""
                if not ref_text.strip():
                    self._show_snack_bar("请输入参考文本", ft.Colors.RED)
                    return None

                x_vector_val = self.x_vector_checkbox.value
                return VoiceCloneConfig(
                    mode="new",
                    ref_audio_path=self._ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only=x_vector_val if x_vector_val else False,
                )

        return None

    def _show_snack_bar(self, message: str, color: str):
        """显示提示消息"""
        try:
            self._page.show_dialog(ft.SnackBar(ft.Text(message), bgcolor=color))
        except:
            pass

    # ==================== 模型下拉框刷新 ====================

    def _refresh_all_model_dropdowns(self):
        """刷新所有模型下拉框"""
        self._refresh_model_dropdown()
        self._refresh_design_model_dropdown()
        self._refresh_clone_model_dropdown()

    def _refresh_model_dropdown(self):
        """刷新自定义语音模型下拉框"""
        models = self.model_manager.list_usable_models_by_type("customvoice")
        options = []
        for model_id in models:
            info = self.model_manager.get_model_info(model_id)
            name = info.name if info else model_id
            options.append(ft.dropdown.Option(model_id, name))

        self.model_dropdown.options = options
        if options:
            self.model_dropdown.value = models[0]

    def _refresh_design_model_dropdown(self):
        """刷新声音设计模型下拉框"""
        models = self.model_manager.list_usable_models_by_type("voicedesign")
        options = []
        for model_id in models:
            info = self.model_manager.get_model_info(model_id)
            name = info.name if info else model_id
            options.append(ft.dropdown.Option(model_id, name))

        self.design_model_dropdown.options = options
        if options:
            self.design_model_dropdown.value = models[0]

    def _refresh_clone_model_dropdown(self):
        """刷新声音克隆模型下拉框"""
        models = self.model_manager.list_usable_models_by_type("base")
        options = []
        for model_id in models:
            info = self.model_manager.get_model_info(model_id)
            name = info.name if info else model_id
            options.append(ft.dropdown.Option(model_id, name))

        self.clone_model_dropdown.options = options
        if options:
            self.clone_model_dropdown.value = models[0]

    # ==================== 音频控制 ====================

    async def _on_play(self, e):
        if not self._last_output_path or not os.path.exists(self._last_output_path):
            self.terminal.add_log("没有可播放的音频")
            return

        try:
            import soundfile as sf

            audio_manager = self.audio_manager_getter()

            async def progress_callback(p, c, t):
                self.audio_control.update_progress(p, c, t)

            audio_manager.set_progress_callback(progress_callback)

            async def completion_callback():
                self.audio_control.reset_progress()

            audio_manager.set_completion_callback(completion_callback)

            audio_data, sr = sf.read(self._last_output_path, dtype="float32")
            duration = audio_manager.get_audio_duration(audio_data)
            self.audio_control.set_duration(duration)

            await audio_manager.play_from_file(self._last_output_path)
            self.terminal.add_log("正在播放音频...")
        except Exception as e:
            logger.error(f"播放失败: {e}")
            self.terminal.add_log(f"播放失败: {e}")

    async def _on_stop(self, e):
        """停止播放"""
        try:
            audio_manager = self.audio_manager_getter()
            await audio_manager.stop()
            self.terminal.add_log("已停止播放")
        except Exception as e:
            logger.error(f"停止失败: {e}")

    async def _on_save(self, e):
        if not self._last_output_path or not os.path.exists(self._last_output_path):
            self.terminal.add_log("没有可保存的音频")
            return

        try:
            import shutil

            save_dir = self.config_manager.get("audio.save_directory", "./output")
            os.makedirs(save_dir, exist_ok=True)

            custom_filename = (
                self.audio_filename_input.value.strip()
                if self.audio_filename_input.value
                else None
            )

            if custom_filename:
                base_name = custom_filename
                if not base_name.endswith(".wav"):
                    base_name += ".wav"
            else:
                srt_name = (
                    Path(self._srt_file_path).stem if self._srt_file_path else "output"
                )
                base_name = f"{srt_name}_dubbed.wav"

            dest_path = os.path.join(save_dir, base_name)

            counter = 1
            while os.path.exists(dest_path):
                name_without_ext = base_name.rsplit(".", 1)[0]
                dest_path = os.path.join(save_dir, f"{name_without_ext}_{counter}.wav")
                counter += 1

            shutil.copy2(self._last_output_path, dest_path)

            self.terminal.add_log(f"音频已保存: {dest_path}")
            self._show_snack_bar(
                f"已保存: {os.path.basename(dest_path)}", ft.Colors.GREEN
            )
        except Exception as e:
            logger.error(f"保存失败: {e}")
            self.terminal.add_log(f"保存失败: {e}")

    async def _on_seek(self, e):
        try:
            audio_manager = self.audio_manager_getter()
            progress = e.control.value

            if (
                hasattr(audio_manager, "_audio_data")
                and audio_manager._audio_data is not None
            ):
                duration = len(audio_manager._audio_data) / audio_manager.sample_rate
                position = progress * duration
                await audio_manager.seek(position)
        except Exception as ex:
            logger.error(f"跳转失败: {ex}")

    def refresh_model_dropdown(self):
        """刷新所有模型下拉框（外部接口）"""
        self._refresh_all_model_dropdowns()
        try:
            self.model_dropdown.update()
            self.design_model_dropdown.update()
            self.clone_model_dropdown.update()
        except:
            pass
