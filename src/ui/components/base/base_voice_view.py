"""
语音生成视图基类

提供三个语音生成页面的共享逻辑：
- 模型管理
- 音频播放控制
- 批量推理
- 音频保存
"""

import flet as ft
import logging
import time
import gc
import numpy as np
import torch
from abc import abstractmethod
from typing import Optional, Tuple, List, Callable, Any

from ui.components.base.text_panel import TextPanel
from ui.components.base.audio_control_panel import AudioControlPanel
from ui.components.shared_controls import create_labeled_control, create_advanced_options_tile
from tts.audio_temp_manager import AudioTempManager
from tts.text_splitter import smart_split
from utils.time_utils import format_elapsed_time

logger = logging.getLogger(__name__)


class BaseVoiceView(ft.Container):
    """
    语音生成视图基类

    子类需要实现：
    - _get_model_type() -> str: 返回模型类型
    - _build_specific_controls() -> list: 构建特有UI
    - _get_save_prefix() -> str: 获取保存文件前缀
    - _on_generate_single_impl(): 单文本生成实现
    - _on_generate_batch_impl(): 批量文本生成实现
    """

    def __init__(
        self,
        page: ft.Page,
        tts_engine_getter: Callable,
        audio_manager_getter: Callable,
        terminal,
        voice_library,
        config_manager,
        model_manager,
        on_clear_engine_cache: Optional[Callable] = None
    ):
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.audio_manager_getter = audio_manager_getter
        self.terminal = terminal
        self.voice_library = voice_library
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.on_clear_engine_cache = on_clear_engine_cache

        # 音频状态
        self._last_audio: Optional[Tuple[np.ndarray, int]] = None
        self._temp_audio_file: Optional[str] = None
        self._is_generating: bool = False

        # 音频临时文件管理器
        self._audio_temp_manager = AudioTempManager()

        # 创建 FAB
        self._fab = ft.FloatingActionButton(
            icon=ft.Icons.SEND,
            bgcolor=ft.Colors.BLUE,
            on_click=self._on_generate,
            tooltip="生成语音",
        )

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

    # ==================== 抽象方法 ====================

    @abstractmethod
    def _get_model_type(self) -> str:
        """返回模型类型（如 'customvoice', 'base', 'voicedesign'）"""
        pass

    @abstractmethod
    def _build_specific_controls(self) -> list:
        """构建右侧控制面板的特有部分，返回控件列表"""
        pass

    @abstractmethod
    def _get_save_prefix(self) -> str:
        """获取保存文件前缀"""
        pass

    @abstractmethod
    async def _on_generate_single_impl(self, text: str, tts_engine) -> None:
        """单个文本生成的具体实现"""
        pass

    @abstractmethod
    async def _on_generate_batch_impl(self, texts: List[str], tts_engine, batch_size: int) -> None:
        """批量生成的具体实现"""
        pass

    # ==================== 可选覆盖的方法 ====================

    def _on_batch_streaming_toggle(self, e):
        """批量推理开关切换事件"""
        enabled = e.control.value
        self.batch_progress_text.visible = enabled
        self.batch_progress_bar.visible = enabled
        if enabled:
            self.batch_progress_text.value = "准备就绪"
            self.batch_progress_bar.value = 0
        self._page.update()
        self.terminal.add_log(f"批量推理: {'已启用' if enabled else '已禁用'}")

    # ==================== UI 构建 ====================

    def _build_ui(self):
        """构建UI界面"""
        # 模型选择下拉框
        self._build_model_dropdown()

        # 文本输入面板
        self.text_panel = TextPanel(
            placeholder="请输入要转换的文本...",
            min_lines=6,
            max_lines=10,
            on_clear=self._on_clear_text
        )

        # 批量推理控件
        self._build_batch_controls()

        # 高级选项
        self.advanced_options_tile = create_advanced_options_tile()

        # 批量推理 ExpansionTile
        self.batch_inference_tile = self._build_batch_inference_tile()

        # 音频控制面板
        self.audio_control = AudioControlPanel(
            on_play=self._on_play,
            on_stop=self._on_stop,
            on_save=self._on_save,
            on_seek=self._on_seek,
            has_audio=False
        )

        # 音频文件名输入框
        self.audio_filename_input = ft.TextField(
            label="音频文件名（可选，留空则自动生成）",
            hint_text="例如: 我的语音",
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            expand=True
        )

        # 左侧面板
        left_panel = self._build_left_panel()

        # 右侧控制面板
        control_panel = self._build_control_panel()

        # 主布局
        return ft.Row(
            [left_panel, control_panel],
            spacing=20,
            expand=True
        )

    def _build_model_dropdown(self):
        """构建模型选择下拉框"""
        model_type = self._get_model_type()
        usable_models = self.model_manager.list_usable_models_by_type(model_type)
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

    def _build_batch_controls(self):
        """构建批量推理控件"""
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

    def _build_batch_inference_tile(self) -> ft.ExpansionTile:
        """构建批量推理 ExpansionTile"""
        return ft.ExpansionTile(
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
                    ft.Text("按行分割: 每行一个文本\n按句分割: 自动识别句子边界",
                           size=11, color=ft.Colors.with_opacity(0.7, ft.Colors.ON_SURFACE)),
                    self.batch_progress_text,
                    self.batch_progress_bar,
                ], spacing=5),
            ],
        )

    def _build_left_panel(self) -> ft.Container:
        """构建左侧面板"""
        return ft.Container(
            content=ft.Column([
                create_labeled_control("模型选择", self.model_dropdown),
                ft.Divider(),
                ft.Text("文本输入", size=16, weight=ft.FontWeight.BOLD),
                self.text_panel,
                ft.Divider(),
                self.advanced_options_tile,
                self.batch_inference_tile,
            ], spacing=10, scroll=ft.ScrollMode.AUTO),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
            border_radius=12,
            expand=True
        )

    def _build_control_panel(self) -> ft.Container:
        """构建右侧控制面板"""
        specific_controls = self._build_specific_controls()

        controls = [
            *specific_controls,
            ft.Divider(),
            self.audio_control,
            ft.Divider(),
            create_labeled_control("保存设置", self.audio_filename_input),
        ]

        return ft.Container(
            content=ft.Column(controls, spacing=10, scroll=ft.ScrollMode.AUTO),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
            border_radius=12,
            width=380
        )

    # ==================== 共享方法 ====================

    def _on_model_changed(self, e):
        """模型选择改变事件"""
        if self.on_clear_engine_cache:
            self.on_clear_engine_cache(self.model_dropdown.value)
        self.terminal.add_log(f"模型已切换: {self.model_dropdown.value}")

    def refresh_model_dropdown(self):
        """刷新模型下拉框选项"""
        try:
            current_value = self.model_dropdown.value
            model_type = self._get_model_type()
            usable_models = self.model_manager.list_usable_models_by_type(model_type)

            model_options = []
            for model_id in usable_models:
                model_info = self.model_manager.get_model_info(model_id)
                if model_info:
                    model_options.append(ft.dropdown.Option(model_id, model_info.name))

            self.model_dropdown.options = model_options

            if current_value in usable_models:
                self.model_dropdown.value = current_value
            elif usable_models:
                self.model_dropdown.value = usable_models[0]
            else:
                self.model_dropdown.value = None

            self.model_dropdown.disabled = len(usable_models) == 0
            self.model_dropdown.update()
        except Exception as e:
            logger.error(f"刷新模型下拉框失败: {str(e)}", exc_info=True)

    def _on_clear_text(self, _):
        """清空文本"""
        self.text_panel.clear()

    # ==================== 生成逻辑 ====================

    async def _on_generate(self, e):
        """生成语音按钮点击事件"""
        if self._is_generating:
            return

        text = self.text_panel.get_text()
        if not text or not text.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请输入要转换的文本"),
                bgcolor=ft.Colors.RED
            ))
            return

        if self.batch_streaming_switch.value:
            await self._on_generate_with_batch(text)
        else:
            await self._on_generate_single(text)

    async def _on_generate_single(self, text: str):
        """单个文本生成"""
        self._is_generating = True
        start_time = time.perf_counter()
        self.terminal.add_log("正在生成语音...")

        try:
            self._page.update()
            tts_engine = await self.tts_engine_getter()
            await self._on_generate_single_impl(text, tts_engine)
        except Exception as e:
            logger.error(f"生成语音失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"生成失败: {str(e)}")
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"生成失败: {str(e)}"),
                bgcolor=ft.Colors.RED
            ))
        finally:
            self._is_generating = False
            elapsed_time = time.perf_counter() - start_time
            self.terminal.add_log(f"语音生成完成 (用时: {format_elapsed_time(elapsed_time)})")

    async def _on_generate_with_batch(self, text: str):
        """批量模式生成语音"""
        if self._is_generating:
            return

        self._is_generating = True
        start_time = time.perf_counter()

        try:
            split_mode = self.split_mode_dropdown.value
            texts = smart_split(text, mode=split_mode, language="chinese")

            if not texts:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("没有有效的文本可生成"),
                    bgcolor=ft.Colors.RED
                ))
                return

            try:
                batch_size = int(self.batch_size_input.value)
                batch_size = max(1, min(batch_size, 64))
            except ValueError:
                batch_size = 16

            tts_engine = await self.tts_engine_getter()
            await self._on_generate_batch_impl(texts, tts_engine, batch_size)

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"批量生成失败: {str(e)}")
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"批量生成失败: {str(e)}"),
                bgcolor=ft.Colors.RED
            ))
        finally:
            self._is_generating = False
            elapsed_time = time.perf_counter() - start_time
            self.terminal.add_log(f"批量生成完成 (用时: {format_elapsed_time(elapsed_time)})")

    def _save_generated_audio(self, audio: np.ndarray, sample_rate: int, prefix: str = None):
        """保存生成的音频到临时文件"""
        if prefix is None:
            prefix = self._get_save_prefix()

        if self._temp_audio_file:
            self._audio_temp_manager.cleanup_file(self._temp_audio_file)

        self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sample_rate, prefix=prefix or "audio")
        self._last_audio = (audio, sample_rate)
        self.audio_control.update_audio_state(True)

        auto_save = self.config_manager.get("audio.auto_save", False)
        if auto_save:
            self._auto_save_audio()

    # ==================== 音频控制 ====================

    async def _on_play(self, e):
        """播放音频"""
        if not self._temp_audio_file or not self._audio_temp_manager.file_exists(self._temp_audio_file):
            self.terminal.add_log("没有可播放的音频")
            return

        try:
            audio_manager = self.audio_manager_getter()

            async def progress_callback(p, c, t):
                self.audio_control.update_progress(p, c, t)
            audio_manager.set_progress_callback(progress_callback)

            async def completion_callback():
                self.audio_control.reset_progress()
            audio_manager.set_completion_callback(completion_callback)

            if self._last_audio:
                audio_data, sr = self._last_audio
                duration = audio_manager.get_audio_duration(audio_data)
                self.audio_control.set_duration(duration)

            await audio_manager.play_from_file(self._temp_audio_file)
            self.terminal.add_log("正在播放音频...")
        except Exception as e:
            logger.error(f"播放音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"播放失败: {str(e)}")

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
        import os
        from utils.dialog_utils import ConfirmDialogHelper

        if not self._temp_audio_file:
            self.terminal.add_log("没有可保存的音频")
            return

        try:
            save_dir = self.config_manager.get("audio.save_directory", "./output")
            output_format = self.config_manager.get("audio.output_format", "wav")
            custom_filename = self.audio_filename_input.value.strip() if self.audio_filename_input.value else None

            target_path = self._audio_temp_manager.get_persistent_target_path(
                save_dir,
                prefix=self._get_save_prefix(),
                custom_filename=custom_filename,
                output_format=output_format
            )

            def do_save():
                try:
                    save_path = self._audio_temp_manager.save_to_persistent(
                        self._temp_audio_file,
                        save_dir,
                        prefix=self._get_save_prefix(),
                        custom_filename=custom_filename,
                        output_format=output_format,
                        target_path=target_path
                    )

                    self.terminal.add_log(f"音频已保存: {save_path}")
                    filename = os.path.basename(save_path)
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text(f"音频已保存: {filename}"),
                        bgcolor=ft.Colors.GREEN
                    ))
                except Exception as ex:
                    logger.error(f"保存音频失败: {str(ex)}", exc_info=True)
                    self.terminal.add_log(f"保存失败: {str(ex)}")
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text(f"保存失败: {str(ex)}"),
                        bgcolor=ft.Colors.RED
                    ))

            if os.path.exists(target_path):
                filename = os.path.basename(target_path)
                helper = ConfirmDialogHelper(self._page)
                helper.show_overwrite_dialog(filename, on_confirm=do_save)
            else:
                do_save()

        except Exception as e:
            logger.error(f"保存音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"保存失败: {str(e)}")
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"保存失败: {str(e)}"),
                bgcolor=ft.Colors.RED
            ))

    def _auto_save_audio(self):
        import os
        from pathlib import Path

        if not self._last_audio:
            return

        try:
            audio_data, sample_rate = self._last_audio
            save_dir = self.config_manager.get("audio.save_directory", "./output")
            output_format = self.config_manager.get("audio.output_format", "wav")
            prefix = self._get_save_prefix()

            target_path = self._audio_temp_manager.get_audio_to_format_target_path(
                save_dir,
                prefix=prefix,
                output_format=output_format
            )

            final_path = target_path
            counter = 1
            while os.path.exists(final_path):
                name_without_ext = Path(target_path).stem
                final_path = str(Path(save_dir) / f"{name_without_ext}_{counter}.{output_format}")
                counter += 1

            save_path = self._audio_temp_manager.save_audio_to_format(
                audio_data,
                sample_rate,
                save_dir,
                prefix=prefix,
                output_format=output_format,
                target_path=final_path
            )
            self.terminal.add_log(f"音频已自动保存: {save_path}")
        except Exception as e:
            logger.error(f"自动保存音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"自动保存失败: {str(e)}")


    def _cleanup_gpu_memory(self):
        """清理GPU显存"""
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                self.terminal.add_log("已清理 GPU 显存")
        except Exception as e:
            logger.warning(f"清理 GPU 显存失败: {e}")

    # ==================== 批量生成辅助方法 ====================

    async def _execute_batch_generation(
        self,
        texts: List[str],
        tts_engine,
        batch_size: int,
        stream_method,
        prefix: str = "batch"
    ):
        """
        执行批量生成的通用逻辑

        Args:
            texts: 文本列表
            tts_engine: TTS引擎
            batch_size: 批次大小
            stream_method: 异步流式生成方法
            prefix: 文件前缀
        """
        total = len(texts)
        self.terminal.add_log(f"开始批量生成 {total} 个文本（每批最多 {batch_size} 个）...")

        # 显示进度
        self.batch_progress_text.visible = True
        self.batch_progress_bar.visible = True
        self.batch_progress_text.value = f"准备生成 {total} 个文本..."
        self.batch_progress_bar.value = 0
        self._page.update()

        item_chunks = [[] for _ in range(len(texts))]
        sample_rate = 24000
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
                async for chunks_list, sr in stream_method():
                    sample_rate = sr
                    chunk_count += 1

                    # 累积每个文本的音频块
                    for i, chunk in enumerate(chunks_list):
                        global_idx = batch_start + i
                        if chunk.size > 0:
                            item_chunks[global_idx].append(chunk)
                            item_started[i] = True
                        elif item_started[i] and not item_completed[i]:
                            item_completed[i] = True

                    # 计算进度
                    batch_completed = sum(item_completed)
                    global_completed = batch_start + batch_completed
                    progress = global_completed / total

                    self.batch_progress_text.value = f"批次 {batch_num}/{num_batches} - 已完成 {global_completed}/{total}"
                    self.batch_progress_bar.value = progress
                    self._page.update()

                # 批次间显存清理
                if batch_idx < num_batches - 1:
                    self._cleanup_gpu_memory()

            # 合并音频
            self.terminal.add_log("正在合并音频...")
            combined_audios = []
            for i, chunks in enumerate(item_chunks):
                if chunks:
                    non_empty = [c for c in chunks if c.size > 0]
                    if non_empty:
                        combined = np.concatenate(non_empty)
                        combined_audios.append(combined)
                        self.terminal.add_log(f"  文本 {i+1}: {len(combined)/sample_rate:.2f}s")

            if combined_audios:
                final_audio = np.concatenate(combined_audios)
                self._last_audio = (final_audio, sample_rate)

                if self._temp_audio_file:
                    self._audio_temp_manager.cleanup_file(self._temp_audio_file)

                self._temp_audio_file = self._audio_temp_manager.save_audio(
                    final_audio, sample_rate, prefix=prefix
                )

                self.batch_progress_text.value = f"完成: {total} 个文本, 总时长 {len(final_audio)/sample_rate:.2f}s"
                self.batch_progress_bar.value = 1.0
                self.terminal.add_log(f"批量语音生成成功: {total} 个文本")

                self.audio_control.update_audio_state(True)
                await self._on_play(None)
            else:
                self.batch_progress_text.value = "生成失败: 没有有效的音频"

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}", exc_info=True)
            self.batch_progress_text.value = f"生成失败: {str(e)}"
            raise
        finally:
            self._cleanup_gpu_memory()
