"""
自定义语音 (Custom Voice) 页面

使用 Qwen3-TTS 的预设说话人 + 情感指令生成语音
"""

import flet as ft
import logging
import asyncio
import os

from ui.components.shared_controls import TextPanel, AudioControlPanel
from ui.components.voice_library import VoiceLibrary
from tts.audio_temp_manager import AudioTempManager

logger = logging.getLogger(__name__)

# 常用情感预设
EMOTION_PRESETS = {
    "正常": "",
    "开心": "用开心的语气说",
    "悲伤": "用悲伤的语气说",
    "愤怒": "用愤怒的语气说",
    "疑问": "用疑问的语气说",
    "惊讶": "用惊讶的语气说"
}


class CustomVoiceView(ft.Container):
    """自定义语音页面"""

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

        # 当前生成的音频和临时文件路径
        self._last_audio = None
        self._temp_audio_file = None
        self._is_generating = False

        # 音频临时文件管理器
        self._audio_temp_manager = AudioTempManager()

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

    def _build_ui(self):
        """构建UI界面"""
        # 模型选择下拉框 - 只显示 CustomVoice 模型
        usable_models = self.model_manager.list_usable_models_by_type("customvoice")
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

        # 说话人选择
        speakers = self.voice_library.get_custom_voice_speakers()
        self.speaker_dropdown = ft.Dropdown(
            label="说话人",
            options=[ft.DropdownOption(text=s) for s in speakers],
            value=self.config_manager.get("custom_voice.default_speaker", "Vivian"),
            width=200,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 语言选择
        languages = self.voice_library.get_supported_languages()
        self.language_radio = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="Chinese", label="中文"),
                ft.Radio(value="English", label="英语"),
                ft.Radio(value="Japanese", label="日语"),
                ft.Radio(value="Auto", label="自动检测")
            ]),
            value=self.config_manager.get("custom_voice.default_language", "Chinese")
        )

        # 情感指令输入框
        self.instruct_input = ft.TextField(
            label="情感指令",
            multiline=True,
            min_lines=2,
            max_lines=3,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 收藏按钮
        self.favorite_button = ft.IconButton(
            icon=ft.Icons.FAVORITE_BORDER,
            tooltip="收藏当前情感指令",
            on_click=self._on_toggle_favorite
        )

        # 常用预设按钮
        emotion_buttons = []
        for emotion, instruct in EMOTION_PRESETS.items():
            btn = ft.TextButton(
                content=ft.Text(emotion),
                on_click=lambda e, i=instruct: self._on_emotion_preset(e, i)
            )
            emotion_buttons.append(btn)

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
                    # 说话人选择
                    ft.Column([
                        ft.Text("说话人选择", size=14, weight=ft.FontWeight.BOLD),
                        self.speaker_dropdown,
                    ], spacing=5),

                    ft.Divider(),

                    # 语言选择
                    ft.Column([
                        ft.Text("语言选择", size=14, weight=ft.FontWeight.BOLD),
                        self.language_radio,
                    ], spacing=5),

                    ft.Divider(),

                    # 情感指令
                    ft.Column([
                        ft.Row([
                            ft.Text("情感指令", size=14, weight=ft.FontWeight.BOLD),
                            ft.Icon(
                                ft.Icons.INFO_OUTLINE,
                                size=16,
                                tooltip="可选填，描述情感和语气"
                            ),
                            ft.Container(expand=True),
                            self.favorite_button
                        ]),
                        self.instruct_input,
                        ft.Text("常用预设:", size=12),
                        ft.Row(emotion_buttons, spacing=5, wrap=True),
                    ], spacing=5),

                    ft.Divider(),

                    # 生成按钮
                    ft.Button(
                        "生成语音",
                        icon=ft.Icons.SEND,
                        style=ft.ButtonStyle(
                            text_style=ft.TextStyle(
                                font_family="Microsoft YaHei",
                                weight=ft.FontWeight.BOLD
                            )
                        ),
                        on_click=self._on_generate
                    ),

                    ft.Divider(),

                    # 音频控制
                    self.audio_control,

                    ft.Divider(),

                    # 音频文件名设置
                    ft.Column([
                        ft.Text("保存设置", size=14, weight=ft.FontWeight.BOLD),
                        self.audio_filename_input,
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

        # 主布局
        return ft.Row(
            [
                # 左侧文本输入区
                ft.Container(
                    content=ft.Column([
                        # 模型选择
                        ft.Column([
                            ft.Text("模型选择", size=14, weight=ft.FontWeight.BOLD),
                            self.model_dropdown,
                        ], spacing=5),

                        ft.Divider(),

                        ft.Text("文本输入", size=16, weight=ft.FontWeight.BOLD),
                        self.text_panel,
                    ], spacing=10),
                    padding=10,
                    expand=True
                ),

                # 右侧控制面板
                control_panel
            ],
            spacing=20,
            expand=True
        )

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

            # 重新获取 CustomVoice 模型列表
            usable_models = self.model_manager.list_usable_models_by_type("customvoice")
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

    def _on_emotion_preset(self, e, instruct: str):
        """情感预设按钮点击事件"""
        self.instruct_input.value = instruct
        self.instruct_input.update()

    def _on_clear_text(self, e):
        """清空文本"""
        self.text_panel.clear()

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

        self._is_generating = True
        self.terminal.add_log("正在生成语音...")

        # 强制UI更新，让第一条日志立即显示
        try:
            self._page.update()
        except:
            pass

        try:
            # 获取参数
            speaker = self.speaker_dropdown.value
            language = self.language_radio.value
            instruct = self.instruct_input.value or ""

            # 保存配置
            self.config_manager.set("custom_voice.default_speaker", speaker)
            self.config_manager.set("custom_voice.default_language", language)

            # 获取TTS引擎
            tts_engine = self.tts_engine_getter()

            # 生成语音（使用后台线程，避免阻塞UI）
            self.terminal.add_log(f"说话人: {speaker}, 语言: {language}")
            if instruct:
                self.terminal.add_log(f"情感指令: {instruct}")

            # 强制UI更新，让参数日志立即显示
            try:
                self._page.update()
            except:
                pass

            # 在后台线程中执行TTS生成（使用异步API）
            audio, sr = await tts_engine.custom_voice_synthesize_async(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                timeout=300.0
            )

            self.terminal.add_log("✓ 语音生成成功")

            # 使用临时文件管理器保存音频
            if self._temp_audio_file:
                self._audio_temp_manager.cleanup_file(self._temp_audio_file)

            self._temp_audio_file = self._audio_temp_manager.save_audio(audio, sr, prefix="custom")

            # 保存音频数据用于计算时长
            self._last_audio = (audio, sr)

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
                prefix="custom",
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

    def _on_toggle_favorite(self, e):
        """切换收藏按钮点击事件"""
        instruct = self.instruct_input.value or ""
        if not instruct.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请先输入情感指令"),
                bgcolor=ft.Colors.ORANGE
            ))
            return

        instruct = instruct.strip()

        # 检查是否已收藏
        if self.voice_library.is_favorite_instruct(instruct):
            # 取消收藏
            self.voice_library.remove_favorite_instruct(instruct)
            self.favorite_button.icon = ft.Icons.FAVORITE_BORDER
            self.favorite_button.tooltip = "收藏当前情感指令"
            self.favorite_button.update()
            self._page.show_dialog(ft.SnackBar(
                ft.Text("已取消收藏"),
                bgcolor=ft.Colors.GREY
            ))
        else:
            # 添加收藏
            self.voice_library.add_favorite_instruct(instruct)
            self.favorite_button.icon = ft.Icons.FAVORITE
            self.favorite_button.tooltip = "取消收藏"
            self.favorite_button.update()
            self._page.show_dialog(ft.SnackBar(
                ft.Text("已添加到收藏"),
                bgcolor=ft.Colors.GREEN
            ))

    def _refresh_favorite_button(self):
        """刷新收藏按钮状态（当指令输入框内容变化时调用）"""
        instruct = self.instruct_input.value or ""
        if instruct.strip() and self.voice_library.is_favorite_instruct(instruct.strip()):
            self.favorite_button.icon = ft.Icons.FAVORITE
            self.favorite_button.tooltip = "取消收藏"
        else:
            self.favorite_button.icon = ft.Icons.FAVORITE_BORDER
            self.favorite_button.tooltip = "收藏当前情感指令"
        try:
            self.favorite_button.update()
        except:
            pass
