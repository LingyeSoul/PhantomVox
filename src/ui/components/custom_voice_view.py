"""
自定义语音 (Custom Voice) 页面

使用 Qwen3-TTS 的预设说话人 + 情感指令生成语音
"""

import flet as ft
import logging
import asyncio

from ui.components.shared_controls import TextPanel, AudioControlPanel
from ui.components.voice_library import VoiceLibrary

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
        model_manager
    ):
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.audio_manager_getter = audio_manager_getter
        self.terminal = terminal
        self.voice_library = voice_library
        self.config_manager = config_manager
        self.model_manager = model_manager

        # 当前生成的音频
        self._last_audio = None
        self._is_generating = False

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

    def _build_ui(self):
        """构建UI界面"""
        # 模型选择下拉框
        usable_models = self.model_manager.list_usable_models()
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
            disabled=len(usable_models) == 0
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
            content=ft.Column([
                ft.Radio(value="Chinese", label="中文"),
                ft.Radio(value="English", label="英语"),
                ft.Radio(value="Japanese", label="日语"),
                ft.Radio(value="Korean", label="韩语"),
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
            has_audio=False
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
                            )
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
                        bgcolor=ft.Colors.BLUE,
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

                        ft.Divider(),

                        # 终端日志
                        ft.Column([
                            ft.Text("运行日志", size=14, weight=ft.FontWeight.BOLD),
                            self.terminal.view,
                        ], spacing=5),
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

    def _on_emotion_preset(self, e, instruct: str):
        """情感预设按钮点击事件"""
        self.instruct_input.value = instruct
        self.instruct_input.update()

        # 保存到最近使用
        if instruct:
            self.voice_library.add_recent_instruct(instruct)

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

        try:
            # 获取参数
            speaker = self.speaker_dropdown.value
            language = self.language_radio.value
            instruct = self.instruct_input.value or ""

            # 保存配置
            self.config_manager.set("custom_voice.default_speaker", speaker)
            self.config_manager.set("custom_voice.default_language", language)

            # 保存情感指令到最近使用
            if instruct:
                self.voice_library.add_recent_instruct(instruct)

            # 获取TTS引擎
            tts_engine = self.tts_engine_getter()

            # 生成语音
            self.terminal.add_log(f"说话人: {speaker}, 语言: {language}")
            if instruct:
                self.terminal.add_log(f"情感指令: {instruct}")

            audio, sr = tts_engine.custom_voice_synthesize(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct
            )

            self._last_audio = (audio, sr)
            self.terminal.add_log("✓ 语音生成成功")

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
        if not self._last_audio:
            return

        try:
            audio_manager = self.audio_manager_getter()
            audio_data, sr = self._last_audio
            await audio_manager.play(audio_data)
            self.terminal.add_log("正在播放音频...")
        except Exception as e:
            logger.error(f"播放音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 播放失败: {str(e)}")

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
        if not self._last_audio:
            return

        try:
            # 获取保存路径
            save_dir = self.config_manager.get("audio.save_directory", "./output")

            # TODO: 实现文件保存对话框
            self.terminal.add_log(f"音频保存功能开发中... (保存到: {save_dir})")

        except Exception as e:
            logger.error(f"保存音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 保存失败: {str(e)}")
