"""
声音设计 (Voice Design) 页面

通过自然语言描述设计声音
"""

import flet as ft
import logging
import asyncio

from ui.components.shared_controls import TextPanel, AudioControlPanel
from ui.components.voice_library import VoiceLibrary

logger = logging.getLogger(__name__)


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
        model_manager
    ):
        # 使用私有变量存储 page，避免与 ft.Container 的 page 属性冲突
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
            size=12,
            color=ft.Colors.GREY_400
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
                content=ft.Column([
                    ft.Text(name, size=13, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        desc[:30] + "..." if len(desc) > 30 else desc,
                        size=11,
                        color=ft.Colors.GREY_400
                    ),
                ], spacing=5),
                padding=10,
                bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
                border_radius=8,
                width=150,
                height=80,
                on_click=lambda e, n=name, d=desc: self._on_preset_click(e, n, d),
                tooltip=desc
            )
            preset_cards.append(card)

        # 我的收藏 (Chips)
        self.fav_chips = ft.Row(
            [ft.Text("暂无收藏", size=12, color=ft.Colors.GREY_500)],
            spacing=5,
            wrap=True
        )

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
                            max_extent=160,
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

    def _on_design_change(self, e):
        """声音描述输入变化事件"""
        text = self.design_input.value or ""
        char_count = len(text)
        self.char_count.value = f"字符数: {char_count} / 推荐 30-80"

        # 根据字符数改变颜色
        if 30 <= char_count <= 80:
            self.char_count.color = ft.Colors.GREEN
        elif char_count < 30 or char_count > 100:
            self.char_count.color = ft.Colors.ORANGE
        else:
            self.char_count.color = ft.Colors.GREY_400

        self.char_count.update()

    def _on_preset_click(self, e, name: str, desc: str):
        """预设声音卡片点击事件"""
        self.design_input.value = desc
        self.design_input.update()
        self._on_design_change(None)
        self.terminal.add_log(f"已选择预设: {name}")

    def _on_save_favorite(self, e):
        """保存当前描述为收藏"""
        desc = self.design_input.value or ""
        if not desc or not desc.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请先输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

        # 生成收藏名称（使用描述的前10个字符）
        name = desc.strip()[:10] + ("..." if len(desc) > 10 else "")

        # 保存到声音库
        success = self.voice_library.add_design_preset(name, desc)

        if success:
            self._page.show_dialog(ft.SnackBar(
                ft.Text(f"已收藏: {name}"),
                bgcolor=ft.Colors.GREEN
            ))
            self.terminal.add_log(f"已保存收藏: {name}")
        else:
            self._page.show_dialog(ft.SnackBar(
                ft.Text("保存失败"),
                bgcolor=ft.Colors.RED
            ))

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
                    ft.Text(desc, size=11, color=ft.Colors.GREY_400, expand=True),
                ], spacing=5),
                padding=5,
                on_click=lambda e, d=item["description"]: self._on_history_click(e, d),
                tooltip=item["description"]
            )
            self.history_list.controls.append(control)

        # 只有在控件已添加到页面时才调用 update()
        try:
            self.history_list.update()
        except RuntimeError:
            pass

    def _on_history_click(self, e, desc: str):
        """历史记录点击事件"""
        self.design_input.value = desc
        self.design_input.update()
        self._on_design_change(None)
        self.terminal.add_log("已加载历史设计")

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

        # 获取声音描述
        design_prompt = self.design_input.value or ""
        if not design_prompt or not design_prompt.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

        self._is_generating = True
        self.terminal.add_log("正在生成语音...")

        try:
            # 获取TTS引擎
            tts_engine = self.tts_engine_getter()

            # 生成语音
            self.terminal.add_log(f"声音描述: {design_prompt[:50]}...")

            audio, sr = tts_engine.voice_design_synthesize(
                text=text,
                design_prompt=design_prompt,
                language="Chinese"
            )

            self._last_audio = (audio, sr)
            self.terminal.add_log("✓ 语音生成成功")

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

    async def _on_play(self, e):
        """播放音频"""
        if not self._last_audio:
            return

        try:
            audio_manager = self.audio_manager_getter()
            audio_data, sr = self._last_audio
            await audio_manager.play(audio_data, sr)
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
            save_dir = self.config_manager.get("audio.save_directory", "./output")
            self.terminal.add_log(f"音频保存功能开发中... (保存到: {save_dir})")

        except Exception as e:
            logger.error(f"保存音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 保存失败: {str(e)}")
