"""
声音克隆 (Voice Clone) 页面

使用参考音频克隆声音
"""

import flet as ft
import logging
import asyncio
import os

from ui.components.shared_controls import TextPanel, AudioControlPanel
from ui.components.voice_library import VoiceLibrary

logger = logging.getLogger(__name__)


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
        model_manager
    ):
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.audio_manager_getter = audio_manager_getter
        self.terminal = terminal
        self.voice_library = voice_library
        self.config_manager = config_manager
        self.model_manager = model_manager

        # 当前生成的音频和克隆数据
        self._last_audio = None
        self._ref_audio_path = None
        self._is_generating = False

        # 文件选择器 (FilePicker 不是控件，不需要添加到 overlay)
        self.file_picker = ft.FilePicker()

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

        # 参考音频选择
        self.ref_audio_button = ft.Button(
            "选择文件",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self._on_pick_file
        )

        self.ref_audio_status = ft.Text(
            "未选择文件",
            size=12,
            color=ft.Colors.GREY_500
        )

        # 参考文本输入框
        self.ref_text_input = ft.TextField(
            label="参考文本",
            multiline=True,
            min_lines=3,
            max_lines=5,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 克隆选项
        self.save_clone_checkbox = ft.Checkbox(
            label="保存为可重用克隆",
            value=False
        )

        self.clone_name_input = ft.TextField(
            label="克隆名称",
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            visible=False
        )

        self.x_vector_only_checkbox = ft.Checkbox(
            label="仅使用 x_vector (快速模式，质量可能降低)",
            value=False
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
            has_audio=False
        )

        # 右侧控制面板
        control_panel = ft.Container(
            content=ft.Column(
                [
                    # 参考音频
                    ft.Column([
                        ft.Text("参考音频 📁", size=14, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            self.ref_audio_button,
                            self.ref_audio_status,
                        ], spacing=10),
                    ], spacing=5),

                    ft.Divider(),

                    # 参考文本
                    ft.Column([
                        ft.Row([
                            ft.Text("参考文本 📝", size=14, weight=ft.FontWeight.BOLD),
                            ft.Icon(
                                ft.Icons.INFO_OUTLINE,
                                size=16,
                                tooltip="必填项，输入参考音频中对应的文本内容"
                            )
                        ]),
                        self.ref_text_input,
                    ], spacing=5),

                    ft.Divider(),

                    # 克隆模式选择
                    ft.Column([
                        ft.Text("当前克隆", size=14, weight=ft.FontWeight.BOLD),
                        self.use_saved_clone_radio,
                        self.saved_clone_dropdown,
                    ], spacing=5),

                    ft.Divider(),

                    # 克隆选项
                    ft.Column([
                        ft.Text("克隆选项", size=14, weight=ft.FontWeight.BOLD),
                        self.save_clone_checkbox,
                        self.clone_name_input,
                        self.x_vector_only_checkbox,
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

                    # 克隆声音库
                    ft.Column([
                        ft.Row([
                            ft.Text("我的克隆声音库", size=14, weight=ft.FontWeight.BOLD),
                            ft.Text(
                                f"({len(self.voice_library.get_all_clones())})",
                                size=12,
                                color=ft.Colors.GREY_500
                            )
                        ]),
                        ft.Container(
                            content=self.clone_library_grid,
                            height=200,
                            bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
                            border_radius=8,
                            padding=10,
                            alignment=ft.Alignment(0, 0)
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
                self.ref_audio_status.color = ft.Colors.GREEN
                self.ref_audio_status.update()
                self.terminal.add_log(f"已选择参考音频: {filename}")
        except Exception as ex:
            logger.error(f"选择文件失败: {str(ex)}", exc_info=True)
            self.terminal.add_log(f"✗ 选择文件失败: {str(ex)}")

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

    def _update_saved_clone_dropdown(self):
        """更新已保存克隆下拉框"""
        clones = self.voice_library.get_all_clones()
        options = [
            ft.DropdownOption(
                text=c["name"],
                value=c["id"]
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
                ft.Text("暂无克隆", size=12, color=ft.Colors.GREY_500)
            )
        else:
            for clone in clones:
                card = ft.Container(
                    content=ft.Column([
                        ft.Text(clone["name"], size=13, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            clone["created_at"][:10],
                            size=11,
                            color=ft.Colors.GREY_400
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

        # 判断克隆模式
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

            ref_audio = clone["ref_audio"]
            ref_text = clone["ref_text"]
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

        self._is_generating = True
        self.terminal.add_log("正在生成语音...")

        try:
            # 获取TTS引擎
            tts_engine = self.tts_engine_getter()

            # 获取参数
            x_vector_only = self.x_vector_only_checkbox.value

            # 生成语音
            self.terminal.add_log(f"参考音频: {os.path.basename(ref_audio)}")

            audio, sr = tts_engine.voice_clone_synthesize(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only=x_vector_only
            )

            self._last_audio = (audio, sr)
            self.terminal.add_log("✓ 语音生成成功")

            # 检查是否需要保存为可重用克隆
            if self.save_clone_checkbox.value and clone_mode == "new":
                clone_name = self.clone_name_input.value or "未命名克隆"

                # 创建克隆提示
                clone_id = self.voice_library.add_clone(
                    name=clone_name,
                    ref_audio=ref_audio,
                    ref_text=ref_text
                )

                if clone_id:
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text(f"已保存克隆: {clone_name}"),
                        bgcolor=ft.Colors.GREEN
                    ))
                    self.terminal.add_log(f"已保存克隆: {clone_name} ({clone_id})")
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

    async def _on_save_audio(self, e):
        """保存音频"""
        if not self._last_audio:
            return

        try:
            save_dir = self.config_manager.get("audio.save_directory", "./output")
            self.terminal.add_log(f"音频保存功能开发中... (保存到: {save_dir})")

        except Exception as e:
            logger.error(f"保存音频失败: {str(e)}", exc_info=True)
            self.terminal.add_log(f"✗ 保存失败: {str(e)}")
