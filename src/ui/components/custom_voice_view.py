"""
自定义语音 (Custom Voice) 页面

使用 Qwen3-TTS 的预设说话人 + 情感指令生成语音
"""

import flet as ft
import logging
from ui.components.base import BaseVoiceView
from ui.components.shared_controls import create_labeled_control

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

# 收藏相关常量
DEFAULT_NAME_MAX_LENGTH = 15
MAX_NAME_LENGTH = 100
MAX_CONTENT_LENGTH = 5000


class CustomVoiceView(BaseVoiceView):
    """自定义语音页面"""

    def _get_model_type(self) -> str:
        """返回模型类型"""
        return "customvoice"

    def _get_save_prefix(self) -> str:
        """获取保存文件前缀"""
        return "custom"

    def _build_specific_controls(self) -> list:
        """构建右侧控制面板的特有部分"""
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
        self.language_dropdown = ft.Dropdown(
            label="语言选择",
            options=[
                ft.dropdown.Option("Auto", "自动检测"),
                ft.dropdown.Option("Chinese", "中文"),
                ft.dropdown.Option("English", "英语"),
                ft.dropdown.Option("Japanese", "日语"),
                ft.dropdown.Option("Korean", "韩语"),
                ft.dropdown.Option("German", "德语"),
                ft.dropdown.Option("French", "法语"),
                ft.dropdown.Option("Russian", "俄语"),
                ft.dropdown.Option("Portuguese", "葡萄牙语"),
                ft.dropdown.Option("Spanish", "西班牙语"),
                ft.dropdown.Option("Italian", "意大利语"),
            ],
            value=self.config_manager.get("custom_voice.default_language", "Chinese"),
            width=200,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 情感指令输入框
        self.instruct_input = ft.TextField(
            label="情感指令",
            multiline=True,
            min_lines=2,
            max_lines=3,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )
        self.instruct_input.on_change = self._on_instruct_change

        # 收藏按钮
        self.favorite_button = ft.IconButton(
            icon=ft.Icons.ADD,
            icon_size=18,
            tooltip="保存当前情感指令为收藏",
            on_click=self._on_save_favorite
        )

        # 我的收藏列表
        self.favorite_chips = ft.Row(
            [ft.Text("暂无收藏", size=12)],
            spacing=5,
            wrap=False,
            scroll=ft.ScrollMode.AUTO,
            height=40
        )

        # 常用预设按钮
        emotion_buttons = []
        for emotion, instruct in EMOTION_PRESETS.items():
            btn = ft.TextButton(
                content=ft.Text(emotion),
                on_click=lambda _, i=instruct: self._on_emotion_preset(_, i)
            )
            emotion_buttons.append(btn)

        # 初始化收藏
        self._refresh_favorites()

        return [
            # 说话人选择
            create_labeled_control("说话人选择", self.speaker_dropdown),
            ft.Divider(),
            # 语言选择
            create_labeled_control("语言选择", self.language_dropdown),
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
                ft.Divider(height=10),
                ft.Text("我的收藏:", size=12),
                self.favorite_chips,
            ], spacing=5),
        ]

    async def _on_generate_single_impl(self, text: str, tts_engine):
        """单个文本生成的具体实现"""
        # 获取参数
        speaker = self.speaker_dropdown.value
        language = self.language_dropdown.value
        instruct = self.instruct_input.value or ""

        # 保存配置
        self.config_manager.set("custom_voice.default_speaker", speaker)
        self.config_manager.set("custom_voice.default_language", language)

        # 日志
        self.terminal.add_log(f"说话人: {speaker}, 语言: {language}")
        if instruct:
            self.terminal.add_log(f"情感指令: {instruct}")

        try:
            self._page.update()
        except Exception:
            pass

        # 生成语音
        audio, sr = await tts_engine.custom_voice_synthesize_async(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            timeout=300.0
        )

        self.terminal.add_log("语音生成成功")

        # 保存音频
        self._save_generated_audio(audio, sr, "custom")

        # 播放音频
        await self._on_play(None)

    async def _on_generate_batch_impl(self, texts: list, tts_engine, batch_size: int):
        """批量生成的具体实现"""
        speaker = self.speaker_dropdown.value
        language = self.language_dropdown.value
        instruct = self.instruct_input.value or ""
        
        # 保存配置
        self.config_manager.set("custom_voice.default_speaker", speaker)
        self.config_manager.set("custom_voice.default_language", language)
        
        self.terminal.add_log(f"说话人: {speaker}, 语言: {language}")
        if instruct:
            self.terminal.add_log(f"情感指令: {instruct}")
        
        def stream_method():
            return tts_engine.custom_voice_batch_stream_synthesize_async(
                texts=texts,
                speaker=speaker,
                language=language,
                instruct=instruct,
            )
        
        await self._execute_batch_generation(
            texts=texts,
            tts_engine=tts_engine,
            batch_size=batch_size,
            stream_method=stream_method,
            prefix="batch"
        )
    # ==================== 特有方法：情感预设 ====================

    def _on_emotion_preset(self, _, instruct: str):
        """情感预设按钮点击事件"""
        self.instruct_input.value = instruct
        self.instruct_input.update()

    def _on_instruct_change(self, _):
        """指令输入框变化事件"""
        pass

    # ==================== 特有方法：收藏功能 ====================

    def _on_save_favorite(self, _):
        """保存当前情感指令为收藏"""
        instruct = self.instruct_input.value or ""
        if not instruct.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请先输入情感指令"),
                bgcolor=ft.Colors.ORANGE
            ))
            return

        instruct = instruct.strip()
        default_name = instruct[:DEFAULT_NAME_MAX_LENGTH] + "..." if len(instruct) > DEFAULT_NAME_MAX_LENGTH else instruct

        name_input = ft.TextField(
            label="收藏名称",
            value=default_name,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            autofocus=True
        )

        def save_dialog(_):
            name = name_input.value.strip() or default_name

            if len(name) > MAX_NAME_LENGTH:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"名称过长（最多{MAX_NAME_LENGTH}字符）"),
                    bgcolor=ft.Colors.RED
                ))
                return

            if len(instruct) > MAX_CONTENT_LENGTH:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"内容过长（最多{MAX_CONTENT_LENGTH}字符）"),
                    bgcolor=ft.Colors.RED
                ))
                return

            existing_names = [f["name"] for f in self.voice_library.get_favorite_instructs()]
            if name in existing_names:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"收藏名称 \"{name}\" 已存在"),
                    bgcolor=ft.Colors.RED
                ))
                return

            success = self.voice_library.add_favorite_instruct(name, instruct)

            if success:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"已收藏: {name}"),
                    bgcolor=ft.Colors.GREEN
                ))
                self.terminal.add_log(f"已保存收藏: {name}")
                self._refresh_favorites()
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
        favorites = self.voice_library.get_favorite_instructs()

        if not favorites:
            self.favorite_chips.controls = [ft.Text("暂无收藏", size=12)]
        else:
            self.favorite_chips.controls.clear()
            for fav in favorites:
                menu_button = ft.PopupMenuButton(
                    icon=ft.Icons.MORE_VERT,
                    items=[
                        ft.PopupMenuItem(
                            content=ft.Text("编辑"),
                            icon=ft.Icons.EDIT,
                            on_click=lambda _, f=fav: self._on_edit_favorite(_, f)
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
                        on_click=lambda _, i=fav["instruct"]: self._on_favorite_click(_, i),
                        tooltip=fav["instruct"]
                    ),
                    menu_button
                ], spacing=5)

                self.favorite_chips.controls.append(chip_row)

        try:
            self.favorite_chips.update()
        except RuntimeError:
            pass

    def _on_favorite_click(self, _, instruct: str):
        """收藏项点击事件"""
        self.instruct_input.value = instruct
        self.instruct_input.update()
        self.terminal.add_log("已加载收藏的情感指令")

    def _on_edit_favorite(self, _, fav: dict):
        """编辑收藏"""
        old_name = fav["name"]
        old_instruct = fav["instruct"]

        name_input = ft.TextField(
            label="收藏名称",
            value=old_name,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
        )

        instruct_input = ft.TextField(
            label="情感指令",
            value=old_instruct,
            multiline=True,
            min_lines=2,
            max_lines=4,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            autofocus=True
        )

        def save_dialog(_):
            new_name = name_input.value.strip()
            new_instruct = instruct_input.value.strip()

            if not new_name or not new_instruct:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("名称和指令不能为空"),
                    bgcolor=ft.Colors.RED
                ))
                return

            success = self.voice_library.update_favorite_instruct(old_instruct, new_name, new_instruct)

            if success:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("已修改"),
                    bgcolor=ft.Colors.GREEN
                ))
                self._refresh_favorites()
                if self.instruct_input.value == old_instruct:
                    self.instruct_input.value = new_instruct
                    self.instruct_input.update()
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
            content=ft.Column([name_input, instruct_input], spacing=10, tight=True),
            actions=[
                ft.TextButton("取消", on_click=close_dialog),
                ft.TextButton("保存", on_click=save_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self._page.show_dialog(dialog)

    def _on_delete_favorite(self, _, fav: dict):
        """删除收藏"""
        instruct = fav["instruct"]
        name = fav["name"]

        def confirm_delete(_):
            success = self.voice_library.remove_favorite_instruct(instruct)
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
