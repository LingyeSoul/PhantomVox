"""
声音设计 (Voice Design) 页面

通过自然语言描述设计声音
"""

import flet as ft
import logging
import numpy as np

from ui.components.base import BaseVoiceView
from ui.components.shared_controls import create_labeled_control

logger = logging.getLogger(__name__)

# 收藏相关常量
DEFAULT_DESCRIPTION_MAX_LENGTH = 10
MAX_NAME_LENGTH = 100
MAX_CONTENT_LENGTH = 5000


class VoiceDesignView(BaseVoiceView):
    """声音设计页面"""

    def _get_model_type(self) -> str:
        """返回模型类型"""
        return "voicedesign"

    def _get_save_prefix(self) -> str:
        """获取保存文件前缀"""
        return "design"

    def _build_specific_controls(self) -> list:
        """构建右侧控制面板的特有部分"""
        # 声音描述输入框
        self.design_input = ft.TextField(
            label="声音描述",
            multiline=True,
            min_lines=4,
            max_lines=6,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 字符计数
        self.char_count = ft.Text("字符数: 0 / 推荐 30-80", size=12)
        self.design_input.on_change = self._on_design_change

        # 预设声音卡片
        preset_cards = []
        presets = self.voice_library.get_all_design_presets()

        for name, desc in presets.items():
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

        # 我的收藏
        self.fav_chips = ft.Row(
            [ft.Text("暂无收藏", size=12)],
            spacing=5,
            wrap=False,
            scroll=ft.ScrollMode.AUTO,
            height=40
        )
        self._refresh_favorites()

        # 设计历史
        self.history_list = ft.ListView(
            expand=1,
            spacing=5,
            item_extent=40
        )
        self._refresh_history()

        return [
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
                ft.Text("预设声音", size=14, weight=ft.FontWeight.BOLD),
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
                    ft.Text("我的收藏", size=14, weight=ft.FontWeight.BOLD),
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
        ]

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

    async def _on_generate_single_impl(self, text: str, tts_engine):
        """单个文本生成的具体实现"""
        design_prompt = self.design_input.value or ""
        if not design_prompt or not design_prompt.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

        self.terminal.add_log(f"声音描述: {design_prompt[:50]}...")

        try:
            self._page.update()
        except:
            pass

        # 生成语音
        audio, sr = await tts_engine.voice_design_synthesize_async(
            text=text,
            design_prompt=design_prompt,
            language="Chinese",
            timeout=300.0
        )

        self.terminal.add_log("语音生成成功")
        self._save_generated_audio(audio, sr, "design")

        # 保存到设计历史
        self.voice_library.save_design_history("自定义设计", design_prompt)
        self._refresh_history()

        await self._on_play(None)

    async def _on_generate_batch_impl(self, texts: list, tts_engine, batch_size: int):
        """批量生成的具体实现"""
        design_prompt = self.design_input.value or ""
        if not design_prompt or not design_prompt.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

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
        num_batches = (total + batch_size - 1) // batch_size

        try:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total)
                batch_texts = texts[batch_start:batch_end]
                batch_num = batch_idx + 1

                self.terminal.add_log(f"处理第 {batch_num}/{num_batches} 批 (文本 {batch_start+1}-{batch_end})...")

                item_started = [False] * len(batch_texts)
                item_completed = [False] * len(batch_texts)

                async for chunks_list, sr in tts_engine.voice_design_batch_stream_synthesize_async(
                    texts=batch_texts,
                    design_prompt=design_prompt,
                    language="Chinese",
                ):
                    sample_rate = sr

                    for i, chunk in enumerate(chunks_list):
                        global_idx = batch_start + i
                        if chunk.size > 0:
                            item_chunks[global_idx].append(chunk)
                            item_started[i] = True
                        elif item_started[i] and not item_completed[i]:
                            item_completed[i] = True

                    batch_completed = sum(item_completed)
                    global_completed = batch_start + batch_completed
                    progress = global_completed / total

                    self.batch_progress_text.value = f"批次 {batch_num}/{num_batches} - 已完成 {global_completed}/{total}"
                    self.batch_progress_bar.value = progress
                    self._page.update()

                if batch_idx < num_batches - 1:
                    self._cleanup_gpu_memory()
                    self.terminal.add_log(f"批次 {batch_num}/{num_batches} 完成，已清理显存")

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
                self._save_generated_audio(final_audio, sample_rate, "batch")

                self.batch_progress_text.value = f"批量生成完成: {total} 个文本, 总时长 {len(final_audio)/sample_rate:.2f}s"
                self.batch_progress_bar.value = 1.0
                self.terminal.add_log(f"批量语音生成成功: {total} 个文本")

                # 保存到设计历史
                self.voice_library.save_design_history("批量设计", design_prompt)
                self._refresh_history()

                await self._on_play(None)
            else:
                self.batch_progress_text.value = "生成失败: 没有有效的音频"
                self.terminal.add_log("批量生成失败: 没有生成任何音频")

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}", exc_info=True)
            self.batch_progress_text.value = f"生成失败: {str(e)}"
            raise
        finally:
            self._cleanup_gpu_memory()

    # ==================== 特有方法：收藏和历史 ====================

    def _on_save_favorite(self, _):
        """保存当前描述为收藏"""
        desc = self.design_input.value or ""
        if not desc or not desc.strip():
            self._page.show_dialog(ft.SnackBar(
                ft.Text("请先输入声音描述"),
                bgcolor=ft.Colors.RED
            ))
            return

        default_name = desc.strip()[:DEFAULT_DESCRIPTION_MAX_LENGTH] + ("..." if len(desc) > DEFAULT_DESCRIPTION_MAX_LENGTH else "")

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

            if len(desc) > MAX_CONTENT_LENGTH:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"内容过长（最多{MAX_CONTENT_LENGTH}字符）"),
                    bgcolor=ft.Colors.RED
                ))
                return

            existing_names = [f["name"] for f in self.voice_library.get_favorite_designs()]
            if name in existing_names:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"收藏名称 \"{name}\" 已存在"),
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
        favorites = self.voice_library.get_favorite_designs()

        if not favorites:
            self.fav_chips.controls = [ft.Text("暂无收藏", size=12)]
        else:
            self.fav_chips.controls.clear()
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
                        on_click=lambda _, d=fav["description"]: self._on_favorite_click(_, d),
                    ),
                    menu_button
                ], spacing=5)

                self.fav_chips.controls.append(chip_row)

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

    def _on_edit_favorite(self, _, fav: dict):
        """编辑收藏"""
        old_name = fav["name"]
        old_desc = fav["description"]

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
            timestamp = item["timestamp"][:10]
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
