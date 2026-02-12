"""
声音克隆 (Voice Clone) 页面

使用参考音频克隆声音
"""

import flet as ft
import logging
import os
import numpy as np

from ui.components.base import BaseVoiceView
from ui.components.shared_controls import create_labeled_control

logger = logging.getLogger(__name__)


class VoiceCloneView(BaseVoiceView):
    """声音克隆页面"""

    def __init__(self, *args, **kwargs):
        # 克隆特有属性
        self._ref_audio_path = None
        self._last_saved_clone_ref_audio = None
        self._last_saved_clone_ref_text = None
        self.file_picker = ft.FilePicker()
        super().__init__(*args, **kwargs)

    def _get_model_type(self) -> str:
        """返回模型类型"""
        return "base"

    def _get_save_prefix(self) -> str:
        """获取保存文件前缀"""
        return "clone"

    def _build_specific_controls(self) -> list:
        """构建右侧控制面板的特有部分"""
        # 参考音频选择
        self.ref_audio_button = ft.Button(
            "选择文件",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self._on_pick_file
        )

        self.ref_audio_status = ft.Text("未选择文件", size=12)

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

        # 克隆模式选择
        self.use_saved_clone_radio = ft.RadioGroup(
            content=ft.Column([
                ft.Radio(value="new", label="使用新音频（每次提取特征）"),
                ft.Radio(value="saved", label="使用已保存的克隆"),
            ]),
            value="new"
        )
        self.use_saved_clone_radio.on_change = self._on_clone_mode_change

        self.saved_clone_dropdown = ft.Dropdown(
            label="选择克隆",
            options=[],
            width=200,
            visible=False,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )
        self.saved_clone_dropdown.on_change = self._on_saved_clone_changed

        return [
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
        ]

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

    async def _on_generate_single_impl(self, text: str, tts_engine):
        """单个文本生成的具体实现"""
        clone_mode = self.use_saved_clone_radio.value
        clone_prompt = None
        ref_audio = None
        ref_text = None
        x_vector_only = self.x_vector_only_checkbox.value

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

            # 检查是否有预计算的特征
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
                self.terminal.add_log("特征已保存，下次可直接使用")

            # 使用特征生成语音
            audio, sr = await tts_engine.voice_clone_synthesize_async(
                text=text,
                clone_prompt=clone_prompt,
                timeout=300.0
            )

            self._save_generated_audio(audio, sr, "clone")
            self.terminal.add_log("语音生成成功")
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
        self.terminal.add_log(f"参考音频: {os.path.basename(ref_audio)}")

        try:
            self._page.update()
        except:
            pass

        # 生成语音
        audio, sr = await tts_engine.voice_clone_synthesize_async(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
            timeout=300.0
        )

        self.terminal.add_log("语音生成成功")
        self._save_generated_audio(audio, sr, "clone")

        # 检查是否需要保存为可重用克隆
        if self.save_clone_checkbox.value and clone_mode == "new":
            clone_name = self.clone_name_input.value or "未命名克隆"

            if (self._last_saved_clone_ref_audio == ref_audio and
                self._last_saved_clone_ref_text == ref_text):
                self.terminal.add_log("该克隆已存在，跳过保存")
            else:
                self.terminal.add_log("正在提取声音特征并保存克隆...")

                prompt_features = await tts_engine.create_voice_clone_prompt_async(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only=x_vector_only
                )

                clone_id = self.voice_library.add_clone(
                    name=clone_name,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    prompt_features=prompt_features,
                    x_vector_only=x_vector_only
                )

                if clone_id:
                    self._last_saved_clone_ref_audio = ref_audio
                    self._last_saved_clone_ref_text = ref_text
                    self._page.show_dialog(ft.SnackBar(
                        ft.Text(f"已保存克隆: {clone_name}"),
                        bgcolor=ft.Colors.GREEN
                    ))
                    self.terminal.add_log(f"已保存克隆: {clone_name} ({clone_id})")
                    self._refresh_clone_library()

        await self._on_play(None)

    async def _on_generate_batch_impl(self, texts: list, tts_engine, batch_size: int):
        """批量生成的具体实现"""
        # 获取 clone_prompt
        clone_prompt = await self._get_clone_prompt(tts_engine)
        if clone_prompt is None:
            return

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

                async for chunks_list, sr in tts_engine.voice_clone_batch_stream_synthesize_async(
                    texts=batch_texts,
                    clone_prompt=clone_prompt,
                    language="Auto",
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

    async def _get_clone_prompt(self, tts_engine):
        """获取克隆提示（从已保存克隆或新音频）"""
        clone_mode = self.use_saved_clone_radio.value

        if clone_mode == "saved":
            clone_id = self.saved_clone_dropdown.value
            if not clone_id:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("请选择要使用的克隆"),
                    bgcolor=ft.Colors.RED
                ))
                return None

            clone = self.voice_library.get_clone(clone_id)
            if not clone:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("克隆不存在"),
                    bgcolor=ft.Colors.RED
                ))
                return None

            if "prompt_features" in clone and clone["prompt_features"]:
                self.terminal.add_log(f"使用预计算特征: {clone['name']}")
                return clone["prompt_features"]
            else:
                self.terminal.add_log(f"首次使用克隆 '{clone['name']}'，正在提取特征...")
                clone_prompt = await tts_engine.create_voice_clone_prompt_async(
                    ref_audio=clone["ref_audio"],
                    ref_text=clone["ref_text"],
                    x_vector_only=False
                )
                self.voice_library.update_clone_features(clone_id, clone_prompt)
                self.terminal.add_log("特征已保存，下次可直接使用")
                return clone_prompt
        else:
            if not self._ref_audio_path:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("请选择参考音频"),
                    bgcolor=ft.Colors.RED
                ))
                return None

            ref_text = self.ref_text_input.value or ""
            if not ref_text or not ref_text.strip():
                self._page.show_dialog(ft.SnackBar(
                    ft.Text("请输入参考文本"),
                    bgcolor=ft.Colors.RED
                ))
                return None

            x_vector_only = self.x_vector_only_checkbox.value
            self.terminal.add_log(f"正在提取声音特征: {os.path.basename(self._ref_audio_path)}")
            self._page.update()

            return await tts_engine.create_voice_clone_prompt_async(
                ref_audio=self._ref_audio_path,
                ref_text=ref_text.strip(),
                x_vector_only=x_vector_only
            )

    # ==================== 特有方法：克隆管理 ====================

    def _on_saved_clone_changed(self, e):
        """已保存克隆选择改变事件"""
        self.terminal.add_log(f"已选择克隆: {self.saved_clone_dropdown.value}")

    def _on_clone_mode_change(self, e):
        """克隆模式切换事件"""
        mode = self.use_saved_clone_radio.value

        if mode == "saved":
            self._update_saved_clone_dropdown()
            self.saved_clone_dropdown.visible = True
            self.ref_audio_button.visible = False
            self.ref_audio_status.visible = False
            self.ref_text_input.visible = False
        else:
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
        self._last_saved_clone_ref_audio = None
        self._last_saved_clone_ref_text = None

    def _update_saved_clone_dropdown(self):
        """更新已保存克隆下拉框"""
        clones = self.voice_library.get_all_clones()
        options = [
            ft.DropdownOption(text=c["name"], key=c["id"])
            for c in clones
        ]
        self.saved_clone_dropdown.options = options
        self.saved_clone_dropdown.update()

    def _refresh_clone_library(self):
        """刷新克隆声音库"""
        self.clone_library_grid.controls.clear()
        clones = self.voice_library.get_all_clones()

        if not clones:
            self.clone_library_grid.controls.append(ft.Text("暂无克隆", size=12))
        else:
            for clone in clones:
                card = ft.Container(
                    content=ft.Column([
                        ft.Text(clone["name"], size=13, weight=ft.FontWeight.BOLD),
                        ft.Text(clone["created_at"][:10], size=11),
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

        try:
            self.clone_library_grid.update()
        except RuntimeError:
            pass

    def _on_preview_clone(self, e, clone: dict):
        """预览克隆声音"""
        self.terminal.add_log(f"试听克隆: {clone['name']}")

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

    async def _on_pick_file(self, e):
        """文件选择按钮点击事件"""
        try:
            result = await self.file_picker.pick_files(
                allowed_extensions=["wav", "mp3", "flac"]
            )
            if result and len(result) > 0:
                self._ref_audio_path = result[0].path
                filename = os.path.basename(self._ref_audio_path)
                self.ref_audio_status.value = f"已选择: {filename}"
                self.ref_audio_status.update()
                self.terminal.add_log(f"已选择参考音频: {filename}")

                self._last_saved_clone_ref_audio = None
                self._last_saved_clone_ref_text = None
        except Exception as ex:
            logger.error(f"选择文件失败: {str(ex)}", exc_info=True)
            self.terminal.add_log(f"选择文件失败: {str(ex)}")
