"""
模型管理页面组件 (Model Manager View)

提供模型下载、删除、状态查看等功能
"""

import flet as ft
import logging
import asyncio
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ModelManagerView(ft.Container):
    """模型管理页面组件"""

    def __init__(
        self,
        page: ft.Page,
        model_manager,
        terminal,
        on_models_changed: Optional[Callable] = None
    ):
        """
        初始化模型管理视图

        Args:
            page: Flet Page 对象
            model_manager: 模型管理器实例
            terminal: 终端日志组件
            on_models_changed: 模型变更回调函数（用于刷新其他视图的下拉框）
        """
        self._page = page
        self._model_manager = model_manager
        self._terminal = terminal
        self._on_models_changed = on_models_changed

        # UI 样式配置
        self.BStyle = ft.ButtonStyle(
            icon_size=20,
            text_style=ft.TextStyle(size=14, font_family="Microsoft YaHei")
        )

        # ========== 模型列表组件 ==========
        self.model_list = ft.ListView(
            expand=True,
            spacing=10,
            padding=10
        )

        self.refresh_models_button = ft.Button(
            "刷新列表",
            icon=ft.Icons.REFRESH,
            style=self.BStyle,
            on_click=self._on_refresh_models_click
        )

        # 下载进度显示组件
        self._download_percent_ref = ft.Ref[ft.Text]()
        self._download_progress_ref = ft.Ref[ft.ProgressBar]()
        self._download_status_ref = ft.Ref[ft.Text]()

        self.download_progress_container = ft.Container(
            visible=False,  # 默认隐藏
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.DOWNLOAD, size=20),
                    ft.Text("下载中...", size=14, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    ft.Text("0%", size=14, ref=self._download_percent_ref)
                ], spacing=10),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                ft.ProgressBar(
                    width=400,
                    bar_height=8,
                    color=ft.Colors.BLUE,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE),
                    ref=self._download_progress_ref
                ),
                ft.Divider(height=5, color=ft.Colors.TRANSPARENT),
                ft.Text(
                    "",
                    size=12,
                    color=ft.Colors.GREY_400,
                    ref=self._download_status_ref
                )
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            padding=15,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLUE),
            border_radius=8,
            margin=ft.margin.only(bottom=10)
        )

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

        # 初始化时填充模型列表
        self._populate_model_list()

    def _build_ui(self) -> ft.Control:
        """构建主UI界面"""
        return ft.Column([
            ft.Row([
                ft.Text("模型管理", size=24, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                self.refresh_models_button
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

            # 下载进度显示
            self.download_progress_container,

            ft.Container(
                content=self.model_list,
                bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
                border_radius=12,
                padding=15,
                expand=True
            )

        ], scroll=ft.ScrollMode.AUTO, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def _populate_model_list(self):
        """填充模型列表"""
        self.model_list.controls.clear()

        available_models = self._model_manager.list_available_models()
        installed_models = self._model_manager.get_installed_models()

        # 按类别分组模型
        categories = {
            "分词器": [],
            "1.7B 系列": [],
            "0.6B 系列": [],
        }

        for model_id, model_info in available_models.items():
            is_installed = model_id in installed_models
            is_usable, status_msg = self._model_manager.check_model_usable(model_id)

            # 确定类别
            if "tokenizer" in model_id:
                categories["分词器"].append((model_id, model_info, is_installed, is_usable, status_msg))
            elif "0.6b" in model_id:
                categories["0.6B 系列"].append((model_id, model_info, is_installed, is_usable, status_msg))
            elif "1.7b" in model_id:
                categories["1.7B 系列"].append((model_id, model_info, is_installed, is_usable, status_msg))

        # 为每个类别创建卡片组
        for category, models in categories.items():
            if not models:
                continue

            # 类别标题
            self.model_list.controls.append(
                ft.Text(category, size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_GREY_300)
            )

            for model_id, model_info, is_installed, is_usable, status_msg in models:
                # 构建依赖信息
                dep_info = ""
                is_required = getattr(model_info, 'required', False)
                if model_info.dependencies:
                    dep_names = []
                    for dep_id in model_info.dependencies:
                        dep_model_info = self._model_manager.get_model_info(dep_id)
                        if dep_model_info:
                            dep_installed = dep_id in installed_models
                            dep_status = "✓" if dep_installed else "✗"
                            dep_names.append(f"{dep_status} {dep_model_info.name}")
                    if dep_names:
                        dep_info = f"\n依赖: {', '.join(dep_names)}"

                # 状态标签和必装标识
                if is_usable:
                    status_text = "可用"
                    status_bg = ft.Colors.with_opacity(0.1, ft.Colors.GREEN)
                elif is_installed:
                    status_text = "不可用"
                    status_bg = ft.Colors.with_opacity(0.1, ft.Colors.ORANGE)
                else:
                    status_text = "未安装"
                    status_bg = ft.Colors.with_opacity(0.1, ft.Colors.GREY)

                # 必装标识
                required_badge = None
                if is_required:
                    required_badge = ft.Container(
                        content=ft.Text(
                            "必装",
                            size=10,
                            color=ft.Colors.RED,
                            weight=ft.FontWeight.BOLD
                        ),
                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.RED),
                        border_radius=8
                    )

                # 模型卡片 - 构建控件列表
                # 构建右侧标签行（根据是否必装动态添加）
                right_badges = [ft.Container(
                    content=ft.Text(
                        status_text,
                        size=11
                    ),
                    padding=ft.padding.symmetric(horizontal=8, vertical=4),
                    bgcolor=status_bg,
                    border_radius=12
                )]
                if is_required:
                    right_badges.insert(0, required_badge)

                card_controls = [
                    ft.Row([
                        ft.Text(model_info.name, size=15, weight=ft.FontWeight.BOLD),
                        ft.Row(right_badges, spacing=5)
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Container(height=3),
                    ft.Text(f"大小: {model_info.size}", size=12),
                    ft.Text(model_info.description, size=12),
                ]

                # 如果有依赖信息，添加到列表
                if dep_info:
                    card_controls.append(ft.Text(dep_info, size=11, color=ft.Colors.GREY_500))

                # 添加按钮行
                card_controls.extend([
                    ft.Container(height=8),
                    ft.Row([
                        ft.Button(
                            "下载" if not is_installed else "重新下载",
                            icon=ft.Icons.DOWNLOAD,
                            style=self.BStyle,
                            on_click=lambda e, mid=model_id: self._on_download_model_click(e, mid),
                            width=100
                        ),
                        ft.Button(
                            "删除",
                            icon=ft.Icons.DELETE,
                            style=self.BStyle,
                            on_click=lambda e, mid=model_id: self._on_delete_model_click(e, mid),
                            width=80,
                            disabled=not is_installed or is_required  # 必装模型禁用删除按钮
                        ),
                    ], spacing=8)
                ])

                # 模型卡片
                card = ft.Card(
                    content=ft.Container(
                        content=ft.Column(card_controls, spacing=3),
                        padding=12,
                        border_radius=8
                    ),
                    elevation=1
                )

                self.model_list.controls.append(card)

            # 类别之间的间隔
            self.model_list.controls.append(ft.Container(height=15))

        # 刷新模型列表显示
        try:
            self.model_list.update()
        except Exception as e:
            logger.debug(f"模型列表更新失败: {e}")

    def refresh_model_list(self):
        """刷新模型列表（外部调用接口）"""
        self._populate_model_list()

    # ========== 事件处理方法 ==========

    def _on_refresh_models_click(self, e):
        """刷新模型列表"""
        self._populate_model_list()
        self._page.show_dialog(ft.SnackBar(ft.Text("列表已刷新")))

    def _on_download_model_click(self, e, model_id: str):
        """下载模型按钮点击事件"""
        model_info = self._model_manager.get_model_info(model_id)
        if not model_info:
            return

        # 检查 ModelScope 是否安装
        if not self._model_manager._check_modelscope():
            self._page.show_dialog(
                ft.AlertDialog(
                    title=ft.Text("缺少依赖"),
                    content=ft.Text(
                        "ModelScope 未安装。\n\n"
                        "请在终端运行以下命令安装：\n"
                        "pip install modelscope"
                    ),
                    actions=[
                        ft.TextButton("确定", on_click=lambda _: self._page.pop_dialog())
                    ]
                )
            )
            return

        self._terminal.add_log(f"开始下载模型: {model_info.name}")

        # 显示依赖信息
        if model_info.dependencies:
            dep_names = []
            for dep_id in model_info.dependencies:
                dep_model_info = self._model_manager.get_model_info(dep_id)
                if dep_model_info:
                    dep_names.append(dep_model_info.name)
            self._terminal.add_log(f"  包含依赖: {', '.join(dep_names)}")

        # 创建圆形进度对话框
        progress_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("正在下载模型", size=16),
            content=ft.Column([
                ft.Row([
                    ft.ProgressRing(stroke_width=3, width=30, height=30),
                    ft.Text("   请在终端查看下载进度", size=14, color=ft.Colors.GREY_400)
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
            actions=[]  # 无操作按钮，下载完成自动关闭
        )

        # 显示进度对话框
        self._page.show_dialog(progress_dialog)

        # 在后台线程中下载
        def download_in_background():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def download():
                try:
                    # 使用带依赖的下载方法
                    success = await self._model_manager.download_model_with_dependencies(
                        model_id,
                        progress_callback=None  # 不需要更新UI
                    )

                    # 使用 run_task 在主线程中执行 UI 操作
                    async def update_ui_on_success():
                        self._terminal.add_log(f"✓ 模型下载完成: {model_info.name}")
                        # 刷新模型管理页面的列表
                        self._populate_model_list()
                        # 触发模型变更回调
                        if self._on_models_changed:
                            self._on_models_changed()
                        # 关闭对话框
                        self._page.pop_dialog()
                        # 显示成功提示
                        self._page.show_dialog(ft.SnackBar(ft.Text(f"✓ {model_info.name} 下载完成")))

                    async def update_ui_on_failure():
                        self._terminal.add_log("✗ 下载失败")
                        self._page.pop_dialog()
                        self._page.show_dialog(ft.SnackBar(ft.Text("✗ 下载失败")))

                    if success:
                        self._page.run_task(update_ui_on_success)
                    else:
                        self._page.run_task(update_ui_on_failure)

                except Exception as ex:
                    self._terminal.add_log(f"✗ 下载失败: {str(ex)}")
                    logger.exception("模型下载异常")
                    error_message = str(ex)
                    try:
                        self._page.pop_dialog()
                        async def show_error_dialog(msg=error_message):
                            self._page.show_dialog(ft.SnackBar(ft.Text(f"✗ 下载失败: {msg}")))
                        self._page.run_task(show_error_dialog)
                    except Exception:
                        pass

            loop.run_until_complete(download())
            loop.close()

        thread = threading.Thread(target=download_in_background, daemon=True)
        thread.start()

    def _on_download_progress(self, model_id: str, progress: float, status: str):
        """下载进度回调"""
        self._terminal.add_log(f"[{model_id}] {status}")

        # 更新进度UI
        try:
            # 显示进度容器
            self.download_progress_container.visible = True

            # 更新进度条
            if self._download_progress_ref.current:
                self._download_progress_ref.current.value = progress / 100  # ProgressBar 使用 0-1 范围

            # 更新百分比文本
            if self._download_percent_ref.current:
                self._download_percent_ref.current.value = f"{progress:.0f}%"

            # 更新状态文本
            if self._download_status_ref.current:
                self._download_status_ref.current.value = status

            # 刷新显示
            self.download_progress_container.update()

        except Exception:
            logger.exception("更新下载进度UI失败")

    def _on_delete_model_click(self, e, model_id: str):
        """删除模型按钮点击事件"""
        model_info = self._model_manager.get_model_info(model_id)
        if not model_info:
            return

        # 检查是否为必装模型
        is_required = getattr(model_info, 'required', False)
        if is_required:
            self._page.show_dialog(
                ft.AlertDialog(
                    title=ft.Text("无法删除"),
                    content=ft.Text(f"\"{model_info.name}\" 是必装模型，不能删除。"),
                    actions=[
                        ft.TextButton("确定", on_click=lambda _: self._page.pop_dialog())
                    ]
                )
            )
            return

        # 确认对话框
        def confirm_delete(dialog):
            async def delete():
                try:
                    success = await self._model_manager.delete_model(model_id)
                    if success:
                        self._terminal.add_log(f"✓ 模型已删除: {model_info.name}")
                        # 刷新模型管理页面的列表
                        self._populate_model_list()
                        # 触发模型变更回调
                        if self._on_models_changed:
                            self._on_models_changed()
                        # 刷新整个视图以确保UI更新
                        self.model_list.update()
                    else:
                        self._terminal.add_log("✗ 删除失败")
                finally:
                    self._page.pop_dialog()

            self._page.run_task(delete)

        dialog = ft.AlertDialog(
            title=ft.Text("确认删除"),
            content=ft.Text(f"确定要删除模型 \"{model_info.name}\" 吗？\n此操作不可撤销。"),
            actions=[
                ft.TextButton("取消", on_click=lambda _: self._page.pop_dialog()),
                ft.TextButton("删除", on_click=lambda _: confirm_delete(dialog)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self._page.show_dialog(dialog)
