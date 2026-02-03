"""
PhantomVox 主 UI 控制器

基于 Flet 的文本转语音应用界面
采用 NavigationRail 架构
"""

import flet as ft
import logging
import asyncio
from typing import Optional

from ui.components.app_bar import PhantomAppBar
from ui.components.custom_voice_view import CustomVoiceView
from ui.components.voice_design_view import VoiceDesignView
from ui.components.voice_clone_view import VoiceCloneView
from ui.components.voice_library import VoiceLibrary
from ui.components.settings_view import SettingsView
from core.terminal import AsyncTerminal
from core.model_manager import ModelManager
from config.config_manager import ConfigManager
from tts.qwen_engine import QwenEngine
from tts.audio_manager import AudioManager

logger = logging.getLogger(__name__)


class PhantomUI:
    """PhantomVox 主 UI 控制器 - NavigationRail 架构"""

    def __init__(self, page: ft.Page, version: str):
        self.page = page
        self.version = version

        # 初始化核心组件
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config

        # 初始化终端
        self.terminal = AsyncTerminal(page)

        # 初始化模型管理器
        self.model_manager = ModelManager(
            models_dir=self.config_manager.get("model.model_path", "./models") or "./models",
            config_manager=self.config_manager
        )

        # 初始化声音库管理器
        self.voice_library = VoiceLibrary(self.config_manager)

        # 懒加载组件
        self._tts_engine: Optional[QwenEngine] = None
        self._audio_manager: Optional[AudioManager] = None
        self._engine_lock = asyncio.Lock()

        # 当前视图索引
        self._current_view_index = 0

        # 当前选择的模型（用于初始化 TTS 引擎）
        self._current_model_id = None

        # 初始化时设置默认模型
        usable_models = self.model_manager.list_usable_models()
        if usable_models:
            self._current_model_id = usable_models[0]

        # 三个新视图（延迟初始化）
        self.custom_voice_view = None
        self.voice_design_view = None
        self.voice_clone_view = None

        # 设置视图（延迟初始化）
        self.settings_view = None

        # UI 样式配置
        self.BStyle = ft.ButtonStyle(
            icon_size=20,
            text_style=ft.TextStyle(size=14, font_family="Microsoft YaHei")
        )

        # 当前生成的音频
        self._last_audio = None

        # 文件选择器
        self._file_picker = ft.FilePicker()

        # 终端展开状态
        self._terminal_expanded = True

        # 初始化 AppBar
        self.app_bar = PhantomAppBar(
            page=page,
            version=version,
            on_theme_toggle=self._on_theme_toggle,
            on_close=self._on_close_window
        )

        # 创建 UI 组件
        self._create_ui_components()

        logger.info("PhantomVox UI 初始化完成")

    @property
    def tts_engine(self) -> QwenEngine:
        """懒加载：TTS 引擎"""
        if self._tts_engine is None:
            self.terminal.add_log("正在初始化 TTS 引擎...")

            try:
                # 使用 _current_model_id 或尝试获取选中的模型
                model_id = self._current_model_id

                self.terminal.add_log(f"DEBUG: _current_model_id = {model_id}")

                # 如果没有指定模型，尝试从下拉框获取
                if not model_id:
                    model_id = getattr(self, 'model_dropdown', None)
                    if model_id is not None:
                        model_id = model_id.value
                    self.terminal.add_log(f"DEBUG: 从 model_dropdown 获取 model_id = {model_id}")

                # 如果仍然没有模型，使用第一个可用模型
                if not model_id:
                    usable_models = self.model_manager.list_usable_models()
                    self.terminal.add_log(f"DEBUG: 可用模型列表 = {usable_models}")

                    if not usable_models:
                        # 没有可用的模型
                        self.terminal.add_log("✗ 没有可用的 TTS 模型，请先在「模型管理」中下载模型")

                        # 检查是否有已安装但不可用的模型
                        installed = self.model_manager.get_installed_models()
                        if installed:
                            self.terminal.add_log("提示: 已安装的模型缺少依赖，请重新下载")

                        raise RuntimeError("没有可用的 TTS 模型")

                    model_id = usable_models[0]
                    self.terminal.add_log(f"DEBUG: 使用第一个可用模型 = {model_id}")

                model_path = self.model_manager.get_model_path(model_id)

                if not model_path:
                    raise RuntimeError(f"模型路径无效: {model_id}")

                device = self.config_manager.get("model.device", "cuda:0")
                dtype = self.config_manager.get("model.dtype", None)
                attn_implementation = self.config_manager.get("model.attn_implementation", None)

                model_info = self.model_manager.get_model_info(model_id)
                self.terminal.add_log(f"使用模型: {model_info.name if model_info else model_id}")
                self.terminal.add_log(f"模型ID: {model_id}")
                self.terminal.add_log(f"模型路径: {model_path}")
                self.terminal.add_log(f"设备: {device}")

                # 根据模型ID确定模型类型
                if "customvoice" in model_id:
                    model_type = "CustomVoice"
                elif "voicedesign" in model_id:
                    model_type = "VoiceDesign"
                elif "base" in model_id:
                    model_type = "Base"
                else:
                    model_type = None  # 默认类型

                self._tts_engine = QwenEngine(
                    model_path=str(model_path),
                    model_type=model_type,
                    device=device,
                    dtype=dtype,
                    attn_implementation=attn_implementation
                )

                self.terminal.add_log("✓ TTS 引擎初始化完成")

            except Exception as e:
                self.terminal.add_log(f"✗ TTS 引擎初始化失败: {str(e)}")
                raise

        return self._tts_engine

    @property
    def audio_manager(self) -> AudioManager:
        """懒加载：音频管理器"""
        if self._audio_manager is None:
            self._audio_manager = AudioManager(self.page)
        return self._audio_manager

    def _create_ui_components(self):
        """创建 UI 组件"""

        # ========== 模型管理视图组件 ==========
        self.model_list = ft.ListView(
            expand=True,
            spacing=10,
            padding=10
        )

        self.refresh_models_button = ft.Button(
            "刷新列表",
            icon=ft.Icons.REFRESH,
            style=self.BStyle,
            on_click=self.on_refresh_models_click
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

    def build_navigation_rail(self) -> ft.NavigationRail:
        """构建导航栏"""
        rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=200,
            destinations=[
                ft.NavigationRailDestination(
                    icon=ft.Icons.MIC,
                    selected_icon=ft.Icons.MIC,
                    label="自定义语音",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.PALETTE,
                    selected_icon=ft.Icons.PALETTE,
                    label="声音设计",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.CONTENT_COPY,
                    selected_icon=ft.Icons.COPY_ALL,
                    label="声音克隆",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.DOWNLOAD,
                    selected_icon=ft.Icons.DOWNLOAD,
                    label="模型管理",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.SETTINGS,
                    selected_icon=ft.Icons.SETTINGS,
                    label="设置",
                ),
            ],
            on_change=self.on_navigation_change,
        )

        self.navigation_rail = rail
        return rail

    def on_navigation_change(self, e):
        """导航切换事件"""
        self._current_view_index = e.control.selected_index

        # 更新内容区域
        self._update_content_area()

    def _update_content_area(self):
        """更新内容区域"""
        # 清空内容区域
        self.content_area.controls.clear()

        # 根据选中的索引添加对应视图
        if self._current_view_index == 0:
            # 自定义语音页面
            view = self._get_custom_voice_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 1:
            # 声音设计页面
            view = self._get_voice_design_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 2:
            # 声音克隆页面
            view = self._get_voice_clone_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 3:
            # 模型管理页面
            self.content_area.controls.append(self._build_model_view())
        elif self._current_view_index == 4:
            # 设置页面
            view = self._get_settings_view()
            self.content_area.controls.append(view)

        self.content_area.update()

    def _get_custom_voice_view(self) -> ft.Control:
        """获取自定义语音视图（延迟初始化）"""
        if self.custom_voice_view is None:
            self.custom_voice_view = CustomVoiceView(
                page=self.page,
                tts_engine_getter=lambda: self.tts_engine,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(model_id)
            )
        return self.custom_voice_view

    def _get_voice_design_view(self) -> ft.Control:
        """获取声音设计视图（延迟初始化）"""
        if self.voice_design_view is None:
            self.voice_design_view = VoiceDesignView(
                page=self.page,
                tts_engine_getter=lambda: self.tts_engine,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(model_id)
            )
        return self.voice_design_view

    def _get_voice_clone_view(self) -> ft.Control:
        """获取声音克隆视图（延迟初始化）"""
        if self.voice_clone_view is None:
            self.voice_clone_view = VoiceCloneView(
                page=self.page,
                tts_engine_getter=lambda: self.tts_engine,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(model_id)
            )
        return self.voice_clone_view

    def _get_settings_view(self) -> ft.Control:
        """获取设置视图（延迟初始化）"""
        if self.settings_view is None:
            self.settings_view = SettingsView(
                page=self.page,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_settings_changed=self._on_settings_changed
            )
        return self.settings_view

    def _on_settings_changed(self):
        """处理设置更改事件"""
        # 如果 TTS 引擎已初始化，清除它以强制重新初始化
        if self._tts_engine is not None:
            self._tts_engine = None
            self.terminal.add_log("设置已更改，TTS 引擎将重新初始化")

    def _clear_tts_engine_cache(self, model_id: str = None):
        """清除 TTS 引擎缓存以强制重新初始化

        Args:
            model_id: 可选，指定要使用的模型 ID
        """
        self.terminal.add_log(f"DEBUG: _clear_tts_engine_cache 被调用，model_id = {model_id}")
        self.terminal.add_log(f"DEBUG: 更新前 _current_model_id = {self._current_model_id}")

        if model_id:
            self._current_model_id = model_id
            self.terminal.add_log(f"DEBUG: 已设置 _current_model_id = {self._current_model_id}")

            # 同步所有视图的下拉框
            self._sync_model_dropdowns(model_id)

        if self._tts_engine is not None:
            self._tts_engine = None
            self.terminal.add_log("TTS 引擎缓存已清除，将使用新选择的模型")
        else:
            self.terminal.add_log("TTS 引擎未初始化，无需清除缓存")

    def _sync_model_dropdowns(self, model_id: str):
        """同步所有视图的模型下拉框选择

        Args:
            model_id: 要同步到的模型 ID
        """
        if self.custom_voice_view and hasattr(self.custom_voice_view, 'model_dropdown'):
            self.custom_voice_view.model_dropdown.value = model_id
            try:
                self.custom_voice_view.model_dropdown.update()
            except RuntimeError:
                pass

        if self.voice_design_view and hasattr(self.voice_design_view, 'model_dropdown'):
            self.voice_design_view.model_dropdown.value = model_id
            try:
                self.voice_design_view.model_dropdown.update()
            except RuntimeError:
                pass

        if self.voice_clone_view and hasattr(self.voice_clone_view, 'model_dropdown'):
            self.voice_clone_view.model_dropdown.value = model_id
            try:
                self.voice_clone_view.model_dropdown.update()
            except RuntimeError:
                pass

    def _refresh_all_model_dropdowns(self):
        """刷新所有视图的模型下拉框选项"""
        # 刷新 CustomVoiceView
        if self.custom_voice_view and hasattr(self.custom_voice_view, 'refresh_model_dropdown'):
            try:
                self.custom_voice_view.refresh_model_dropdown()
            except Exception as e:
                logger.error(f"刷新 CustomVoiceView 模型下拉框失败: {str(e)}")

        # 刷新 VoiceDesignView
        if self.voice_design_view and hasattr(self.voice_design_view, 'refresh_model_dropdown'):
            try:
                self.voice_design_view.refresh_model_dropdown()
            except Exception as e:
                logger.error(f"刷新 VoiceDesignView 模型下拉框失败: {str(e)}")

        # 刷新 VoiceCloneView
        if self.voice_clone_view and hasattr(self.voice_clone_view, 'refresh_model_dropdown'):
            try:
                self.voice_clone_view.refresh_model_dropdown()
            except Exception as e:
                logger.error(f"刷新 VoiceCloneView 模型下拉框失败: {str(e)}")

    def _build_model_view(self) -> ft.Control:
        """构建模型管理视图"""
        # 填充模型列表
        self._populate_model_list()

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

    def _build_settings_view(self) -> ft.Control:
        """构建设置视图"""
        return ft.Column([
            ft.Text("设置", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
            ft.Text("设置功能开发中...", size=16, color=ft.Colors.GREY),
        ], expand=True)

    def _populate_model_list(self):
        """填充模型列表"""
        self.model_list.controls.clear()

        available_models = self.model_manager.list_available_models()
        installed_models = self.model_manager.get_installed_models()

        # 按类别分组模型
        categories = {
            "分词器": [],
            "1.7B 系列": [],
            "0.6B 系列": [],
        }

        for model_id, model_info in available_models.items():
            is_installed = model_id in installed_models
            is_usable, status_msg = self.model_manager.check_model_usable(model_id)

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
                if model_info.dependencies:
                    dep_names = []
                    for dep_id in model_info.dependencies:
                        dep_model_info = self.model_manager.get_model_info(dep_id)
                        if dep_model_info:
                            dep_installed = dep_id in installed_models
                            dep_status = "✓" if dep_installed else "✗"
                            dep_names.append(f"{dep_status} {dep_model_info.name}")
                    if dep_names:
                        dep_info = f"\n依赖: {', '.join(dep_names)}"

                # 状态标签
                if is_usable:
                    status_text = "可用"
                    status_color = ft.Colors.GREEN
                    status_bg = ft.Colors.with_opacity(0.1, ft.Colors.GREEN)
                elif is_installed:
                    status_text = "不可用"
                    status_color = ft.Colors.ORANGE
                    status_bg = ft.Colors.with_opacity(0.1, ft.Colors.ORANGE)
                else:
                    status_text = "未安装"
                    status_color = ft.Colors.GREY
                    status_bg = ft.Colors.with_opacity(0.1, ft.Colors.GREY)

                # 模型卡片 - 构建控件列表
                card_controls = [
                    ft.Row([
                        ft.Text(model_info.name, size=15, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=ft.Text(
                                status_text,
                                size=11,
                                color=status_color
                            ),
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            bgcolor=status_bg,
                            border_radius=12
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Container(height=3),
                    ft.Text(f"大小: {model_info.size}", size=12, color=ft.Colors.GREY_400),
                    ft.Text(model_info.description, size=12, color=ft.Colors.GREY_400),
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
                            on_click=lambda e, mid=model_id: self.on_download_model_click(e, mid),
                            width=100
                        ),
                        ft.Button(
                            "删除",
                            icon=ft.Icons.DELETE,
                            style=self.BStyle,
                            on_click=lambda e, mid=model_id: self.on_delete_model_click(e, mid),
                            width=80,
                            disabled=not is_installed
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

    def build_main_view(self) -> ft.Control:
        """构建主界面"""
        # 创建导航栏
        rail = self.build_navigation_rail()

        # 创建内容区域 - 默认显示第一个导航页面（自定义语音）
        self.content_area = ft.Column([
            self._get_custom_voice_view()
        ], expand=True)

        # 创建终端日志容器引用（用于折叠/展开）
        self._terminal_logs_container = ft.Container(
            content=self.terminal.logs,
            border=ft.Border.all(1, ft.Colors.GREY_400),
            border_radius=8,
            padding=5,
            height=150,
        )

        # 创建右侧主容器（包含内容和终端）
        right_panel = ft.Column([
            # 内容区域
            ft.Container(
                content=self.content_area,
                expand=True,
                padding=20
            ),

            # 分隔线
            ft.Divider(height=1, color=ft.Colors.GREY_300),

            # 全局固定终端
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("运行日志", size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.IconButton(
                            icon=ft.Icons.CLEAR,
                            icon_size=18,
                            tooltip="清空日志",
                            on_click=self._on_clear_terminal
                        ),
                        ft.IconButton(
                            icon=ft.Icons.EXPAND_LESS,
                            icon_size=18,
                            tooltip="折叠/展开",
                            on_click=self._on_toggle_terminal
                        )
                    ], spacing=10),
                    self._terminal_logs_container,
                ], spacing=5),
                padding=ft.padding.symmetric(horizontal=20, vertical=10),
                bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
            )
        ], expand=True, spacing=0)

        # 主布局
        main_view = ft.Row([
            # 左侧导航栏
            rail,
            ft.VerticalDivider(width=1),
            # 右侧主容器
            right_panel
        ], expand=True)

        return main_view

    # ========== 事件处理方法 ==========

    def on_refresh_models_click(self, e):
        """刷新模型列表"""
        self._populate_model_list()
        self.page.show_dialog(ft.SnackBar(ft.Text("列表已刷新")))

    def on_download_model_click(self, e, model_id: str):
        """下载模型按钮点击事件"""
        model_info = self.model_manager.get_model_info(model_id)
        if not model_info:
            return

        # 检查 ModelScope 是否安装
        if not self.model_manager._check_modelscope():
            self.page.show_dialog(
                ft.AlertDialog(
                    title=ft.Text("缺少依赖"),
                    content=ft.Text(
                        "ModelScope 未安装。\n\n"
                        "请在终端运行以下命令安装：\n"
                        "pip install modelscope"
                    ),
                    actions=[
                        ft.TextButton("确定", on_click=lambda _: self.page.pop_dialog())
                    ]
                )
            )
            return

        self.terminal.add_log(f"开始下载模型: {model_info.name}")

        # 显示依赖信息
        if model_info.dependencies:
            dep_names = []
            for dep_id in model_info.dependencies:
                dep_model_info = self.model_manager.get_model_info(dep_id)
                if dep_model_info:
                    dep_names.append(dep_model_info.name)
            self.terminal.add_log(f"  包含依赖: {', '.join(dep_names)}")

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
        self.page.show_dialog(progress_dialog)

        # 在后台线程中下载
        def download_in_background():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def download():
                try:
                    # 使用带依赖的下载方法
                    success = await self.model_manager.download_model_with_dependencies(
                        model_id,
                        progress_callback=None  # 不需要更新UI
                    )

                    # 使用 run_task 在主线程中执行 UI 操作
                    async def update_ui_on_success():
                        self.terminal.add_log(f"✓ 模型下载完成: {model_info.name}")
                        # 刷新模型管理页面的列表
                        self._populate_model_list()
                        # 刷新各个视图的模型下拉框
                        self._refresh_all_model_dropdowns()
                        # 关闭对话框
                        self.page.pop_dialog()
                        # 显示成功提示
                        self.page.show_dialog(ft.SnackBar(ft.Text(f"✓ {model_info.name} 下载完成")))

                    async def update_ui_on_failure():
                        self.terminal.add_log(f"✗ 下载失败")
                        self.page.pop_dialog()
                        self.page.show_dialog(ft.SnackBar(ft.Text("✗ 下载失败")))

                    if success:
                        self.page.run_task(update_ui_on_success)
                    else:
                        self.page.run_task(update_ui_on_failure)

                except Exception as ex:
                    self.terminal.add_log(f"✗ 下载失败: {str(ex)}")
                    logger.exception("模型下载异常")
                    try:
                        self.page.pop_dialog()
                        async def show_error_dialog():
                            self.page.show_dialog(ft.SnackBar(ft.Text(f"✗ 下载失败: {str(ex)}")))
                        self.page.run_task(show_error_dialog)
                    except:
                        pass

            loop.run_until_complete(download())
            loop.close()

        import threading
        thread = threading.Thread(target=download_in_background, daemon=True)
        thread.start()

    def _on_download_progress(self, model_id: str, progress: float, status: str):
        """下载进度回调"""
        self.terminal.add_log(f"[{model_id}] {status}")

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

        except Exception as e:
            logger.exception("更新下载进度UI失败")

    def on_delete_model_click(self, e, model_id: str):
        """删除模型按钮点击事件"""
        model_info = self.model_manager.get_model_info(model_id)
        if not model_info:
            return

        # 确认对话框
        def confirm_delete(dialog):
            async def delete():
                try:
                    success = await self.model_manager.delete_model(model_id)
                    if success:
                        self.terminal.add_log(f"✓ 模型已删除: {model_info.name}")
                        # 刷新模型管理页面的列表
                        self._populate_model_list()
                        # 刷新各个视图的模型下拉框
                        self._refresh_all_model_dropdowns()
                        # 刷新整个视图以确保UI更新
                        self.model_list.update()
                    else:
                        self.terminal.add_log(f"✗ 删除失败")
                finally:
                    self.page.pop_dialog()

            self.page.run_task(delete)

        dialog = ft.AlertDialog(
            title=ft.Text("确认删除"),
            content=ft.Text(f"确定要删除模型 \"{model_info.name}\" 吗？\n此操作不可撤销。"),
            actions=[
                ft.TextButton("取消", on_click=lambda _: self.page.pop_dialog()),
                ft.TextButton("删除", on_click=lambda _: confirm_delete(dialog)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.show_dialog(dialog)

    def _on_theme_toggle(self, new_theme_mode):
        """
        处理主题切换事件

        Args:
            new_theme_mode: 新的主题模式 (ft.ThemeMode.LIGHT 或 ft.ThemeMode.DARK)
        """
        try:
            # 保存主题偏好到配置
            theme_str = "light" if new_theme_mode == ft.ThemeMode.LIGHT else "dark"
            self.config_manager.set("theme_mode", theme_str)
            self.config_manager.save_config()

            # 记录日志
            theme_name = "亮色" if new_theme_mode == ft.ThemeMode.LIGHT else "深色"
            self.terminal.add_log(f"✓ 主题已切换至: {theme_name}")

            logger.info(f"主题切换至: {theme_str}")
        except Exception as e:
            logger.exception("主题切换失败")
            self.terminal.add_log(f"✗ 主题切换失败: {str(e)}")

    def _on_close_window(self, e):
        """
        处理窗口关闭事件，显示确认对话框
        """
        async def close_app_async(dialog):
            """异步关闭应用"""
            try:
                # 关闭对话框
                try:
                    self.page.pop_dialog()
                except RuntimeError:
                    # 会话已关闭，继续清理
                    pass

                # 保存配置
                self.config_manager.save_config()

                # 隐藏窗口（立即响应）
                try:
                    self.page.window.visible = False
                    self.page.window.prevent_close = False
                    self.page.update()
                except RuntimeError:
                    # 会话已关闭，继续清理
                    pass

                # 关闭窗口
                try:
                    await self.page.window.close()
                except RuntimeError as e:
                    # 如果是会话已关闭错误，这是正常的，不需要记录
                    if "Session closed" not in str(e):
                        raise
            except Exception as ex:
                logger.exception("关闭应用时出错")
                try:
                    await self.page.window.destroy()
                except:
                    pass

        def confirm_close(dialog):
            """确认关闭操作"""
            # 使用 run_task 执行异步关闭操作
            self.page.run_task(close_app_async, dialog)

        # 显示确认对话框
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("确认退出"),
            content=ft.Text("确定要退出 PhantomVox 吗？"),
            actions=[
                ft.TextButton(
                    "取消",
                    on_click=lambda _: self.page.pop_dialog()
                ),
                ft.TextButton(
                    "退出",
                    on_click=lambda _: confirm_close(dialog)
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.show_dialog(dialog)

    def _on_clear_terminal(self, e):
        # 清空终端日志
        self.terminal.clear_terminal()

    def _on_toggle_terminal(self, e):
        # 折叠/展开终端
        if self._terminal_expanded:
            # 折叠
            self._terminal_logs_container.height = 0
            self._terminal_logs_container.visible = False
            e.control.icon = ft.Icons.EXPAND_MORE
        else:
            # 展开
            self._terminal_logs_container.height = 150
            self._terminal_logs_container.visible = True
            e.control.icon = ft.Icons.EXPAND_LESS
        self._terminal_expanded = not self._terminal_expanded
        self._terminal_logs_container.update()
        e.control.update()





