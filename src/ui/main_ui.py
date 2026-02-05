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
from ui.components.about_view import AboutView
from ui.components.model_manager_view import ModelManagerView
from ui.components.tts_service_view import TTSServiceView
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

        # 不要在初始化时设置默认模型，让 TTS 引擎根据当前页面自动选择

        # 三个新视图（延迟初始化）
        self.custom_voice_view = None
        self.voice_design_view = None
        self.voice_clone_view = None

        # 设置视图（延迟初始化）
        self.settings_view = None

        # 关于视图（延迟初始化）
        self.about_view = None

        # 模型管理视图（延迟初始化）
        self.model_manager_view = None

        # TTS 服务视图（延迟初始化）
        self.tts_service_view = None

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

                # 如果仍然没有模型，根据当前页面选择对应类型的模型
                if not model_id:
                    # 根据当前页面确定模型类型
                    if self._current_view_index == 0:  # Custom Voice
                        model_type = "customvoice"
                    elif self._current_view_index == 1:  # Voice Design
                        model_type = "voicedesign"
                    elif self._current_view_index == 2:  # Voice Clone
                        model_type = "base"
                    else:
                        model_type = None

                    if model_type:
                        usable_models = self.model_manager.list_usable_models_by_type(model_type)
                        self.terminal.add_log(f"DEBUG: 当前页面需要 {model_type} 类型模型")
                    else:
                        usable_models = self.model_manager.list_usable_models()

                    self.terminal.add_log(f"DEBUG: 可用模型列表 ({model_type or 'all'}) = {usable_models}")

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

                # 准备共享 tokenizer 路径
                # tokenizer 应该在 models_dir/tokenizer-12hz
                tokenizer_path = self.model_manager.models_dir / "tokenizer-12hz"
                shared_tokenizer_path = str(tokenizer_path) if tokenizer_path.exists() else None

                if shared_tokenizer_path:
                    self.terminal.add_log(f"使用共享 tokenizer: {shared_tokenizer_path}")
                else:
                    self.terminal.add_log("未找到共享 tokenizer，将使用模型内置 tokenizer")

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
                    attn_implementation=attn_implementation,
                    shared_tokenizer_path=shared_tokenizer_path,
                    enable_streaming=True,  # 显式启用流式输出
                    streaming_decode_window=80  # 流式解码窗口大小
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
        # 模型管理已抽离为独立组件，不再需要在这里创建
        pass

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
                    icon=ft.Icons.CLOUD,
                    selected_icon=ft.Icons.CLOUD_DONE,
                    label="TTS服务",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.SETTINGS,
                    selected_icon=ft.Icons.SETTINGS,
                    label="设置",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.INFO_OUTLINE,
                    selected_icon=ft.Icons.INFO,
                    label="关于",
                ),
            ],
            on_change=self.on_navigation_change,
        )

        self.navigation_rail = rail
        return rail

    def on_navigation_change(self, e):
        """导航切换事件"""
        old_index = self._current_view_index
        self._current_view_index = e.control.selected_index

        # 如果切换到不同的语音相关页面，清空当前模型ID以重新加载
        voice_pages = {0, 1, 2}  # Custom Voice, Voice Design, Voice Clone
        if old_index in voice_pages and self._current_view_index in voice_pages:
            # 清空模型ID，强制重新加载合适的模型
            self._current_model_id = None
            # 清除缓存的引擎
            if self._tts_engine is not None:
                self.terminal.add_log("切换页面，重新加载模型...")
                self._clear_tts_engine_cache(None)

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
            view = self._get_model_manager_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 4:
            # TTS 服务页面
            view = self._get_tts_service_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 5:
            # 设置页面
            view = self._get_settings_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 6:
            # 关于页面
            view = self._get_about_view()
            self.content_area.controls.append(view)

        self.content_area.update()

        # 在页面更新后刷新模型下拉框（需要在控件添加到页面后）
        if self._current_view_index == 0:
            self.custom_voice_view.refresh_model_dropdown()
        elif self._current_view_index == 1:
            self.voice_design_view.refresh_model_dropdown()
        elif self._current_view_index == 2:
            self.voice_clone_view.refresh_model_dropdown()

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

    def _get_about_view(self) -> ft.Control:
        """获取关于视图（延迟初始化）"""
        if self.about_view is None:
            self.about_view = AboutView(
                page=self.page,
                version=self.version
            )
        return self.about_view

    def _get_model_manager_view(self) -> ft.Control:
        """获取模型管理视图（延迟初始化）"""
        if self.model_manager_view is None:
            self.model_manager_view = ModelManagerView(
                page=self.page,
                model_manager=self.model_manager,
                terminal=self.terminal,
                on_models_changed=self._on_models_changed
            )
        return self.model_manager_view

    def _get_tts_service_view(self) -> ft.Control:
        """获取 TTS 服务视图（延迟初始化）"""
        if self.tts_service_view is None:
            self.tts_service_view = TTSServiceView(
                page=self.page,
                tts_engine_getter=lambda: self.tts_engine,
                terminal=self.terminal,
                config_manager=self.config_manager,
                on_service_state_change=self._on_service_state_change
            )
        return self.tts_service_view

    def _on_service_state_change(self, running: bool):
        """处理服务状态变化"""
        status = "运行中" if running else "已停止"
        self.terminal.add_log(f"TTS 服务状态: {status}")

    def _on_models_changed(self):
        """处理模型变更事件"""
        # 刷新各个视图的模型下拉框
        self._refresh_all_model_dropdowns()

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

                # 清理TTS线程池
                try:
                    from src.tts.thread_pool_manager import TTSThreadPoolManager
                    TTSThreadPoolManager().shutdown(wait=True)
                    logger.info("TTS线程池已关闭")
                except Exception as thread_pool_err:
                    logger.warning(f"关闭TTS线程池时出错: {thread_pool_err}")

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





