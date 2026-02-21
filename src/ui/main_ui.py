"""
PhantomVox 主 UI 控制器

基于 Flet 的文本转语音应用界面
采用 NavigationRail 架构
"""

import flet as ft
import logging
import asyncio
from typing import Optional, AsyncGenerator, Tuple
import numpy as np

from ui.components.app_bar import PhantomAppBar
from ui.components.custom_voice_view import CustomVoiceView
from ui.components.voice_design_view import VoiceDesignView
from ui.components.voice_clone_view import VoiceCloneView
from ui.components.srt_batch_view import SRTBatchView
from ui.components.voice_library import VoiceLibrary
from ui.components.settings_view import SettingsView
from ui.components.about_view import AboutView
from ui.components.model_manager_view import ModelManagerView
from ui.components.tts_service_view import TTSServiceView
from ui.components.system_monitor_view import SystemMonitorView
from core.terminal import AsyncTerminal
from core.model_manager import ModelManager
from core.task_engine import get_task_engine, TaskType
from core.engine_proxy_base import BaseEngineProxy
from config.config_manager import ConfigManager
from tts.qwen_engine import QwenEngine
from tts.audio_manager import AudioManager
from utils.async_helpers import create_task_with_error_handling

logger = logging.getLogger(__name__)


class SafeTTSEngineProxy(BaseEngineProxy):
    """
    TTS 引擎安全代理

    包装 QwenEngine，确保所有操作都通过任务引擎执行，防止并发冲突。
    继承BaseEngineProxy，使用UI terminal记录日志。
    """

    def __init__(self, engine: QwenEngine, task_engine, terminal):
        """
        初始化代理

        Args:
            engine: 实际的 TTS 引擎
            task_engine: 任务引擎实例
            terminal: 终端日志实例
        """
        # 保存引擎引用，用于同步方法
        self._engine = engine
        self._terminal = terminal

        # 调用父类初始化，传入引擎getter函数
        super().__init__(engine_getter=lambda: engine, task_engine=task_engine)

    def _log(self, message: str):
        """
        记录日志到UI terminal

        Args:
            message: 日志消息
        """
        self._terminal.add_log(f"[任务队列] {message}")


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
            models_dir=self.config_manager.get("model.model_path", "./models")
            or "./models",
            config_manager=self.config_manager,
        )

        # 初始化声音库管理器
        self.voice_library = VoiceLibrary(self.config_manager)

        # 初始化任务引擎
        self.task_engine = get_task_engine()

        # 启动任务引擎（带错误处理）
        def handle_task_engine_start_error(e):
            self.terminal.add_log(f"❌ 任务引擎启动失败: {str(e)}")
            self.terminal.add_log("TTS 功能将无法使用，请重启应用")

        create_task_with_error_handling(
            self.task_engine.start(),
            task_name="TaskEngineStartup",
            on_error=handle_task_engine_start_error,
        )

        # 懒加载组件
        self._tts_engine: Optional[QwenEngine] = None
        self._audio_manager: Optional[AudioManager] = None
        self._engine_loading_event = asyncio.Event()  # 加载完成事件

        # 当前视图索引
        self._current_view_index = 0

        # 当前选择的模型（用于初始化 TTS 引擎）
        self._current_model_id = None

        # 不要在初始化时设置默认模型，让 TTS 引擎根据当前页面自动选择

        # 三个新视图（延迟初始化）
        self.custom_voice_view = None
        self.voice_design_view = None
        self.voice_clone_view = None
        self.srt_batch_view = None

        # 设置视图（延迟初始化）
        self.settings_view = None

        # 关于视图（延迟初始化）
        self.about_view = None

        # 模型管理视图（延迟初始化）
        self.model_manager_view = None

        # TTS 服务视图（延迟初始化）
        self.tts_service_view = None

        # 系统监控组件
        self.system_monitor = SystemMonitorView(page=page, update_interval=1.0)

        # UI 样式配置
        self.BStyle = ft.ButtonStyle(
            icon_size=20,
            text_style=ft.TextStyle(size=14, font_family="Microsoft YaHei"),
        )

        # 当前生成的音频
        self._last_audio = None

        # 文件选择器
        self._file_picker = ft.FilePicker()

        # 初始化 AppBar
        self.app_bar = PhantomAppBar(
            page=page,
            version=version,
            on_theme_toggle=self._on_theme_toggle,
            on_close=self._on_close_window,
        )

        # 设置窗口事件拦截
        self.page.window.prevent_close = True
        self.page.window.on_event = self._window_event

        # 创建 UI 组件
        self._create_ui_components()

        logger.info("PhantomVox UI 初始化完成")

    def _create_qwen_engine_lazy(
        self,
        model_path: str,
        model_type: str,
        device: str,
        dtype,
        attn_implementation: str,
        shared_tokenizer_path: Optional[str],
    ) -> QwenEngine:
        """
        创建 Qwen 引擎（延迟加载模式）

        此方法只创建引擎对象，不加载模型，用于任务引擎中异步加载。

        Returns:
            QwenEngine: 未加载模型的引擎实例
        """
        return QwenEngine(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            shared_tokenizer_path=shared_tokenizer_path,
            enable_streaming=True,
            streaming_decode_window=80,
            lazy_load=True,  # 延迟加载模式
        )

    def _load_qwen_engine_sync(self, engine: QwenEngine):
        """
        同步加载 Qwen 引擎（在线程池中执行）

        Args:
            engine: 延迟加载的 QwenEngine 实例
        """
        engine.load_model(force_reload=True)

    async def _load_model_async(self, model_id: str) -> SafeTTSEngineProxy:
        """
        异步加载模型（通过任务引擎，不阻塞UI）

        Args:
            model_id: 要加载的模型ID

        Returns:
            SafeTTSEngineProxy: 加载完成的安全代理
        """
        self.terminal.add_log("正在加载模型...")

        try:
            # 获取模型路径
            model_path = self.model_manager.get_model_path(model_id)
            if not model_path:
                raise RuntimeError(f"模型路径无效: {model_id}")

            device = self.config_manager.get("model.device", "auto")
            dtype = self.config_manager.get("model.dtype", "bfloat16")
            attn_implementation = self.config_manager.get(
                "model.attn_implementation", "sdpa"
            )

            model_info = self.model_manager.get_model_info(model_id)
            self.terminal.add_log(
                f"使用模型: {model_info.name if model_info else model_id}"
            )
            self.terminal.add_log(f"模型ID: {model_id}")
            self.terminal.add_log(f"设备: {device}")

            # 准备共享 tokenizer 路径
            tokenizer_path = self.model_manager.models_dir / "tokenizer-12hz"
            shared_tokenizer_path = (
                str(tokenizer_path) if tokenizer_path.exists() else None
            )

            # 根据模型ID确定模型类型
            if "customvoice" in model_id:
                model_type = "CustomVoice"
            elif "voicedesign" in model_id:
                model_type = "VoiceDesign"
            elif "base" in model_id:
                model_type = "Base"
            else:
                model_type = None

            # 步骤1：创建延迟加载的引擎（快速，不阻塞）
            raw_engine = self._create_qwen_engine_lazy(
                model_path=str(model_path),
                model_type=model_type,
                device=device,
                dtype=dtype,
                attn_implementation=attn_implementation,
                shared_tokenizer_path=shared_tokenizer_path,
            )

            await self.task_engine.submit(
                task_type=TaskType.LOAD,
                func=self._load_qwen_engine_sync,
                args=(raw_engine,),
                description=f"加载模型: {model_info.name if model_info else model_id}",
                priority=10,
                model_id=model_id,
            )

            if raw_engine.model is None:
                raise RuntimeError(
                    f"模型加载失败: {model_info.name if model_info else model_id} - "
                    "模型对象为空，请检查日志了解详情"
                )

            proxy = SafeTTSEngineProxy(
                engine=raw_engine, task_engine=self.task_engine, terminal=self.terminal
            )

            self.terminal.add_log("✓ 模型加载完成")

            return proxy

        except Exception as e:
            self.terminal.add_log(f"✗ 模型加载失败: {str(e)}")
            raise

    def _get_model_id(self) -> str:
        """
        确定要使用的模型ID

        Returns:
            str: 模型ID
        """
        # 使用 _current_model_id 或尝试获取选中的模型
        model_id = self._current_model_id

        self.terminal.add_log(f"DEBUG: _current_model_id = {model_id}")

        # 如果没有指定模型，尝试从下拉框获取
        if not model_id:
            model_id = getattr(self, "model_dropdown", None)
            if model_id is not None:
                model_id = model_id.value
            self.terminal.add_log(
                f"DEBUG: 从 model_dropdown 获取 model_id = {model_id}"
            )

        # 如果仍然没有模型，根据当前页面选择对应类型的模型
        if not model_id:
            # 根据当前页面确定模型类型
            if self._current_view_index == 0:
                model_type = "customvoice"
            elif self._current_view_index == 1:
                model_type = "voicedesign"
            elif self._current_view_index == 2:
                model_type = "base"
            elif self._current_view_index == 3:
                model_type = "base"
            else:
                model_type = None

            if model_type:
                usable_models = self.model_manager.list_usable_models_by_type(
                    model_type
                )
                self.terminal.add_log(f"DEBUG: 当前页面需要 {model_type} 类型模型")
            else:
                usable_models = self.model_manager.list_usable_models()

            self.terminal.add_log(
                f"DEBUG: 可用模型列表 ({model_type or 'all'}) = {usable_models}"
            )

            if not usable_models:
                # 没有可用的模型
                self.terminal.add_log(
                    "✗ 没有可用的 TTS 模型，请先在「模型管理」中下载模型"
                )

                # 检查是否有已安装但不可用的模型
                installed = self.model_manager.get_installed_models()
                if installed:
                    self.terminal.add_log("提示: 已安装的模型缺少依赖，请重新下载")

                raise RuntimeError("没有可用的 TTS 模型")

            model_id = usable_models[0]
            self.terminal.add_log(f"DEBUG: 使用第一个可用模型 = {model_id}")

        return model_id

    async def get_tts_engine(self) -> SafeTTSEngineProxy:
        """
        获取 TTS 引擎（异步，使用事件等待防止重复加载）

        Returns:
            SafeTTSEngineProxy: TTS 引擎代理
        """
        # 如果引擎未加载且事件未设置（表示没有正在进行的加载）
        if self._tts_engine is None and not self._engine_loading_event.is_set():
            try:
                self.terminal.add_log("正在初始化 TTS 引擎...")
                model_id = self._get_model_id()
                self._tts_engine = await self._load_model_async(model_id)
                # 标记加载完成
                self._engine_loading_event.set()
            except Exception as e:
                self.terminal.add_log(f"✗ TTS 引擎初始化失败: {str(e)}")
                # 即使失败也要设置事件，避免无限等待
                self._engine_loading_event.set()
                raise

        # 等待加载完成（如果其他协程正在加载）
        await self._engine_loading_event.wait()

        return self._tts_engine

    async def _get_tts_engine_for_view(self) -> SafeTTSEngineProxy:
        """
        为视图提供的异步 TTS 引擎获取方法

        Returns:
            SafeTTSEngineProxy: TTS 引擎代理
        """
        return await self.get_tts_engine()

    def _get_sync_tts_engine_for_service(self):
        """
        为 TTS 服务提供同步的引擎获取器

        注意：此方法返回已加载的引擎或 None，不会触发加载。
        这是为了解决 API 路由中同步访问引擎属性的问题。

        Returns:
            SafeTTSEngineProxy | None: TTS 引擎代理或 None
        """
        return self._tts_engine

    @property
    def tts_engine(self):  # 同步访问器
        """
        懒加载：TTS 引擎（返回安全代理）

        注意：如果引擎未加载，返回 None 而不是等待加载
        推荐使用 get_tts_engine() 异步方法
        """
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
            expand=True,
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
                    icon=ft.Icons.SUBTITLES,
                    selected_icon=ft.Icons.SUBTITLES,
                    label="SRT批量",
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

        # PDCA 循环 #1 修复: 集中管理 FloatingActionButton
        # 根据当前视图索引设置对应的 FAB，解决视图切换时 FAB 不匹配的问题
        voice_pages = {0, 1, 2, 3}  # Custom Voice, Voice Design, Voice Clone, SRT Batch
        if self._current_view_index not in voice_pages:
            # 非语音页面，隐藏 FAB
            self.page.floating_action_button = None
        else:
            # 语音页面，设置对应视图的 FAB
            if self._current_view_index == 0:
                fab = self.custom_voice_view._fab if self.custom_voice_view else None
            elif self._current_view_index == 1:
                fab = self.voice_design_view._fab if self.voice_design_view else None
            elif self._current_view_index == 2:
                fab = self.voice_clone_view._fab if self.voice_clone_view else None
            elif self._current_view_index == 3:
                fab = self.srt_batch_view._fab if self.srt_batch_view else None
            else:
                fab = None
            self.page.floating_action_button = fab

        # 如果切换到不同的语音相关页面，清空当前模型ID以重新加载
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
            # SRT批量推理页面
            view = self._get_srt_batch_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 4:
            # 模型管理页面
            view = self._get_model_manager_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 5:
            # TTS 服务页面
            view = self._get_tts_service_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 6:
            # 设置页面
            view = self._get_settings_view()
            self.content_area.controls.append(view)
        elif self._current_view_index == 7:
            # 关于页面
            view = self._get_about_view()
            self.content_area.controls.append(view)

        self.content_area.update()

        # PDCA 循环 #1 修复: 更新 FAB（视图可能刚刚初始化）
        # 确保在视图创建后设置正确的 FAB
        voice_pages = {0, 1, 2, 3}
        if self._current_view_index in voice_pages:
            if self._current_view_index == 0:
                fab = self.custom_voice_view._fab if self.custom_voice_view else None
            elif self._current_view_index == 1:
                fab = self.voice_design_view._fab if self.voice_design_view else None
            elif self._current_view_index == 2:
                fab = self.voice_clone_view._fab if self.voice_clone_view else None
            elif self._current_view_index == 3:
                fab = self.srt_batch_view._fab if self.srt_batch_view else None
            else:
                fab = None
            self.page.floating_action_button = fab

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
                tts_engine_getter=self._get_tts_engine_for_view,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(
                    model_id
                ),
            )
        return self.custom_voice_view

    def _get_voice_design_view(self) -> ft.Control:
        """获取声音设计视图（延迟初始化）"""
        if self.voice_design_view is None:
            self.voice_design_view = VoiceDesignView(
                page=self.page,
                tts_engine_getter=self._get_tts_engine_for_view,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(
                    model_id
                ),
            )
        return self.voice_design_view

    def _get_voice_clone_view(self) -> ft.Control:
        """获取声音克隆视图（延迟初始化）"""
        if self.voice_clone_view is None:
            self.voice_clone_view = VoiceCloneView(
                page=self.page,
                tts_engine_getter=self._get_tts_engine_for_view,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(
                    model_id
                ),
            )
        return self.voice_clone_view

    def _get_srt_batch_view(self) -> ft.Control:
        if self.srt_batch_view is None:
            self.srt_batch_view = SRTBatchView(
                page=self.page,
                tts_engine_getter=self._get_tts_engine_for_view,
                audio_manager_getter=lambda: self.audio_manager,
                terminal=self.terminal,
                voice_library=self.voice_library,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_clear_engine_cache=lambda model_id: self._clear_tts_engine_cache(
                    model_id
                ),
                on_load_model=self._load_model_for_srt,
            )
        return self.srt_batch_view

    async def _load_model_for_srt(self, mode: str) -> bool:
        mode_to_type = {
            "custom_voice": "customvoice",
            "voice_design": "voicedesign",
            "voice_clone": "base",
        }
        model_type = mode_to_type.get(mode, "base")
        model_id = self._select_model_by_type(model_type)
        if not model_id:
            self.terminal.add_log(f"没有可用的 {model_type} 类型模型")
            return False
        try:
            self._current_model_id = model_id
            self._tts_engine = await self._load_model_async(model_id)
            self._engine_loading_event.set()
            return True
        except Exception as e:
            self.terminal.add_log(f"模型加载失败: {str(e)}")
            return False

    def _get_settings_view(self) -> ft.Control:
        """获取设置视图（延迟初始化）"""
        if self.settings_view is None:
            self.settings_view = SettingsView(
                page=self.page,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                on_settings_changed=self._on_settings_changed,
            )
        return self.settings_view

    def _get_about_view(self) -> ft.Control:
        """获取关于视图（延迟初始化）"""
        if self.about_view is None:
            self.about_view = AboutView(page=self.page, version=self.version)
        return self.about_view

    def _get_model_manager_view(self) -> ft.Control:
        """获取模型管理视图（延迟初始化）"""
        if self.model_manager_view is None:
            self.model_manager_view = ModelManagerView(
                page=self.page,
                model_manager=self.model_manager,
                terminal=self.terminal,
                on_models_changed=self._on_models_changed,
            )
        return self.model_manager_view

    def _get_tts_service_view(self) -> ft.Control:
        if self.tts_service_view is None:
            self.tts_service_view = TTSServiceView(
                page=self.page,
                tts_engine_getter=self._get_sync_tts_engine_for_service,
                terminal=self.terminal,
                config_manager=self.config_manager,
                model_manager=self.model_manager,
                voice_library=self.voice_library,
                on_service_state_change=self._on_service_state_change,
                on_load_model=self._load_model_for_service,
            )
        return self.tts_service_view

    async def _load_model_for_service(self, model_type: str) -> bool:
        model_id = self._select_model_by_type(model_type)
        if not model_id:
            self.terminal.add_log(f"✗ 没有可用的 {model_type} 类型模型")
            return False
        try:
            self._current_model_id = model_id
            self._tts_engine = await self._load_model_async(model_id)
            self._engine_loading_event.set()
            return True
        except Exception as e:
            self.terminal.add_log(f"✗ 模型加载失败: {str(e)}")
            return False

    def _select_model_by_type(self, model_type: str) -> str | None:
        usable = self.model_manager.list_usable_models_by_type(model_type)
        if usable:
            return usable[0]
        return None

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
            # 先卸载旧模型以释放资源
            self.terminal.add_log("设置已更改，正在卸载旧模型...")

            # 通过任务引擎提交卸载任务
            async def unload_and_clear():
                await self.task_engine.submit(
                    task_type=TaskType.UNLOAD,
                    func=self._do_unload_engine,
                    description="设置更改后卸载 TTS 引擎",
                    priority=10,
                )
                self.terminal.add_log("TTS 引擎将重新初始化")

            # 在后台执行卸载（带错误处理）
            def handle_unload_error(e):
                self.terminal.add_log(f"❌ 卸载引擎失败: {str(e)}")
                self.terminal.add_log("将尝试继续使用当前模型")

            create_task_with_error_handling(
                unload_and_clear(),
                task_name="SettingsChangeUnload",
                on_error=handle_unload_error,
            )

    def _clear_tts_engine_cache(self, model_id: str = None):
        """清除 TTS 引擎缓存以强制重新初始化

        Args:
            model_id: 可选，指定要使用的模型 ID
        """
        self.terminal.add_log(
            f"DEBUG: _clear_tts_engine_cache 被调用，model_id = {model_id}"
        )
        self.terminal.add_log(
            f"DEBUG: 更新前 _current_model_id = {self._current_model_id}"
        )

        if model_id:
            self._current_model_id = model_id
            self.terminal.add_log(
                f"DEBUG: 已设置 _current_model_id = {self._current_model_id}"
            )

            # 同步所有视图的下拉框
            self._sync_model_dropdowns(model_id)

        if self._tts_engine is not None:
            # 通过任务引擎提交卸载任务，确保不会在推理时卸载
            async def unload_task():
                self.terminal.add_log("正在卸载旧模型（任务队列）...")

                # 等待当前推理完成（通过任务引擎保证）
                await self.task_engine.submit(
                    task_type=TaskType.UNLOAD,
                    func=self._do_unload_engine,
                    description="卸载 TTS 引擎",
                    priority=10,  # 卸载任务优先级较高，但仍需等待当前任务完成
                )

            # 在后台执行卸载（带错误处理）
            def handle_unload_error(e):
                self.terminal.add_log(f"❌ 卸载引擎失败: {str(e)}")
                self.terminal.add_log("可能需要重启应用以清理资源")

            create_task_with_error_handling(
                unload_task(),
                task_name="ModelSwitchUnload",
                on_error=handle_unload_error,
            )
        else:
            self.terminal.add_log("TTS 引擎未初始化，无需清除缓存")

    def _do_unload_engine(self):
        if self._tts_engine is not None:
            self.terminal.add_log("正在执行模型卸载...")
            self._tts_engine.unload()
            self._tts_engine = None
            self._engine_loading_event.clear()
            self.terminal.add_log("✓ TTS 引擎已卸载")

    def _sync_model_dropdowns(self, model_id: str):
        """同步所有视图的模型下拉框选择

        Args:
            model_id: 要同步到的模型 ID
        """
        if self.custom_voice_view and hasattr(self.custom_voice_view, "model_dropdown"):
            self.custom_voice_view.model_dropdown.value = model_id
            try:
                self.custom_voice_view.model_dropdown.update()
            except RuntimeError:
                pass

        if self.voice_design_view and hasattr(self.voice_design_view, "model_dropdown"):
            self.voice_design_view.model_dropdown.value = model_id
            try:
                self.voice_design_view.model_dropdown.update()
            except RuntimeError:
                pass

        if self.voice_clone_view and hasattr(self.voice_clone_view, "model_dropdown"):
            self.voice_clone_view.model_dropdown.value = model_id
            try:
                self.voice_clone_view.model_dropdown.update()
            except RuntimeError:
                pass

    def _refresh_all_model_dropdowns(self):
        """刷新所有视图的模型下拉框选项"""
        # 刷新 CustomVoiceView
        if self.custom_voice_view and hasattr(
            self.custom_voice_view, "refresh_model_dropdown"
        ):
            try:
                self.custom_voice_view.refresh_model_dropdown()
            except Exception as e:
                logger.error(f"刷新 CustomVoiceView 模型下拉框失败: {str(e)}")

        # 刷新 VoiceDesignView
        if self.voice_design_view and hasattr(
            self.voice_design_view, "refresh_model_dropdown"
        ):
            try:
                self.voice_design_view.refresh_model_dropdown()
            except Exception as e:
                logger.error(f"刷新 VoiceDesignView 模型下拉框失败: {str(e)}")

        # 刷新 VoiceCloneView
        if self.voice_clone_view and hasattr(
            self.voice_clone_view, "refresh_model_dropdown"
        ):
            try:
                self.voice_clone_view.refresh_model_dropdown()
            except Exception as e:
                logger.error(f"刷新 VoiceCloneView 模型下拉框失败: {str(e)}")

    def build_main_view(self) -> ft.Control:
        """构建主界面"""
        # 创建导航栏
        rail = self.build_navigation_rail()

        # 创建内容区域 - 默认显示第一个导航页面（自定义语音）
        initial_view = self._get_custom_voice_view()
        self.content_area = ft.Column([initial_view], expand=True)

        # 设置初始 FAB（修复程序启动时 FAB 不显示的 bug）
        if self.custom_voice_view and hasattr(self.custom_voice_view, "_fab"):
            self.page.floating_action_button = self.custom_voice_view._fab

        # 创建终端日志组件（用于 ExpansionTile 的 controls）
        self._terminal_logs_content = ft.Container(
            content=self.terminal.logs,
            border=ft.Border.all(1, ft.Colors.GREY_400),
            border_radius=8,
            padding=5,
        )

        # 创建右侧主容器（包含内容和终端）
        right_panel = ft.Column(
            [
                # 内容区域
                ft.Container(content=self.content_area, expand=True, padding=20),
                # 分隔线
                ft.Divider(height=1, color=ft.Colors.GREY_300),
                # 全局固定终端 - 使用 ExpansionTile
                ft.Container(
                    content=ft.ExpansionTile(
                        title=ft.Text("运行日志", size=14, weight=ft.FontWeight.BOLD),
                        subtitle=ft.Text(
                            "点击展开/折叠日志", size=12, color=ft.Colors.GREY
                        ),
                        expanded=True,
                        leading=ft.IconButton(
                            icon=ft.Icons.CLEAR,
                            icon_size=18,
                            tooltip="清空日志",
                            on_click=self._on_clear_terminal,
                        ),
                        controls_padding=ft.padding.symmetric(horizontal=0, vertical=5),
                        controls=[
                            ft.Container(
                                content=self._terminal_logs_content,
                            )
                        ],
                        bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
                        collapsed_bgcolor=ft.Colors.with_opacity(
                            0.02, ft.Colors.ON_SURFACE
                        ),
                    ),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                ),
            ],
            expand=True,
            spacing=0,
        )

        # 创建左侧面板（包含导航栏和系统监控）
        left_panel = ft.Column(
            [
                # 导航栏
                rail,
                # 系统监控组件（在底部）
                self.system_monitor.build(),
            ],
            # expand=True,
            spacing=0,
        )

        # 主布局
        main_view = ft.Row(
            [
                # 左侧面板（导航栏 + 系统监控）
                left_panel,
                ft.VerticalDivider(width=1),
                # 右侧主容器
                right_panel,
            ],
            expand=True,
        )

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

    def _window_event(self, e):
        """
        处理窗口事件（拦截关闭事件）
        """
        if e.data == "close":
            # 拦截关闭事件，显示确认对话框
            self._on_close_window(e)

    async def _do_cleanup_and_close(self):
        """执行清理操作并关闭窗口"""
        try:
            # 1. 先隐藏窗口（立即响应用户）
            try:
                self.page.window.visible = False
                self.page.update()
            except RuntimeError:
                pass

            # 2. 卸载 TTS 引擎（释放显存）
            try:
                if self._tts_engine:
                    self._tts_engine.unload()
                    self._tts_engine = None
                    logger.info("TTS 引擎已卸载")
            except Exception as engine_err:
                logger.warning(f"卸载 TTS 引擎时出错: {engine_err}")

            # 3. 关闭系统监控
            try:
                if hasattr(self, "system_monitor") and self.system_monitor:
                    await self.system_monitor.stop_monitoring()
                    logger.info("系统监控已停止")
            except Exception as monitor_err:
                logger.warning(f"停止系统监控时出错: {monitor_err}")

            # 4. 关闭 TTS 线程池
            try:
                from src.tts.thread_pool_manager import TTSThreadPoolManager

                TTSThreadPoolManager().shutdown(wait=True)
                logger.info("TTS线程池已关闭")
            except Exception as thread_pool_err:
                logger.warning(f"关闭TTS线程池时出错: {thread_pool_err}")

            # 5. 保存配置
            try:
                self.config_manager.save_config()
                logger.info("配置已保存")
            except Exception as config_err:
                logger.warning(f"保存配置时出错: {config_err}")

            # 6. 允许关闭并关闭窗口
            try:
                self.page.window.prevent_close = False
                await self.page.window.close()
            except RuntimeError:
                pass
        except Exception:
            logger.exception("关闭应用时出错")
            try:
                await self.page.window.destroy()
            except:
                pass

    def _on_close_window(self, e):
        """
        处理窗口关闭事件，显示确认对话框
        """

        async def confirm_close_async():
            """确认关闭后的异步操作"""
            try:
                # 关闭对话框
                try:
                    self.page.pop_dialog()
                except RuntimeError:
                    pass
            except:
                pass
            # 执行清理和关闭
            await self._do_cleanup_and_close()

        def on_confirm(_):
            """用户确认关闭"""
            self.page.run_task(confirm_close_async)

        def on_cancel(_):
            """用户取消关闭"""
            try:
                self.page.pop_dialog()
            except:
                pass

        # 显示确认对话框
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("确认退出"),
            content=ft.Text("确定要退出 PhantomVox 吗？"),
            actions=[
                ft.TextButton("取消", on_click=on_cancel),
                ft.TextButton("退出", on_click=on_confirm),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.show_dialog(dialog)

    def _on_clear_terminal(self, e):
        # 清空终端日志
        self.terminal.clear_terminal()
