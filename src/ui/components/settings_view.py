"""
设置页面组件 (Settings View)

实现配置设置页面，包含5个配置类别：
- 界面设置
- 模型设置
- 音频设置
- 网络设置
- 日志设置
"""

import flet as ft
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class SettingsView(ft.Container):
    """设置页面 - 使用 Tabs 组织配置类别"""

    def __init__(
        self,
        page: ft.Page,
        config_manager,
        model_manager,
        on_settings_changed: Optional[Callable] = None
    ):
        self._page = page
        self._config_manager = config_manager
        self._model_manager = model_manager
        self._on_settings_changed = on_settings_changed

        # 跟踪未保存的更改
        self._unsaved_changes = {}

        # 文件选择器
        self._folder_picker = ft.FilePicker()

        # UI 样式配置
        self.BStyle = ft.ButtonStyle(
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

        # 加载配置到UI
        self._load_config_to_ui()

    def _build_ui(self) -> ft.Control:
        """构建主UI界面"""
        # 创建Tabs
        self.tabs = ft.Tabs(
            selected_index=0,
            length=4,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label="界面", icon=ft.Icons.PALETTE),
                            ft.Tab(label="模型", icon=ft.Icons.MEMORY),
                            ft.Tab(label="音频", icon=ft.Icons.VOLUME_UP),
                            ft.Tab(label="日志", icon=ft.Icons.ARTICLE),
                        ]
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            self._build_interface_tab(),
                            self._build_model_tab(),
                            self._build_audio_tab(),
                            self._build_logging_tab(),
                        ],
                    ),
                ],
            ),
        )

        # 保存和重置按钮引用
        self.save_button = ft.Button(
            "保存设置",
            icon=ft.Icons.SAVE,
            style=self.BStyle,
            on_click=self._on_save_click,
            disabled=True
        )

        self.reset_button = ft.Button(
            "重置默认",
            icon=ft.Icons.RESTORE,
            style=self.BStyle,
            on_click=self._on_reset_click
        )

        return ft.Column([
            # 标题栏
            ft.Row([
                ft.Text("设置", size=24, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                self.reset_button,
                ft.Container(width=10),
                self.save_button,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

            # Tabs内容
            ft.Container(
                content=self.tabs,
                expand=True,
                bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
                border_radius=12,
                padding=20
            )
        ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    # ========== Tab 构建方法 ==========

    def _get_available_device_options(self):
        """动态获取可用的设备选项"""
        try:
            import torch

            # 第一个选项始终是 "auto"（自动检测）
            options = [
                ft.dropdown.Option("auto", "自动检测（推荐）"),
                ft.dropdown.Option("cpu", "CPU"),
            ]

            # 检查 CUDA 是否可用
            if torch.cuda.is_available():
                cuda_device_count = torch.cuda.device_count()
                logger.info(f"检测到 {cuda_device_count} 个 CUDA 设备")

                # 添加通用 CUDA 选项
                options.append(ft.dropdown.Option("cuda", "CUDA (通用)"))

                # 添加每个具体的 CUDA 设备
                for i in range(cuda_device_count):
                    device_name = torch.cuda.get_device_name(i)
                    # 限制设备名称长度，避免太长
                    if len(device_name) > 30:
                        device_name = device_name[:27] + "..."
                    options.append(
                        ft.dropdown.Option(f"cuda:{i}", f"CUDA:{i} ({device_name})")
                    )
            else:
                logger.info("未检测到 CUDA 设备")
                # 即使没有 CUDA，也保留通用选项供用户选择
                options.append(ft.dropdown.Option("cuda", "CUDA (如果可用)"))

            return options

        except Exception as e:
            logger.warning(f"检测设备失败: {str(e)}", exc_info=True)
            # 向用户显示警告
            self._show_error(f"⚠️ 设备检测失败: {str(e)}")
            # 如果检测失败，返回基本选项（不包含 CUDA，因为检测失败了）
            return [
                ft.dropdown.Option("auto", "自动检测（推荐）"),
                ft.dropdown.Option("cpu", "CPU"),
            ]

    def _validate_device_value(self, device_value):
        """验证设备值是否有效，如果不有效则返回默认值"""
        # 获取当前可用的设备选项
        available_devices = [opt.key for opt in self.device_dropdown.options]

        # 如果配置的设备不在可用列表中，使用默认值
        if device_value not in available_devices:
            # 优先使用 auto，如果不可用则使用 cpu
            if "auto" in available_devices:
                device_value = "auto"
                logger.info(f"配置的设备不可用，使用自动检测: {device_value}")
            else:
                device_value = "cpu"
                logger.info("使用 CPU")

        return device_value

    def _build_interface_tab(self) -> ft.Control:
        """构建界面设置 Tab"""
        # 主题模式开关
        self.theme_switch = ft.Switch(
            label="深色模式",
            value=False
        )
        self.theme_switch.on_change = self._on_theme_change

        return ft.Column([
            ft.Text("界面设置", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),

            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("主题模式", size=14, expand=1),
                        self.theme_switch
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Text(
                        "切换深色/浅色主题模式（切换后立即生效）",
                        size=12,
                        color=ft.Colors.GREY_400
                    ),
                ], spacing=5),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                border_radius=8
            )
        ], scroll=ft.ScrollMode.AUTO, spacing=15)

    def _build_model_tab(self) -> ft.Control:
        """构建模型设置 Tab"""
        # 动态检测可用设备
        device_options = self._get_available_device_options()

        # 设备选择
        self.device_dropdown = ft.Dropdown(
            label="运行设备",
            options=device_options,
            width=300,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            on_select=self._on_device_changed
        )

        # 数据类型
        self.dtype_dropdown = ft.Dropdown(
            label="数据类型",
            options=[
                ft.dropdown.Option("bfloat16", "BFloat16"),
                ft.dropdown.Option("float16", "Float16"),
                ft.dropdown.Option("float32", "Float32"),
            ],
            width=300,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            on_select=self._on_dtype_changed
        )

        # 注意力实现
        self.attn_dropdown = ft.Dropdown(
            label="注意力实现",
            options=[
                ft.dropdown.Option("sdpa", "SDPA"),
                ft.dropdown.Option("flash_attention_2", "Flash Attention 2"),
            ],
            width=300,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            on_select=self._on_attn_changed
        )

        # 采样率（固定值，由模型决定）
        self.sample_rate_dropdown = ft.Dropdown(
            label="采样率（模型固定）",
            options=[
                ft.dropdown.Option(24000, "24000 Hz"),
            ],
            width=300,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            value=24000,
            disabled=True,
            hint_text="采样率由模型决定，不可更改"
        )

        # 自动下载模型
        self.auto_download_switch = ft.Switch(
            label="自动下载缺失的模型",
            value=False
        )
        self.auto_download_switch.on_change = lambda e: self._on_setting_changed("model.auto_download", e.control.value)

        # 智能显存管理
        self.smart_vram_switch = ft.Switch(
            label="智能显存管理",
            value=False
        )
        self.smart_vram_switch.on_change = lambda e: self._on_setting_changed("model.smart_vram", e.control.value)

        return ft.Column([
            ft.Text("模型设置", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),

            ft.Column([
                ft.Text("运行设备", size=13),
                self.device_dropdown,
                ft.Text("选择模型运行设备。推荐使用「自动检测」让系统选择最佳设备", size=11, color=ft.Colors.GREY_400),
            ], spacing=5),

            ft.Container(height=15),

            ft.Column([
                ft.Text("数据类型", size=13),
                self.dtype_dropdown,
                ft.Text("降低精度可节省显存，可能影响生成质量", size=11, color=ft.Colors.GREY_400),
            ], spacing=5),

            ft.Container(height=15),

            ft.Column([
                ft.Text("注意力实现", size=13),
                self.attn_dropdown,
                ft.Text("选择注意力计算优化方式", size=11, color=ft.Colors.GREY_400),
            ], spacing=5),

            ft.Container(height=15),

            ft.Column([
                ft.Text("采样率", size=13),
                self.sample_rate_dropdown,
                ft.Text("qwen-tts 模型固定输出 24000Hz 采样率", size=11, color=ft.Colors.AMBER),
            ], spacing=5),

            ft.Container(height=15),

            ft.Container(
                content=ft.Column([
                    self.auto_download_switch,
                    ft.Text("启用后，使用缺失模型时会自动下载", size=11, color=ft.Colors.GREY_400),
                ], spacing=5),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                border_radius=8
            ),

            ft.Container(height=15),

            ft.Container(
                content=ft.Column([
                    self.smart_vram_switch,
                    ft.Text("打开：卸载模型时先移到CPU，1分钟后未重新加载自动清除 | 关闭：卸载时直接清除，不移到CPU", size=11, color=ft.Colors.GREY_400),
                ], spacing=5),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                border_radius=8
            )
        ], scroll=ft.ScrollMode.AUTO, spacing=15)

    def _build_audio_tab(self) -> ft.Control:
        """构建音频设置 Tab"""
        # 输出格式
        self.format_dropdown = ft.Dropdown(
            label="输出格式",
            options=[
                ft.dropdown.Option("wav", "WAV (无损)"),
                ft.dropdown.Option("mp3", "MP3 (压缩)"),
                ft.dropdown.Option("ogg", "OGG (压缩)"),
            ],
            width=300,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            on_select=self._on_format_changed
        )

        # 自动保存
        self.auto_save_switch = ft.Switch(
            label="自动保存音频文件",
            value=False
        )
        self.auto_save_switch.on_change = lambda e: self._on_setting_changed("audio.auto_save", e.control.value)

        # 保存目录
        self.save_directory_field = ft.TextField(
            label="保存目录",
            width=400,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )
        self.save_directory_field.on_change = lambda e: self._on_setting_changed("audio.save_directory", e.control.value)

        browse_button = ft.Button(
            "浏览...",
            style=self.BStyle,
            on_click=self._on_browse_save_directory
        )

        return ft.Column([
            ft.Text("音频设置", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),

            ft.Column([
                ft.Text("输出格式", size=13),
                self.format_dropdown,
                ft.Text("选择音频保存格式", size=11, color=ft.Colors.GREY_400),
            ], spacing=5),

            ft.Container(height=15),

            ft.Container(
                content=ft.Column([
                    self.auto_save_switch,
                    ft.Text("启用后，生成的音频会自动保存到指定目录", size=11, color=ft.Colors.GREY_400),
                ], spacing=5),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                border_radius=8
            ),

            ft.Container(height=15),

            ft.Column([
                ft.Text("保存目录", size=13),
                ft.Row([self.save_directory_field, browse_button], spacing=10),
            ], spacing=5),
        ], scroll=ft.ScrollMode.AUTO, spacing=15)

    def _build_logging_tab(self) -> ft.Control:
        """构建日志设置 Tab"""
        # 启用日志
        self.logging_enabled_switch = ft.Switch(
            label="启用日志",
            value=False
        )
        self.logging_enabled_switch.on_change = lambda e: self._on_setting_changed("logging.enabled", e.control.value)

        # 日志级别
        self.log_level_dropdown = ft.Dropdown(
            label="日志级别",
            options=[
                ft.dropdown.Option("DEBUG", "DEBUG (详细)"),
                ft.dropdown.Option("INFO", "INFO (默认)"),
                ft.dropdown.Option("WARNING", "WARNING (警告)"),
                ft.dropdown.Option("ERROR", "ERROR (错误)"),
            ],
            width=300,
            text_style=ft.TextStyle(font_family="Microsoft YaHei"),
            on_select=self._on_log_level_changed
        )

        # 保存日志
        self.save_logs_switch = ft.Switch(
            label="保存日志到文件",
            value=False
        )
        self.save_logs_switch.on_change = self._on_save_logs_switch_change

        # 日志目录
        self.log_directory_field = ft.TextField(
            label="日志目录",
            width=400,
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )
        self.log_directory_field.on_change = lambda e: self._on_setting_changed("logging.log_directory", e.control.value)

        log_browse_button = ft.Button(
            "浏览...",
            style=self.BStyle,
            on_click=self._on_browse_log_directory
        )

        # 日志目录容器（条件显示）
        self.log_directory_container = ft.Container(
            content=ft.Column([
                ft.Text("日志目录", size=13),
                ft.Row([self.log_directory_field, log_browse_button], spacing=10),
            ], spacing=5),
            visible=False
        )

        return ft.Column([
            ft.Text("日志设置", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),

            ft.Container(
                content=ft.Column([
                    self.logging_enabled_switch,
                    ft.Text("禁用后将不记录日志", size=11, color=ft.Colors.GREY_400),
                ], spacing=5),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                border_radius=8
            ),

            ft.Container(height=15),

            ft.Column([
                ft.Text("日志级别", size=13),
                self.log_level_dropdown,
                ft.Text("选择记录的日志详细程度", size=11, color=ft.Colors.GREY_400),
            ], spacing=5),

            ft.Container(height=15),

            ft.Container(
                content=ft.Column([
                    self.save_logs_switch,
                ], spacing=5),
                padding=15,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                border_radius=8
            ),

            ft.Container(height=15),

            self.log_directory_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=15)

    # ========== 配置加载 ==========

    def _load_config_to_ui(self):
        """从配置管理器加载配置到UI控件"""
        try:
            # 界面设置
            theme_mode = self._config_manager.get("theme_mode", "dark")
            self.theme_switch.value = (theme_mode == "dark")

            # 模型设置
            device_value = self._config_manager.get("model.device", "auto")
            dtype_value = self._config_manager.get("model.dtype", "bfloat16")
            attn_value = self._config_manager.get("model.attn_implementation", "sdpa")

            logger.info(f"加载模型设置: device={device_value}, dtype={dtype_value}, attn={attn_value}")

            # 验证设备值是否有效，处理 "auto" 等旧选项
            device_value = self._validate_device_value(device_value)

            self.device_dropdown.value = device_value
            self.dtype_dropdown.value = dtype_value
            self.attn_dropdown.value = attn_value
            # 采样率固定为 24000Hz，由模型决定，不从配置读取
            # self.sample_rate_dropdown.value = self._config_manager.get("model.sample_rate", 24000)
            self.auto_download_switch.value = self._config_manager.get("model.auto_download", True)
            self.smart_vram_switch.value = self._config_manager.get("model.smart_vram", True)

            # 音频设置
            self.format_dropdown.value = self._config_manager.get("audio.output_format", "wav")
            self.auto_save_switch.value = self._config_manager.get("audio.auto_save", False)
            self.save_directory_field.value = self._config_manager.get("audio.save_directory", "./output")

            # 日志设置
            self.logging_enabled_switch.value = self._config_manager.get("logging.enabled", True)
            self.log_level_dropdown.value = self._config_manager.get("logging.level", "INFO")
            save_logs = self._config_manager.get("logging.save_logs", True)
            self.save_logs_switch.value = save_logs
            self.log_directory_field.value = self._config_manager.get("logging.log_directory", "./logs")
            self.log_directory_container.visible = save_logs

            # 更新UI - 仅当已添加到页面时才更新
            try:
                self.update()
            except RuntimeError:
                # 控件还未添加到页面，忽略错误
                pass

        except Exception as e:
            logger.exception("加载配置到UI失败")
            self._show_error(f"加载配置失败: {str(e)}")

    # ========== 事件处理 ==========

    def _on_device_changed(self, e):
        """设备选择变更事件"""
        self._on_setting_changed("model.device", e.control.value)

    def _on_dtype_changed(self, e):
        """数据类型变更事件"""
        self._on_setting_changed("model.dtype", e.control.value)

    def _on_attn_changed(self, e):
        """注意力实现变更事件"""
        self._on_setting_changed("model.attn_implementation", e.control.value)

    def _on_format_changed(self, e):
        """输出格式变更事件"""
        self._on_setting_changed("audio.output_format", e.control.value)

    def _on_log_level_changed(self, e):
        """日志级别变更事件"""
        self._on_setting_changed("logging.level", e.control.value)

    def _on_theme_change(self, e):
        """主题切换事件 - 实时预览"""
        new_mode = "dark" if e.control.value else "light"
        self._unsaved_changes["theme_mode"] = new_mode

        # 实时应用主题预览
        self._page.theme_mode = ft.ThemeMode.DARK if new_mode == "dark" else ft.ThemeMode.LIGHT
        self._page.update()

        self._update_save_button_state()

    def _on_save_logs_switch_change(self, e):
        """保存日志开关切换事件"""
        save_logs = e.control.value
        self.log_directory_container.visible = save_logs
        self.log_directory_container.update()

        self._on_setting_changed("logging.save_logs", save_logs)

    def _on_slider_change(self, key: str, e, value_text: ft.Text, suffix: str = "", scale: int = 1):
        """滑块值变化事件"""
        value = e.control.value
        value_text.value = f"{value * scale:.0f}{suffix}" if scale > 1 else f"{value:.1f}{suffix}"
        value_text.update()
        self._on_setting_changed(key, value)

    def _on_setting_changed(self, key: str, value, e=None):
        """设置值变化事件"""
        # 验证输入
        if not self._validate_setting(key, value):
            return

        # 跟踪更改
        self._unsaved_changes[key] = value
        logger.info(f"设置已变更: {key} = {value}, 未保存更改数: {len(self._unsaved_changes)}")
        self._update_save_button_state()

    def _on_save_click(self, e):
        """保存设置按钮点击事件"""
        if not self._unsaved_changes:
            self._show_message("没有更改需要保存")
            return

        try:
            # 检查是否有模型相关的设置变更
            model_settings_changed = any(key.startswith("model.") for key in self._unsaved_changes.keys())

            # 应用所有更改
            for key, value in self._unsaved_changes.items():
                self._config_manager.set(key, value)

            # 保存到文件
            self._config_manager.save_config()

            # 清除未保存的更改
            self._unsaved_changes.clear()
            self._update_save_button_state()

            # 显示成功消息
            if model_settings_changed:
                self._show_message("✓ 设置已保存。模型相关设置将在下次生成语音时生效", ft.Colors.GREEN)
            else:
                self._show_message("✓ 设置已保存", ft.Colors.GREEN)

            # 通知回调
            if self._on_settings_changed:
                self._on_settings_changed()

        except Exception as ex:
            logger.exception("保存设置失败")
            self._show_error(f"✗ 保存失败: {str(ex)}")

    def _on_reset_click(self, e):
        """重置设置按钮点击事件"""
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("确认重置"),
            content=ft.Text("确定要将所有设置重置为默认值吗？\n此操作不可撤销。"),
            actions=[
                ft.TextButton(
                    "取消",
                    on_click=lambda _: self._page.pop_dialog()
                ),
                ft.TextButton(
                    "重置",
                    on_click=self._confirm_reset
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self._page.show_dialog(dialog)

    def _confirm_reset(self, e):
        """确认并执行重置"""
        try:
            # 重新加载默认配置
            self._config_manager.config = self._config_manager.default_config.copy()
            self._config_manager.save_config()

            # 重新加载UI
            self._load_config_to_ui()
            self._unsaved_changes.clear()
            self._update_save_button_state()

            self._page.pop_dialog()
            self._show_message("✓ 设置已重置", ft.Colors.GREEN)

            # 通知回调
            if self._on_settings_changed:
                self._on_settings_changed()

        except Exception as ex:
            logger.exception("重置设置失败")
            self._show_error(f"✗ 重置失败: {str(ex)}")

    def _on_browse_save_directory(self, e):
        """浏览保存目录"""
        def on_result(path: ft.FilePickerResultEvent):
            if path.path:
                self.save_directory_field.value = path.path
                self.save_directory_field.update()
                self._on_setting_changed("audio.save_directory", path.path)

        self._folder_picker.on_result = on_result
        self._page.overlay.append(self._folder_picker)
        self._page.update()
        self._folder_picker.get_directory_path()

    def _on_browse_log_directory(self, e):
        """浏览日志目录"""
        def on_result(path: ft.FilePickerResultEvent):
            if path.path:
                self.log_directory_field.value = path.path
                self.log_directory_field.update()
                self._on_setting_changed("logging.log_directory", path.path)

        self._folder_picker.on_result = on_result
        self._page.overlay.append(self._folder_picker)
        self._page.update()
        self._folder_picker.get_directory_path()

    # ========== 辅助方法 ==========

    def _validate_setting(self, key: str, value) -> bool:
        """验证设置值"""
        # 目前所有配置项都通过基本验证
        return True

    def _update_save_button_state(self):
        """更新保存按钮状态"""
        self.save_button.disabled = len(self._unsaved_changes) == 0
        self.save_button.update()

    def _show_message(self, message: str, bgcolor=ft.Colors.BLUE):
        """显示消息"""
        self._page.show_dialog(ft.SnackBar(
            ft.Text(message),
            bgcolor=bgcolor,
            duration=3000
        ))

    def _show_error(self, message: str):
        """显示错误消息"""
        self._show_message(message, ft.Colors.RED)

    def has_unsaved_changes(self) -> bool:
        """是否有未保存的更改"""
        return len(self._unsaved_changes) > 0
