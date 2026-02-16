import flet as ft
import logging
import asyncio
from typing import Optional, Callable
from core.tts_server import FastAPITSServer as TTSServer
from core.network import NetworkManager
from api.dependencies import update_service_config

logger = logging.getLogger(__name__)

TTS_MODE_CUSTOM_VOICE = "customvoice"
TTS_MODE_VOICE_DESIGN = "voicedesign"
TTS_MODE_VOICE_CLONE = "base"

MODE_LABELS = {
    TTS_MODE_CUSTOM_VOICE: "自定义语音",
    TTS_MODE_VOICE_DESIGN: "声音设计",
    TTS_MODE_VOICE_CLONE: "声音克隆",
}


class TTSServiceView(ft.Container):
    def __init__(
        self,
        page: ft.Page,
        tts_engine_getter: Callable,
        terminal,
        config_manager,
        model_manager,
        voice_library,
        on_service_state_change: Optional[Callable] = None,
        on_load_model: Optional[Callable] = None,
    ):
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.terminal = terminal
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.voice_library = voice_library
        self.on_service_state_change = on_service_state_change
        self.on_load_model = on_load_model

        self.network_manager = NetworkManager(log_callback=self._log_callback)
        self.server: Optional[TTSServer] = None

        self.port = self.config_manager.get("tts_service.port", 13650)
        self.current_mode = TTS_MODE_CUSTOM_VOICE

        self.port_input: Optional[ft.TextField] = None
        self.start_button: Optional[ft.Button] = None
        self.stop_button: Optional[ft.Button] = None
        self.status_card: Optional[ft.Container] = None
        self.url_text: Optional[ft.Text] = None
        self.stats_text: Optional[ft.Text] = None

        self.speaker_dropdown: Optional[ft.Dropdown] = None
        self.preset_dropdown: Optional[ft.Dropdown] = None
        self.clone_dropdown: Optional[ft.Dropdown] = None
        self.model_dropdown: Optional[ft.Dropdown] = None
        self.selected_model_id: Optional[str] = None

        self.url_text = ft.Text(
            "未运行", size=16, color=ft.Colors.GREY, weight=ft.FontWeight.BOLD
        )
        self.stats_text = ft.Text("总请求: 0 | 成功: 0 | 失败: 0", size=14)

        super().__init__(content=self.build(), expand=True)

    def _log_callback(self, message: str, level: str = "info"):
        if self.terminal:
            try:
                if level == "success":
                    self.terminal.add_log(f"✓ {message}")
                elif level == "error":
                    self.terminal.add_log(f"✗ {message}")
                elif level == "warning":
                    self.terminal.add_log(f"⚠ {message}")
                else:
                    self.terminal.add_log(message)
            except Exception as e:
                logger.error(f"Log callback error: {e}")

    def _update_service_status(self, running: bool):
        if running:
            self.status_card.bgcolor = ft.Colors.with_opacity(0.1, ft.Colors.GREEN)
            self.url_text.value = f"http://{self._get_server_url()}"
            self.url_text.color = ft.Colors.GREEN
            self.start_button.disabled = True
            self.stop_button.disabled = False
        else:
            self.status_card.bgcolor = ft.Colors.with_opacity(0.1, ft.Colors.GREY)
            self.url_text.value = "未运行"
            self.url_text.color = ft.Colors.GREY
            self.start_button.disabled = False
            self.stop_button.disabled = True

        self.update()

        if self.on_service_state_change:
            try:
                self.on_service_state_change(running)
            except Exception as e:
                logger.error(f"Service state change callback error: {e}")

    def _get_server_url(self) -> str:
        local_ip = self.network_manager.get_local_ip()
        if local_ip:
            return f"{local_ip}:{self.port}"
        return f"localhost:{self.port}"

    def _on_tab_change(self, e):
        mode_map = [TTS_MODE_CUSTOM_VOICE, TTS_MODE_VOICE_DESIGN, TTS_MODE_VOICE_CLONE]
        idx = e.data if isinstance(e.data, int) else int(e.data) if e.data else 0
        if 0 <= idx < len(mode_map):
            self.current_mode = mode_map[idx]
            self._log_callback(
                f"切换到 {MODE_LABELS.get(self.current_mode, self.current_mode)} 模式"
            )

    def _build_model_selector(self, model_type: str):
        models = []
        if self.model_manager:
            models = self.model_manager.list_usable_models_by_type(model_type)

        self.model_dropdown = ft.Dropdown(
            label="模型",
            options=[ft.dropdown.Option(m) for m in models]
            if models
            else [ft.dropdown.Option("无可用模型")],
            value=models[0] if models else None,
            width=200,
        )
        if models:
            self.selected_model_id = models[0]

        return ft.Row(
            [
                ft.Icon(ft.Icons.STORAGE, size=20, color=ft.Colors.TEAL),
                self.model_dropdown,
            ],
            spacing=10,
        )

    def _on_model_change(self, e):
        self.selected_model_id = e.control.value
        self._log_callback(f"已选择模型: {self.selected_model_id}")

    def _build_custom_voice_config(self):
        speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna"]
        if self.voice_library:
            speakers = self.voice_library.get_custom_voice_speakers()
        self.speaker_dropdown = ft.Dropdown(
            label="说话人",
            options=[ft.dropdown.Option(s) for s in speakers],
            value=speakers[0] if speakers else "Vivian",
            width=200,
        )
        return ft.Column(
            [
                self._build_model_selector(TTS_MODE_CUSTOM_VOICE),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                ft.Row(
                    [
                        ft.Icon(ft.Icons.PERSON, size=20, color=ft.Colors.BLUE),
                        self.speaker_dropdown,
                    ],
                    spacing=10,
                ),
            ],
            spacing=5,
        )

    def _build_voice_design_config(self):
        presets = []
        if self.voice_library:
            presets = list(self.voice_library.get_all_design_presets().keys())
        self.preset_dropdown = ft.Dropdown(
            label="预设",
            options=[ft.dropdown.Option(p) for p in presets]
            if presets
            else [ft.dropdown.Option("无预设")],
            value=presets[0] if presets else None,
            width=200,
        )
        return ft.Column(
            [
                self._build_model_selector(TTS_MODE_VOICE_DESIGN),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                ft.Row(
                    [
                        ft.Icon(ft.Icons.PALETTE, size=20, color=ft.Colors.PURPLE),
                        self.preset_dropdown,
                    ],
                    spacing=10,
                ),
            ],
            spacing=5,
        )

    def _build_voice_clone_config(self):
        clones = []
        if self.voice_library:
            clones = [c["name"] for c in self.voice_library.get_all_clones()]
        self.clone_dropdown = ft.Dropdown(
            label="克隆音色",
            options=[ft.dropdown.Option(c) for c in clones]
            if clones
            else [ft.dropdown.Option("无克隆音色")],
            value=clones[0] if clones else None,
            width=200,
        )
        return ft.Column(
            [
                self._build_model_selector(TTS_MODE_VOICE_CLONE),
                ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CONTENT_COPY, size=20, color=ft.Colors.ORANGE),
                        self.clone_dropdown,
                    ],
                    spacing=10,
                ),
            ],
            spacing=5,
        )

    def _start_service(self, e):
        self._do_start_service()

    def _do_start_service(self):
        try:
            try:
                port = int(self.port_input.value)
                if port < 1 or port > 65535:
                    raise ValueError("端口必须在 1-65535 之间")
            except ValueError as err:
                self._page.show_dialog(
                    ft.SnackBar(ft.Text(f"端口无效: {err}"), duration=3000)
                )
                return

            self.port = port
            self.config_manager.set("tts_service.port", self.port)
            self.config_manager.save_config()

            self.start_button.disabled = True
            self.start_button.text = "加载中..."
            self.update()

            self._load_model_and_start(port)

        except Exception as ex:
            self._log_callback(f"启动服务失败: {str(ex)}", "error")
            self._reset_start_button()

    def _load_model_and_start(self, port: int):
        async def load_and_start():
            try:
                engine = self.tts_engine_getter()
                if engine is None:
                    self._log_callback(
                        f"正在加载 {MODE_LABELS.get(self.current_mode, self.current_mode)} 模型..."
                    )
                    if self.on_load_model:
                        success = await self.on_load_model(self.current_mode)
                        if not success:
                            self._log_callback("模型加载失败", "error")
                            self._reset_start_button()
                            return
                    else:
                        self._log_callback("无法加载模型：未提供加载回调", "error")
                        self._reset_start_button()
                        return
                    engine = self.tts_engine_getter()

                if engine is None:
                    self._log_callback("模型加载后引擎仍不可用", "error")
                    self._reset_start_button()
                    return

                self._log_callback("模型已就绪，启动服务...", "info")
                self._save_service_config()
                await self._start_server(port)

            except Exception as ex:
                self._log_callback(f"启动过程出错: {str(ex)}", "error")
                self._reset_start_button()

        asyncio.create_task(load_and_start())

    def _save_service_config(self):
        speaker = self.speaker_dropdown.value if self.speaker_dropdown else "Vivian"
        preset = self.preset_dropdown.value if self.preset_dropdown else None
        clone_prompt = None

        if self.current_mode == TTS_MODE_VOICE_CLONE and self.voice_library:
            clone_name = self.clone_dropdown.value if self.clone_dropdown else None
            if clone_name:
                for clone in self.voice_library.get_all_clones():
                    if clone.get("name") == clone_name:
                        clone_id = clone.get("id")
                        if clone_id:
                            clone_data = self.voice_library.get_clone(clone_id)
                            if clone_data:
                                clone_prompt = clone_data.get("prompt_features")
                        break

        update_service_config(
            mode=self.current_mode,
            model_id=self.selected_model_id,
            speaker=speaker,
            preset=preset,
            clone_prompt=clone_prompt,
        )
        self._log_callback(
            f"配置已保存: 模式={MODE_LABELS.get(self.current_mode)}, 说话人/预设={speaker or preset}"
        )

    async def _start_server(self, port: int):
        try:
            if self.server and self.server.is_running():
                self._do_stop_server_sync()
                await asyncio.sleep(0.5)

            self.server = TTSServer(
                host="0.0.0.0",
                port=port,
                tts_engine_getter=self.tts_engine_getter,
                log_callback=self._log_callback,
            )

            self.server.start()
            self._update_service_status(True)
            self._log_callback(f"TTS 服务已启动在端口 {port}", "success")

        except Exception as ex:
            self._log_callback(f"启动服务器失败: {str(ex)}", "error")
            self._reset_start_button()

    def _reset_start_button(self):
        self.start_button.disabled = False
        self.start_button.text = "启动服务"
        self.stop_button.disabled = True
        self.update()

    def _stop_service(self, e):
        self._do_stop_service()

    def _do_stop_service(self):
        try:
            if self.server:
                self._log_callback("正在停止 TTS 服务...", "info")
                self._do_stop_server_sync()
                self._log_callback("TTS 服务已停止", "success")

            self._update_service_status(False)

        except Exception as ex:
            self._log_callback(f"停止服务失败: {str(ex)}", "error")

    def _do_stop_server_sync(self):
        if self.server:
            try:
                if (
                    hasattr(self.server, "_uvicorn_server")
                    and self.server._uvicorn_server
                ):
                    self.server._uvicorn_server.should_exit = True

                self.server._running = False

                if self.server._server_thread and self.server._server_thread.is_alive():
                    self.server._server_thread.join(timeout=3)

                self.server = None

            except Exception as ex:
                logger.error(f"停止服务器时出错: {ex}")
                self.server = None

    def _refresh_stats(self):
        if self.server and self.server.is_running():
            stats = self.server.get_stats()
            self.stats_text.value = (
                f"总请求: {stats['total_requests']} | "
                f"成功: {stats['successful_requests']} | "
                f"失败: {stats['failed_requests']}"
            )
            self.update()

            if stats["recent_requests"]:
                for req in reversed(stats["recent_requests"]):
                    self._log_callback(
                        f"[{req['time']}] {req['status']}",
                        "success" if req["success"] else "error",
                    )

    def build(self):
        self.port_input = ft.TextField(
            label="服务端口",
            value=str(self.port),
            width=150,
            text_align=ft.TextAlign.CENTER,
            input_filter=ft.NumbersOnlyInputFilter(),
        )

        self.start_button = ft.Button(
            "启动服务",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._start_service,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE),
        )

        self.stop_button = ft.Button(
            "停止服务",
            icon=ft.Icons.STOP,
            on_click=self._stop_service,
            disabled=True,
            style=ft.ButtonStyle(bgcolor=ft.Colors.RED, color=ft.Colors.WHITE),
        )

        self.mode_tabs = ft.Tabs(
            selected_index=0,
            length=3,
            animation_duration=300,
            content=ft.Column(
                [
                    ft.TabBar(
                        tab_alignment=ft.TabAlignment.START,
                        indicator_color=ft.Colors.BLUE,
                        tabs=[
                            ft.Tab(icon=ft.Icons.MIC, label="自定义语音"),
                            ft.Tab(icon=ft.Icons.PALETTE, label="声音设计"),
                            ft.Tab(icon=ft.Icons.CONTENT_COPY, label="声音克隆"),
                        ],
                    ),
                    ft.Container(
                        content=ft.TabBarView(
                            [
                                ft.Container(
                                    content=self._build_custom_voice_config(),
                                    padding=15,
                                ),
                                ft.Container(
                                    content=self._build_voice_design_config(),
                                    padding=15,
                                ),
                                ft.Container(
                                    content=self._build_voice_clone_config(), padding=15
                                ),
                            ],
                        ),
                        bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
                        border_radius=8,
                        height=150,
                    ),
                ]
            ),
            on_change=self._on_tab_change,
        )

        self.status_card = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.ROUTER, size=30, color=ft.Colors.BLUE),
                            ft.Text("服务状态", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            ft.Text("访问地址:", size=14, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                        ]
                    ),
                    self.url_text,
                    ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            ft.Text("统计信息:", size=14, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                        ]
                    ),
                    self.stats_text,
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY),
        )

        control_card = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.SETTINGS, size=30, color=ft.Colors.BLUE),
                            ft.Text("服务控制", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            ft.Text("端口:", size=14),
                            self.port_input,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=10,
                    ),
                    ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                    ft.Row(
                        [
                            self.start_button,
                            self.stop_button,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=10,
                    ),
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

        mode_card = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.TUNE, size=30, color=ft.Colors.PURPLE),
                            ft.Text("服务模式", size=18, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=10,
                    ),
                    ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                    self.mode_tabs,
                ],
                spacing=5,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

        result = ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CLOUD, size=40, color=ft.Colors.BLUE),
                        ft.Text("TTS 服务管理", size=24, weight=ft.FontWeight.BOLD),
                    ],
                    spacing=15,
                ),
                ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                ft.Row(
                    [
                        ft.Column(
                            [
                                self.status_card,
                            ],
                            expand=1,
                            spacing=0,
                        ),
                        ft.VerticalDivider(width=20, color=ft.Colors.TRANSPARENT),
                        ft.Column(
                            [
                                control_card,
                            ],
                            expand=1,
                            spacing=0,
                        ),
                    ],
                    expand=False,
                    alignment=ft.MainAxisAlignment.START,
                ),
                ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                mode_card,
            ],
            expand=True,
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
        )

        return result

    def cleanup(self):
        if self.server and self.server.is_running():
            self._do_stop_server_sync()
