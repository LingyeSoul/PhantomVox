"""
TTS 服务管理界面

提供 TTS HTTP 服务的启动/停止、配置管理、状态监控功能
请求日志会输出到运行日志控件中
"""

import flet as ft
import logging
from typing import Optional
from core.tts_server import FastAPITSServer as TTSServer
from core.network import NetworkManager

logger = logging.getLogger(__name__)


class TTSServiceView(ft.Container):
    """TTS 服务管理视图"""

    def __init__(
        self,
        page: ft.Page,
        tts_engine_getter,
        terminal,
        config_manager,
        on_service_state_change: Optional[callable] = None
    ):
        """初始化 TTS 服务管理视图

        Args:
            page: Flet Page 对象
            tts_engine_getter: TTS 引擎获取函数
            terminal: 终端对象（用于日志输出）
            config_manager: 配置管理器
            on_service_state_change: 服务状态变化回调
        """
        self._page = page
        self.tts_engine_getter = tts_engine_getter
        self.terminal = terminal
        self.config_manager = config_manager
        self.on_service_state_change = on_service_state_change

        # 网络管理器
        self.network_manager = NetworkManager(log_callback=self._log_callback)

        # TTS 服务器实例
        self.server: Optional[TTSServer] = None

        # 从配置加载端口
        self.port = self.config_manager.get("tts_service.port", 13650)
        self.auto_start = self.config_manager.get("tts_service.auto_start", False)

        # UI 控件引用
        self.port_input: Optional[ft.TextField] = None
        self.start_button: Optional[ft.Button] = None
        self.stop_button: Optional[ft.Button] = None
        self.status_card: Optional[ft.Container] = None
        self.url_text: Optional[ft.Text] = None
        self.stats_text: Optional[ft.Text] = None

        # 先创建这些控件
        self.url_text = ft.Text(
            "未运行",
            size=16,
            color=ft.Colors.GREY,
            weight=ft.FontWeight.BOLD
        )
        self.stats_text = ft.Text(
            "总请求: 0 | 成功: 0 | 失败: 0",
            size=14
        )

        # 构建UI并设置容器内容
        super().__init__(
            content=self.build(),
            expand=True
        )

    def _log_callback(self, message: str, level: str = 'info'):
        """日志回调"""
        if self.terminal:
            try:
                if level == 'success':
                    self.terminal.add_log(f"✓ {message}")
                elif level == 'error':
                    self.terminal.add_log(f"✗ {message}")
                elif level == 'warning':
                    self.terminal.add_log(f"⚠ {message}")
                else:
                    self.terminal.add_log(message)
            except Exception as e:
                logger.error(f"Log callback error: {e}")

    def _update_service_status(self, running: bool):
        """更新服务状态 UI"""
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

        # 使用父容器的 update 来批量更新所有子控件
        self.update()

        # 调用状态变化回调
        if self.on_service_state_change:
            try:
                self.on_service_state_change(running)
            except Exception as e:
                logger.error(f"Service state change callback error: {e}")

    def _get_server_url(self) -> str:
        """获取服务器 URL"""
        local_ip = self.network_manager.get_local_ip()
        if local_ip:
            return f"{local_ip}:{self.port}"
        return f"localhost:{self.port}"

    def _start_service(self, e):
        """启动 TTS 服务（异步）"""
        try:
            # 获取端口
            try:
                port = int(self.port_input.value)
                if port < 1 or port > 65535:
                    raise ValueError("端口必须在 1-65535 之间")
            except ValueError as err:
                self._page.show_dialog(ft.SnackBar(
                    ft.Text(f"端口无效: {err}"),
                    duration=3000
                ))
                return

            # 立即保存端口到配置（点击启动按钮时保存）
            self.port = port
            self.config_manager.set("tts_service.port", self.port)
            self.config_manager.save_config()

            # 停止现有服务
            if self.server and self.server.is_running():
                self.server.stop()
                import time
                time.sleep(0.5)  # 等待服务完全停止

            # 创建并启动服务器（使用新的端口）
            self.server = TTSServer(
                host="0.0.0.0",
                port=self.port,
                tts_engine_getter=self.tts_engine_getter,
                log_callback=self._log_callback
            )

            # 立即更新 UI（不阻塞）
            self._update_service_status(True)
            self._log_callback(f"正在启动 TTS 服务，端口 {self.port}...", 'info')

            # 启动服务器（异步）
            self.server.start()

            # 延迟检查服务器状态
            import threading
            import socket
            def check_server_started():
                import time
                time.sleep(2)  # 等待服务器完全启动

                # 1. 检查服务器线程状态
                thread_alive = (self.server and
                              self.server.server_thread and
                              self.server.server_thread.is_alive())

                # 2. 尝试实际连接到服务器
                can_connect = False
                if thread_alive:
                    try:
                        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        test_sock.settimeout(1)
                        test_sock.connect(('127.0.0.1', self.port))
                        test_sock.close()
                        can_connect = True
                    except Exception:
                        pass

                # 3. 综合判断
                if thread_alive and can_connect:
                    self._log_callback(f"TTS 服务已成功启动在端口 {self.port}", 'success')
                elif thread_alive and not can_connect:
                    self._update_service_status(False)
                    self._log_callback("TTS 服务线程运行中，但端口无法连接", 'error')
                else:
                    self._update_service_status(False)
                    self._log_callback("TTS 服务启动失败（线程已结束）", 'error')

            threading.Thread(target=check_server_started, daemon=True).start()

            # 启动状态更新定时器
            self._start_status_update_timer()

        except Exception as ex:
            self._log_callback(f"启动服务失败: {str(ex)}", 'error')
            self._update_service_status(False)

    def _stop_service(self, e):
        """停止 TTS 服务（异步）"""
        try:
            if self.server:
                self._log_callback("正在停止 TTS 服务...", 'info')

                # 在后台线程中停止服务器
                import threading
                def stop_server():
                    try:
                        self.server.stop()
                        self._log_callback("TTS 服务已停止", 'success')
                    except Exception as ex:
                        self._log_callback(f"停止服务失败: {str(ex)}", 'error')

                threading.Thread(target=stop_server, daemon=True).start()

            # 立即更新 UI
            self._update_service_status(False)

        except Exception as ex:
            self._log_callback(f"停止服务失败: {str(ex)}", 'error')

    def _refresh_stats(self):
        """刷新统计信息"""
        if self.server and self.server.is_running():
            stats = self.server.get_stats()

            # 更新统计文本
            self.stats_text.value = (
                f"总请求: {stats['total_requests']} | "
                f"成功: {stats['successful_requests']} | "
                f"失败: {stats['failed_requests']}"
            )
            self.update()  # 批量更新所有子控件

            # 输出新的请求日志到 terminal
            if stats['recent_requests']:
                for req in reversed(stats['recent_requests']):
                    status_icon = "✓" if req['success'] else "✗"
                    self._log_callback(
                        f"[{req['time']}] {req['status']}",
                        'success' if req['success'] else 'error'
                    )

    def _start_status_update_timer(self):
        """启动状态更新定时器"""
        def update_timer():
            import asyncio
            while self.server and self.server.is_running():
                try:
                    self._refresh_stats()
                except Exception:
                    pass
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: __import__('time').sleep(2)
                )

    def build(self):
        """构建服务管理界面"""

        # 端口输入
        self.port_input = ft.TextField(
            label="服务端口",
            value=str(self.port),
            width=150,
            text_align=ft.TextAlign.CENTER,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        # 启动/停止按钮
        self.start_button = ft.Button(
            "启动服务",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._start_service,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.GREEN,
                color=ft.Colors.WHITE
            )
        )

        self.stop_button = ft.Button(
            "停止服务",
            icon=ft.Icons.STOP,
            on_click=self._stop_service,
            disabled=True,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.RED,
                color=ft.Colors.WHITE
            )
        )

        # 状态卡片
        self.status_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.ROUTER, size=30, color=ft.Colors.BLUE),
                    ft.Text("服务状态", size=18, weight=ft.FontWeight.BOLD),
                ], spacing=10),
                ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                ft.Row([
                    ft.Text("访问地址:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                ]),
                self.url_text,
                ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
                ft.Row([
                    ft.Text("统计信息:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                ]),
                self.stats_text,
            ], spacing=5),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY),
        )

        # 控制卡片
        control_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SETTINGS, size=30, color=ft.Colors.BLUE),
                    ft.Text("服务控制", size=18, weight=ft.FontWeight.BOLD),
                ], spacing=10),
                ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                ft.Row([
                    ft.Text("端口:", size=14),
                    self.port_input,
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                ft.Row([
                    self.start_button,
                    self.stop_button,
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            ], spacing=5),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
        )

        # 主布局
        return ft.Column([
            # 标题行
            ft.Row([
                ft.Icon(ft.Icons.CLOUD, size=40, color=ft.Colors.BLUE),
                ft.Text("TTS 服务管理", size=24, weight=ft.FontWeight.BOLD),
            ], spacing=15),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

            # 左右两列布局
            ft.Row([
                # 左列：服务状态
                ft.Column([
                    self.status_card,
                ], expand=1, spacing=0),

                ft.VerticalDivider(width=20, color=ft.Colors.TRANSPARENT),

                # 右列：服务控制
                ft.Column([
                    control_card,
                ], expand=1, spacing=0),
            ], expand=True, alignment=ft.MainAxisAlignment.START),

        ], expand=True, spacing=0)

    def cleanup(self):
        """清理资源（需要在视图销毁时手动调用）"""
        if self.server and self.server.is_running():
            # 在后台线程中停止服务器，避免阻塞
            import threading
            def stop_and_cleanup():
                try:
                    self.server.stop()
                except Exception:
                    pass
            threading.Thread(target=stop_and_cleanup, daemon=True).start()
