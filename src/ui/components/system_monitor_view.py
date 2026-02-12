"""
PhantomVox 系统资源监控组件

显示 CPU、内存、GPU 和显存占用百分比
"""

import asyncio
import flet as ft
import logging
import psutil

logger = logging.getLogger(__name__)


class SystemMonitorView:
    """系统资源监控视图组件"""

    def __init__(self, page: ft.Page, update_interval: float = 2.0):
        """
        初始化系统监控组件

        Args:
            page: Flet Page 对象
            update_interval: 更新间隔（秒），默认 2 秒
        """
        self.page = page
        self.update_interval = update_interval
        self._monitoring = False
        self._monitor_task = None

        # 创建显示控件
        self.cpu_label = ft.Text("CPU:", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.memory_label = ft.Text("内存:", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.gpu_label = ft.Text("GPU:", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.vram_label = ft.Text("显存:", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

        self.cpu_text = ft.Text("--%", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.memory_text = ft.Text("--%", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.gpu_text = ft.Text("--%", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.vram_text = ft.Text("--%", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

        # 创建进度条
        self.cpu_progress = ft.ProgressBar(
            width=36, height=3, value=0, bar_height=3,
            color=ft.Colors.BLUE, bgcolor=ft.Colors.GREY_300
        )
        self.memory_progress = ft.ProgressBar(
            width=36, height=3, value=0, bar_height=3,
            color=ft.Colors.GREEN, bgcolor=ft.Colors.GREY_300
        )
        self.gpu_progress = ft.ProgressBar(
            width=36, height=3, value=0, bar_height=3,
            color=ft.Colors.ORANGE, bgcolor=ft.Colors.GREY_300
        )
        self.vram_progress = ft.ProgressBar(
            width=36, height=3, value=0, bar_height=3,
            color=ft.Colors.PURPLE, bgcolor=ft.Colors.GREY_300
        )

        # 检测是否有 CUDA 可用
        self._cuda_available = self._check_cuda_available()

    def _check_cuda_available(self) -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False

    def _get_cpu_percent(self) -> float:
        """获取 CPU 使用率"""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    def _get_memory_percent(self) -> float:
        """获取内存使用率"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception:
            return 0.0

    def _get_gpu_info(self) -> tuple[float, float]:
        """
        获取 GPU 使用率和显存使用率

        Returns:
            tuple: (gpu_percent, vram_percent)
        """
        if not self._cuda_available:
            return 0.0, 0.0

        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0, 0.0

            # 获取 GPU 使用率和显存使用率（使用 pynvml）
            gpu_percent = 0.0
            vram_percent = 0.0

            try:
                import pynvml
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

                    # 使用 pynvml 获取显存信息（更准确，包含所有进程）
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0.0
                finally:
                    pynvml.nvmlShutdown()
            except ImportError:
                # pynvml 不可用，回退到 torch
                try:
                    if hasattr(torch.cuda, 'utilization'):
                        gpu_percent = torch.cuda.utilization()
                    vram_allocated = torch.cuda.memory_allocated()
                    vram_total = torch.cuda.get_device_properties(0).total_memory
                    vram_percent = (vram_allocated / vram_total * 100) if vram_total > 0 else 0.0
                except Exception:
                    pass
            except Exception:
                # pynvml 调用失败，回退到 torch
                try:
                    if hasattr(torch.cuda, 'utilization'):
                        gpu_percent = torch.cuda.utilization()
                    vram_allocated = torch.cuda.memory_allocated()
                    vram_total = torch.cuda.get_device_properties(0).total_memory
                    vram_percent = (vram_allocated / vram_total * 100) if vram_total > 0 else 0.0
                except Exception:
                    pass

            return gpu_percent, vram_percent
        except Exception:
            return 0.0, 0.0

    def _update_monitor(self):
        """更新监控数据"""
        # 获取系统资源数据
        cpu_percent = self._get_cpu_percent()
        memory_percent = self._get_memory_percent()
        gpu_percent, vram_percent = self._get_gpu_info()

        # 更新文本显示
        self.cpu_text.value = f"{cpu_percent:.0f}%"
        self.memory_text.value = f"{memory_percent:.0f}%"

        if self._cuda_available:
            self.gpu_text.value = f"{gpu_percent:.0f}%"
            self.vram_text.value = f"{vram_percent:.0f}%"
        else:
            self.gpu_text.value = "N/A"
            self.vram_text.value = "N/A"

        # 更新进度条（0-1 范围）
        self.cpu_progress.value = cpu_percent / 100
        self.memory_progress.value = memory_percent / 100
        self.gpu_progress.value = gpu_percent / 100 if self._cuda_available else 0
        self.vram_progress.value = vram_percent / 100 if self._cuda_available else 0

        # 检查监控是否仍在运行
        if not self._monitoring:
            return

        # 更新页面
        try:
            # 检查页面是否有效
            if self.page and hasattr(self.page, 'session') and self.page.session:
                self.page.update()
            else:
                # 页面会话已失效，停止监控
                self._monitoring = False
        except RuntimeError:
            # 页面已关闭，停止监控（静默处理）
            self._monitoring = False
        except Exception:
            # 捕获其他异常（如连接重置等，静默处理）
            self._monitoring = False

    async def _monitor_loop(self):
        """监控循环（异步执行，在主线程中运行）"""
        while self._monitoring:
            try:
                self._update_monitor()
            except Exception as e:
                logger.debug(f"监控更新异常: {e}")
                break
            await asyncio.sleep(self.update_interval)

    def start_monitoring(self):
        """启动监控（使用异步任务）"""
        if not self._monitoring:
            self._monitoring = True
            # 使用 asyncio 创建异步任务
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        """停止监控"""
        # 先设置标志位，让循环自然退出
        self._monitoring = False

        if self._monitor_task and not self._monitor_task.done():
            # 取消异步任务
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass  # 任务被取消是预期的
            except Exception as e:
                logger.debug(f"停止监控任务时出现异常: {e}")

    def build(self) -> ft.Container:
        """
        构建监控组件 UI

        Returns:
            ft.Container: 监控组件容器
        """
        # 创建监控项 - 格式：标签 + 进度条 + 百分比
        cpu_row = ft.Row(
            [self.cpu_label, self.cpu_progress, self.cpu_text],
            spacing=1, tight=True
        )
        memory_row = ft.Row(
            [self.memory_label, self.memory_progress, self.memory_text],
            spacing=1, tight=True
        )
        gpu_row = ft.Row(
            [self.gpu_label, self.gpu_progress, self.gpu_text],
            spacing=1, tight=True
        )
        vram_row = ft.Row(
            [self.vram_label, self.vram_progress, self.vram_text],
            spacing=1, tight=True
        )

        # 创建主容器
        container = ft.Container(
            content=ft.Column(
                [
                    cpu_row,
                    memory_row,
                    gpu_row,
                    vram_row,
                ],
                spacing=2,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                tight=True,
            ),
            padding=ft.padding.all(5),
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.ON_SURFACE),
            border_radius=ft.border_radius.only(
                bottom_left=8, bottom_right=8
            ),
            width=100
        )

        # 启动监控
        self.start_monitoring()

        return container
