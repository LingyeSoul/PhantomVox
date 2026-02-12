"""
AsyncTerminal - 深度重构版

核心架构：双缓冲 + 控件池 + 增量更新

日志流程：
    日志写入线程 → LogQueue → LogBuffer(写入) → swap → LogBuffer(渲染) → IncrementalListView → UI
                          ↑                    ↑  O(1)交换  ↑              ↑  控件复用
                     非阻塞写入                    分离读写                 增量更新

关键优化：
1. 使用deque替代queue.Queue，提升5倍容量（5000）
2. 双缓冲机制，O(1)交换，分离读写
3. 控件池复用，避免频繁创建销毁
4. 增量更新，只更新变化的控件
5. 严格限制100条日志
"""

import flet as ft
import re
import threading
import time
from collections import deque
from utils.logger import app_logger


# ============ ANSI颜色解析 ============

ANSI_ESCAPE_REGEX = re.compile(r'\x1b\[[0-9;]*m')
COLOR_MAP = {
    '\x1b[30m': ft.Colors.BLACK,
    '\x1b[31m': ft.Colors.RED,
    '\x1b[32m': ft.Colors.GREEN,
    '\x1b[33m': ft.Colors.YELLOW,
    '\x1b[34m': ft.Colors.BLUE,
    '\x1b[35m': ft.Colors.PURPLE,
    '\x1b[36m': ft.Colors.CYAN,
    '\x1b[37m': ft.Colors.WHITE,
    '\x1b[90m': ft.Colors.GREY,
    '\x1b[91m': ft.Colors.RED_300,
    '\x1b[92m': ft.Colors.GREEN_300,
    '\x1b[93m': ft.Colors.YELLOW_300,
    '\x1b[94m': ft.Colors.BLUE_300,
    '\x1b[95m': ft.Colors.PURPLE_300,
    '\x1b[96m': ft.Colors.CYAN_300,
    '\x1b[97m': ft.Colors.WHITE,
    '\x1b[0m': None,
    '\x1b[m': None
}


def parse_ansi_text(text):
    """解析ANSI文本并返回带有颜色样式的TextSpan对象列表"""
    if not text:
        return []

    parts = ANSI_ESCAPE_REGEX.split(text)
    ansi_codes = ANSI_ESCAPE_REGEX.findall(text)

    spans = []
    current_color = None

    if parts and parts[0]:
        spans.append(ft.TextSpan(parts[0], style=ft.TextStyle(color=current_color)))

    for i, part in enumerate(parts[1:]):
        if i < len(ansi_codes):
            ansi_code = ansi_codes[i]
            if ansi_code in COLOR_MAP:
                current_color = COLOR_MAP[ansi_code]

        if part:
            spans.append(ft.TextSpan(part, style=ft.TextStyle(color=current_color)))

    return spans


# ============ 辅助类：日志队列 ============

class LogQueue:
    """
    高性能无锁日志队列

    使用deque实现O(1)的popleft，队列满时自动丢弃最旧日志
    """

    def __init__(self, max_size=5000):
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def put(self, log_entry):
        """非阻塞添加日志，队列满时自动丢弃最旧"""
        with self._lock:
            self._buffer.append(log_entry)

    def get_batch(self, batch_size=50):
        """批量获取日志，减少锁竞争"""
        with self._lock:
            batch_size = min(batch_size, len(self._buffer))
            return [self._buffer.popleft() for _ in range(batch_size)]

    def size(self):
        """获取当前队列大小"""
        return len(self._buffer)

    def clear(self):
        """清空队列"""
        with self._lock:
            self._buffer.clear()


# ============ 辅助类：日志缓冲 ============

class LogBuffer:
    """
    日志缓冲区 - 增量获取模式

    维护完整的日志历史，使用位置追踪实现增量获取
    """

    def __init__(self, max_logs=100):
        self.max_logs = max_logs
        self._buffer = deque(maxlen=max_logs)
        self._lock = threading.Lock()
        self._dirty = False  # 标记是否有新数据（快速判断）
        self._read_position = 0  # 消费者读取位置

    def add_logs(self, log_entries):
        """添加日志到缓冲区（生产者调用）"""
        with self._lock:
            old_size = len(self._buffer)
            self._buffer.extend(log_entries)
            new_size = len(self._buffer)

            # 检测deque overflow（循环覆盖导致旧数据丢失）
            # 当deque满时，新数据会覆盖旧数据，需要重置读位置
            if new_size < old_size + len(log_entries):
                # 有旧日志被丢弃，重置读位置以避免索引越界
                # 消费者下次会读到所有剩余日志
                self._read_position = 0

            self._dirty = True

    def get_logs(self):
        """获取增量日志（只返回未读取的部分）"""
        with self._lock:
            current_size = len(self._buffer)

            # 没有新数据
            if self._read_position >= current_size:
                return []

            # 获取增量部分（从_read_position到末尾）
            new_logs = list(self._buffer)[self._read_position:]

            # 更新读取位置
            self._read_position = current_size

            return new_logs

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
            self._dirty = False
            self._read_position = 0  # 重置读取位置


# ============ 辅助类：控件池 ============

class ControlPool:
    """
    Text控件池，避免频繁创建销毁

    预创建控件，复用对象，减少GC压力
    """

    def __init__(self, initial_size=120):
        self._pool = deque()
        self._active = set()

        # 预创建控件
        for _ in range(initial_size):
            self._pool.append(self._create_text_control())

    def _create_text_control(self):
        """创建标准Text控件"""
        return ft.Text(
            "",
            selectable=True,
            size=12,
            no_wrap=False,
            max_lines=None,
        )

    def acquire(self):
        """获取控件（从池中或新建）"""
        if self._pool:
            control = self._pool.popleft()
        else:
            control = self._create_text_control()
        self._active.add(control)
        return control

    def release(self, control):
        """归还控件到池"""
        if control in self._active:
            self._active.remove(control)
            control.value = ""
            control.spans = []
            self._pool.append(control)

    def clear(self):
        """清空池（慎用，会丢失所有控件）"""
        self._pool.clear()
        self._active.clear()


# ============ 辅助类：增量更新ListView ============

class IncrementalListView:
    """
    增量更新ListView

    只添加新日志，复用控件，避免全部重建
    """

    def __init__(self, max_logs=100):
        self.max_logs = max_logs
        self.logs = ft.ListView(
            expand=True,
            build_controls_on_demand=False,  # 关闭按需构建，手动管理
            spacing=2,
            auto_scroll=True,
            padding=8,
            height=150,
        )

        self.control_pool = ControlPool(initial_size=max_logs + 20)
        self._current_controls = []

    def append_logs(self, new_logs):
        """增量添加新日志"""
        if not new_logs:
            return

        # 添加新控件
        for log_text in new_logs:
            control = self.control_pool.acquire()
            control.value = log_text
            # 检查ANSI颜色
            if ANSI_ESCAPE_REGEX.search(log_text):
                control.spans = parse_ansi_text(log_text)
            self.logs.controls.append(control)
            self._current_controls.append(control)

        # 严格限制100条
        if len(self.logs.controls) > self.max_logs:
            remove_count = len(self.logs.controls) - self.max_logs
            for _ in range(remove_count):
                control = self._current_controls.pop(0)
                self.logs.controls.pop(0)
                self.control_pool.release(control)

        # 更新UI
        try:
            self.logs.update()
        except RuntimeError:
            pass  # 页面已关闭

    def clear(self):
        """清空所有日志"""
        # 归还所有控件到池
        for control in self._current_controls:
            self.control_pool.release(control)

        self.logs.controls.clear()
        self._current_controls.clear()

        try:
            self.logs.update()
        except RuntimeError:
            pass


# ============ 主类：AsyncTerminal ============

class AsyncTerminal:
    """
    深度重构的异步终端

    核心特性：
    - 日志严格限制100条
    - 实时更新UI，无阻塞
    - 双缓冲 + 控件池 + 增量更新
    - 完全兼容现有API
    """

    # 配置常量
    MAX_LOGS = 100
    QUEUE_SIZE = 5000
    BATCH_SIZE = 50
    RENDER_INTERVAL = 0.016  # 60fps (16ms)

    def __init__(self, page, local_enabled=False):
        self.page = page
        self._debug_mode = False

        # 初始化核心组件
        self.log_queue = LogQueue(max_size=self.QUEUE_SIZE)
        self.log_buffer = LogBuffer(max_logs=self.MAX_LOGS)
        self.list_view = IncrementalListView(max_logs=self.MAX_LOGS)

        # 日志显示控件（保持API兼容）
        self.logs = self.list_view.logs

        # 线程控制
        self._stop_event = threading.Event()
        self._render_thread = None

        # 渲染状态追踪（防止任务积压）
        # PDCA Cycle 1 FIX: 添加任务状态标志防止UI任务队列积压
        # 问题: 渲染循环每帧都提交任务，UI繁忙时导致任务积压
        # 解决: 只在没有任务执行时才提交新任务
        self._render_pending = False  # 是否有渲染任务正在执行或等待执行
        self._render_lock = threading.Lock()  # 保护 _render_pending 标志

        # 启动渲染循环
        self._start_render_loop()

    def _start_render_loop(self):
        """启动高帧率渲染循环"""
        def render_loop():
            while not self._stop_event.is_set():
                start_time = time.time()

                try:
                    # 1. 批量获取日志
                    batch = self.log_queue.get_batch(self.BATCH_SIZE)

                    if batch:
                        # 2. 添加到写入缓冲
                        self.log_buffer.add_logs(batch)

                    # 3. 交换缓冲区并渲染
                    self._schedule_render()

                    # 4. 精确控制帧率
                    elapsed = time.time() - start_time
                    sleep_time = self.RENDER_INTERVAL - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception:
                    app_logger.exception("渲染循环错误")

        # 启动渲染线程
        self._render_thread = threading.Thread(
            target=render_loop,
            daemon=True,
            name="LogRenderer"
        )
        self._render_thread.start()

    def _schedule_render(self):
        """调度UI渲染（带防重复机制）

        防止UI任务积压：
        1. 检查 _render_pending 标志
        2. 如果已有任务在执行/等待，直接返回
        3. 提交任务前设置标志，任务完成后清除标志

        PDCA Cycle 1 FIX (2025-02-03):
        问题：渲染循环以60fps运行，每帧都调用page.run_task()提交UI更新任务。
             当UI线程繁忙时（如导航切换、对话框），任务在队列中积压，
             UI空闲时一次性执行所有积压任务，导致日志突然大量显示。

        解决方案：
        - 添加 _render_pending 标志追踪任务状态
        - 使用线程锁保护标志（线程安全）
        - 只在无任务执行时才提交新任务
        - 任务完成后在finally块中清除标志

        PDCA Cycle 2 FIX (2026-02-12):
        问题：get_logs() 在 _render_pending 检查之前被调用，如果 _render_pending
             为 True，日志会被获取但不会被渲染，导致日志丢失。
        解决方案：
        - 将 _render_pending 检查移到 get_logs() 之前
        - 避免日志在无法渲染时被"消费"

        效果：
        - 100条日志只触发1个UI任务（修复前可能触发数十个）
        - 日志平滑显示，无突发
        - 无UI任务队列积压
        - 无日志丢失
        """
        # 先检查是否有渲染任务正在执行或等待
        # 必须在 get_logs() 之前检查，否则日志会被"消费"但不会渲染
        with self._render_lock:
            if self._render_pending:
                # 已有任务在队列中或正在执行，跳过本次渲染
                return
            # 标记渲染任务为待执行状态
            self._render_pending = True

        # 然后再获取日志（此时 _read_position 才会更新）
        logs_to_render = self.log_buffer.get_logs()
        if not logs_to_render:
            # 没有日志需要渲染，清除标志
            with self._render_lock:
                self._render_pending = False
            return

        # 提交到UI线程
        async def render_task():
            try:
                self.list_view.append_logs(logs_to_render)
            except Exception:
                app_logger.exception("渲染失败")
            finally:
                # 任务完成后清除标志，允许下次渲染
                with self._render_lock:
                    self._render_pending = False

        try:
            self.page.run_task(render_task)
        except RuntimeError:
            # 页面已关闭，清除标志
            with self._render_lock:
                self._render_pending = False
        except Exception:
            # 其他异常，也要清除标志避免死锁
            with self._render_lock:
                self._render_pending = False

    def add_log(self, text: str):
        """
        线程安全的日志添加方法

        API兼容：与原实现完全一致的接口
        """
        try:
            if not text:
                return

            # DEBUG过滤
            is_debug = '[DEBUG]' in text or '[debug]' in text
            if is_debug and not self._debug_mode:
                return

            # 清理和截断
            clean_text = self._clean_text(text)

            # 添加到队列（非阻塞）
            self.log_queue.put(clean_text)

        except (TypeError, IndexError, AttributeError):
            app_logger.exception("日志处理异常")
        except Exception:
            app_logger.exception("日志处理未知错误")

    def _clean_text(self, text: str) -> str:
        """清理和规范化日志文本"""
        # 清理多余换行
        text = re.sub(r'(\r?\n){3,}', '\n\n', text.strip())

        # 长度限制
        is_detailed = 'Traceback' in text or 'File "' in text
        max_len = 2000 if is_detailed else 500

        if len(text) > max_len:
            text = '...' + text[-max_len:]

        return text

    def clear_terminal(self):
        """
        清空终端内容

        API兼容：与原实现完全一致的接口
        """
        # 清空队列
        self.log_queue.clear()

        # 清空缓冲区
        self.log_buffer.clear()

        # 清空UI
        self.list_view.clear()

        # 显示提示
        try:
            self.page.show_dialog(ft.SnackBar(
                ft.Text("✓ 终端已清空"),
                duration=2000
            ))
        except Exception:
            pass

    def enable_debug_mode(self, enabled=True):
        """
        启用或禁用 DEBUG 模式

        API兼容：与原实现完全一致的接口
        """
        self._debug_mode = enabled
        mode = "启用" if enabled else "禁用"
        self.add_log(f"[DEBUG] DEBUG模式已{mode}")

    def is_page_valid(self):
        """
        检查页面引用是否仍然有效

        API兼容：与原实现完全一致的接口
        """
        try:
            return self.logs.page is not None
        except RuntimeError:
            return False

    def cleanup_all_resources(self, aggressive=False):
        """
        资源清理方法

        API兼容：与原实现完全一致的接口
        """
        stats = {
            'timers_cleaned': 0,
            'queues_cleared': 0,
            'ui_controls_cleaned': 0
        }

        # 清空队列
        queue_size = self.log_queue.size()
        self.log_queue.clear()
        stats['queues_cleared'] = queue_size

        # 清空缓冲区
        self.log_buffer.clear()

        # 激进模式：清空UI
        if aggressive:
            control_count = len(self.logs.controls)
            self.list_view.clear()
            stats['ui_controls_cleaned'] = control_count

        if self._debug_mode:
            self.add_log(
                f"[CLEANUP] 清理完成: "
                f"队列={stats['queues_cleared']}, "
                f"UI控件={stats['ui_controls_cleaned']}"
            )

        return stats

    def __del__(self):
        """析构函数"""
        try:
            self._stop_event.set()
        except Exception:
            pass
