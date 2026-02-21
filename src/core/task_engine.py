"""
TTS 模型任务引擎

用于管理模型加载、卸载和推理任务的顺序执行，防止并发冲突
"""

import asyncio
import logging
from typing import Optional, Callable, Any, Dict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""

    GENERATE = "generate"  # 推理任务
    UNLOAD = "unload"  # 卸载任务
    LOAD = "load"  # 加载任务
    CUSTOM = "custom"  # 自定义任务


class Task:
    def __init__(
        self,
        task_type: TaskType,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        priority: int = 0,
        description: str = "",
        model_id: Optional[str] = None,
    ):
        self.task_type = task_type
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.description = description or f"{task_type.value}_task"
        self.model_id = model_id
        self.future: Optional[asyncio.Future] = None

    def __lt__(self, other):
        return self.priority > other.priority

    def __repr__(self):
        return f"Task(type={self.task_type.value}, desc={self.description})"


class TaskEngine:
    """
    TTS 模型任务引擎

    功能：
    1. 管理模型加载、卸载、推理任务的顺序执行
    2. 防止并发冲突（如推理时卸载模型）
    3. 支持任务优先级
    4. 提供任务状态查询
    """

    def __init__(self, max_queue_size: int = 100):
        """
        初始化任务引擎

        Args:
            max_queue_size: 队列最大大小
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()  # 改用 asyncio.Lock 适配异步环境
        self._current_task: Optional[Task] = None
        self._task_count = 0  # 已处理任务计数
        self._loaded_model_id: Optional[str] = None

        logger.info("任务引擎已初始化")

    async def start(self):
        """启动任务引擎"""
        async with self._lock:
            if self._running:
                logger.warning("任务引擎已在运行中")
                return

            self._running = True
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("任务引擎已启动")

    async def stop(self):
        """停止任务引擎"""
        async with self._lock:
            if not self._running:
                return

            self._running = False

        # 等待工作线程完成
        if self._worker_task:
            await self._queue.put(None)  # 发送停止信号
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("任务引擎停止超时")
                self._worker_task.cancel()

        logger.info("任务引擎已停止")

    async def _worker(self):
        """工作协程，从队列中取任务并执行"""
        logger.info("任务工作线程已启动")

        while self._running:
            try:
                # 从队列获取任务（带超时，避免永久阻塞）
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                # 检查停止信号
                if task is None:
                    break

                self._current_task = task

                try:
                    logger.info(f"执行任务 [{self._task_count}]: {task.description}")

                    # 执行任务
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        # 同步函数在线程池中执行
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, lambda: task.func(*task.args, **task.kwargs)
                        )

                    if task.future and not task.future.cancelled():
                        task.future.set_result(result)

                    if task.task_type == TaskType.LOAD and task.model_id:
                        self._loaded_model_id = task.model_id
                        logger.info(f"任务引擎记录已加载模型: {task.model_id}")
                    elif task.task_type == TaskType.UNLOAD:
                        self._loaded_model_id = None
                        logger.info("任务引擎清除已加载模型记录")

                    logger.info(f"✓ 任务完成 [{self._task_count}]: {task.description}")
                    self._task_count += 1

                except Exception as e:
                    logger.error(
                        f"✗ 任务失败 [{self._task_count}]: {task.description} - {str(e)}"
                    )
                    if task.future and not task.future.cancelled():
                        task.future.set_exception(e)

                finally:
                    self._current_task = None
                    self._queue.task_done()

            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                logger.error(f"工作线程异常: {str(e)}")

        logger.info("任务工作线程已退出")

    async def submit(
        self,
        task_type: TaskType,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        priority: int = 0,
        description: str = "",
        model_id: Optional[str] = None,
    ) -> Any:
        if not self._running:
            await self.start()

        task = Task(
            task_type=task_type,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            description=description,
            model_id=model_id,
        )

        task.future = asyncio.Future()

        # 加入队列
        await self._queue.put(task)

        logger.info(
            f"任务已加入队列: {task.description} (队列长度: {self._queue.qsize()})"
        )

        # 等待任务完成
        return await task.future

    async def submit_streaming(
        self,
        task_type: TaskType,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        description: str = "",
    ):
        """
        提交流式生成任务

        使用异步队列传输流式数据，调用者可立即开始迭代

        Args:
            task_type: 任务类型
            func: 要执行的流式函数（返回async generator）
            args: 位置参数
            kwargs: 关键字参数
            description: 任务描述

        Returns:
            AsyncGenerator: 流式结果生成器
        """
        # 自动启动任务引擎
        if not self._running:
            await self.start()

        # 创建通信队列（小缓冲，保持实时性）
        result_queue = asyncio.Queue(maxsize=5)
        started = asyncio.Event()

        async def streaming_worker():
            """执行流式任务的工作协程"""
            try:
                started.set()

                # 执行流式函数并将结果放入队列
                # 直接使用async for，适用于同步和异步生成器
                async for item in func(*args, **(kwargs or {})):
                    await result_queue.put(item)

                # 发送成功完成信号
                await result_queue.put({"status": "complete"})

            except Exception as e:
                logger.error(f"流式任务失败: {description} - {str(e)}", exc_info=True)
                # 发送错误信号，包含异常对象
                await result_queue.put({"status": "error", "exception": e})

        # 创建任务对象
        task = Task(
            task_type=task_type,
            func=streaming_worker,
            description=description,
            priority=0,  # 统一优先级
        )
        task.future = asyncio.Future()

        # 加入队列
        await self._queue.put(task)
        logger.info(
            f"流式任务已加入队列: {description} (队列长度: {self._queue.qsize()})"
        )

        # 等待任务开始执行
        await started.wait()

        # 返回结果生成器
        async def result_generator():
            while True:
                item = await result_queue.get()

                # 检查是否为控制信号
                if isinstance(item, dict):
                    status = item.get("status")

                    # 完成信号 - 正常结束
                    if status == "complete":
                        break

                    # 错误信号 - 立即抛出异常
                    elif status == "error":
                        exception = item.get("exception")
                        if exception:
                            raise exception
                        else:
                            raise RuntimeError(
                                "Streaming task failed with unknown error"
                            )

                # 正常数据项 - yield出去
                yield item

        return result_generator()

    def submit_sync(
        self,
        task_type: TaskType,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        priority: int = 0,
        description: str = "",
    ) -> asyncio.Task:
        """
        同步提交任务（不等待完成），返回 asyncio.Task

        用于需要在非异步上下文中提交任务的场景

        Args:
            task_type: 任务类型
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            priority: 优先级
            description: 任务描述

        Returns:
            asyncio.Task: 可以 await 的任务对象

        Raises:
            RuntimeError: 在没有运行事件循环的同步上下文中调用
        """
        # 自动启动任务引擎
        if not self._running:
            try:
                loop = asyncio.get_running_loop()
                # 事件循环正在运行，创建启动任务
                asyncio.create_task(self.start())
            except RuntimeError:
                # 没有运行的事件循环，创建新循环并启动
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.start())

        # 创建异步任务
        async def _submit_and_wait():
            return await self.submit(
                task_type=task_type,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                description=description,
            )

        return asyncio.create_task(_submit_and_wait())

    @property
    def current_task(self) -> Optional[Task]:
        """获取当前正在执行的任务"""
        return self._current_task

    @property
    def is_running(self) -> bool:
        """任务引擎是否正在运行"""
        return self._running

    @property
    def queue_size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()

    @property
    def is_busy(self) -> bool:
        return self._current_task is not None

    def get_loaded_model_id(self) -> Optional[str]:
        return self._loaded_model_id

    @property
    def loaded_model_id(self) -> Optional[str]:
        return self._loaded_model_id

    async def wait_until_idle(self):
        await self._queue.join()

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "current_task": str(self._current_task) if self._current_task else None,
            "is_busy": self.is_busy,
            "task_count": self._task_count,
            "loaded_model_id": self._loaded_model_id,
        }


# 全局任务引擎实例
_task_engine: Optional[TaskEngine] = None
_engine_lock = threading.Lock()


def get_task_engine() -> TaskEngine:
    """获取全局任务引擎实例（单例）"""
    global _task_engine

    with _engine_lock:
        if _task_engine is None:
            _task_engine = TaskEngine()

        return _task_engine
