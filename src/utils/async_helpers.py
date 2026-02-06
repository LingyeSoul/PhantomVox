"""
异步任务辅助工具

提供安全的异步任务创建和错误处理工具函数
"""

import asyncio
import logging
from typing import Callable, Optional, Any
import traceback

logger = logging.getLogger(__name__)


async def _handle_task_error(
    task: asyncio.Task,
    task_name: str,
    on_error: Optional[Callable[[Exception], None]] = None
):
    """
    处理异步任务的错误

    Args:
        task: 要等待的异步任务
        task_name: 任务名称（用于日志）
        on_error: 错误回调函数（可选）
    """
    try:
        await task
    except Exception as e:
        logger.error(f"[Task Error] {task_name} failed: {str(e)}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")

        # 调用错误回调
        if on_error:
            try:
                on_error(e)
            except Exception as callback_error:
                logger.error(f"[Task Error] Error callback failed: {callback_error}")


def create_task_with_error_handling(
    coro,
    task_name: str = "async_task",
    on_error: Optional[Callable[[Exception], None]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> asyncio.Task:
    """
    创建带错误处理的异步任务

    这个函数解决了 fire-and-forget 任务中错误被静默忽略的问题。
    当任务失败时，错误会被记录到日志，并调用错误回调函数。

    Args:
        coro: 协程对象或协程函数
        task_name: 任务名称（用于日志记录）
        on_error: 错误回调函数，接收异常对象作为参数
        loop: 事件循环（可选，默认使用当前事件循环）

    Returns:
        asyncio.Task: 创建的任务对象

    Example:
        >>> async def my_task():
        ...     raise ValueError("Something went wrong")
        >>>
        >>> def handle_error(e):
        ...     print(f"Task failed: {e}")
        >>>
        >>> # 创建任务，错误会被捕获并处理
        >>> task = create_task_with_error_handling(
        ...     my_task(),
        ...     task_name="MyTask",
        ...     on_error=handle_error
        ... )
    """
    # 获取事件循环
    if loop is None:
        loop = asyncio.get_event_loop()

    # 创建错误处理协程
    error_handler = _handle_task_error(coro, task_name, on_error)

    # 创建任务
    task = loop.create_task(error_handler)

    logger.debug(f"[Task Created] {task_name} (task_id: {id(task)})")

    return task


def create_task_with_retry(
    coro_func: Callable,
    max_retries: int = 3,
    task_name: str = "retry_task",
    on_error: Optional[Callable[[Exception, int], None]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> asyncio.Task:
    """
    创建带重试机制的异步任务

    Args:
        coro_func: 返回协程的函数（可以多次调用）
        max_retries: 最大重试次数
        task_name: 任务名称
        on_error: 错误回调函数，接收(异常, 当前重试次数)
        loop: 事件循环（可选）

    Returns:
        asyncio.Task: 创建的任务对象

    Example:
        >>> async def unstable_task():
        ...     if random.random() < 0.7:  # 70%失败率
        ...         raise ConnectionError("Network error")
        ...     return "success"
        >>>
        >>> # 创建带重试的任务
        >>> task = create_task_with_retry(
        ...     unstable_task,
        ...     max_retries=5,
        ...     task_name="UnstableTask"
        ... )
    """
    async def retry_wrapper():
        last_exception = None

        for attempt in range(max_retries):
            try:
                result = await coro_func()
                logger.info(f"[Retry Success] {task_name} succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Retry Failed] {task_name} attempt {attempt + 1}/{max_retries} failed: {e}"
                )

                # 调用错误回调
                if on_error:
                    try:
                        on_error(e, attempt + 1)
                    except Exception as callback_error:
                        logger.error(f"[Retry Error] Callback failed: {callback_error}")

                # 最后一次尝试失败，不再重试
                if attempt == max_retries - 1:
                    break

                # 等待一小段时间再重试（指数退避）
                wait_time = 0.1 * (2 ** attempt)
                await asyncio.sleep(wait_time)

        # 所有重试都失败了
        logger.error(f"[Retry Failed] {task_name} failed after {max_retries} attempts")
        if last_exception:
            raise last_exception

    # 获取事件循环
    if loop is None:
        loop = asyncio.get_event_loop()

    # 创建任务
    task = loop.create_task(retry_wrapper())

    return task


def run_async_synchronous(
    coro,
    timeout: Optional[float] = None
) -> Any:
    """
    在同步上下文中运行异步函数

    Args:
        coro: 协程对象
        timeout: 超时时间（秒）

    Returns:
        协程的返回值

    Raises:
        TimeoutError: 如果任务超时
        Exception: 如果任务失败
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # 如果事件循环正在运行，使用create_task
        import concurrent.futures
        import threading

        result_future = concurrent.futures.Future()

        def run_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
                result_future.set_result(result)
            except Exception as e:
                result_future.set_exception(e)
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=timeout or 30)

        if thread.is_alive():
            raise TimeoutError("Async operation timed out")

        return result_future.result()
    else:
        # 如果事件循环未运行，直接运行
        if timeout:
            coro = asyncio.wait_for(coro, timeout=timeout)
        return loop.run_until_complete(coro)
