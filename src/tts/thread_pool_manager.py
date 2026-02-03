"""
TTS线程池单例管理器

使用ThreadPoolExecutor管理TTS引擎的后台执行线程
使用单例模式确保全局只有一个线程池实例
"""

import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)


class TTSThreadPoolManager:
    """TTS线程池单例管理器

    使用单例模式确保全局只有一个线程池实例
    max_workers=1 确保模型调用的线程安全
    """

    _instance: Optional['TTSThreadPoolManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化线程池管理器"""
        if hasattr(self, '_initialized'):
            return

        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._initialized = True

    def get_executor(self) -> ThreadPoolExecutor:
        """获取或创建线程池执行器（线程安全）

        Returns:
            ThreadPoolExecutor: 线程池执行器实例
        """
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:  # 双重检查锁定
                    logger.info("创建TTS线程池执行器 (max_workers=1)")
                    self._executor = ThreadPoolExecutor(
                        max_workers=1,  # 单工作线程确保模型线程安全
                        thread_name_prefix="qwen_tts_worker"
                    )
        return self._executor

    def shutdown(self, wait: bool = True):
        """关闭线程池

        Args:
            wait: 是否等待所有任务完成
        """
        if self._executor is not None:
            logger.info("关闭TTS线程池执行器")
            self._executor.shutdown(wait=wait)
            self._executor = None

    def is_running(self) -> bool:
        """检查线程池是否正在运行

        Returns:
            bool: 线程池是否正在运行
        """
        return self._executor is not None
