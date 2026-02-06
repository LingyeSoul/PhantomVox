"""
API引擎代理

为FastAPI路由提供统一的引擎访问接口，自动集成任务引擎
"""

import logging

from core.engine_proxy_base import BaseEngineProxy

logger = logging.getLogger(__name__)


class APIEngineProxy(BaseEngineProxy):
    """
    API引擎代理 - 确保所有调用都通过任务引擎

    继承BaseEngineProxy，使用Python logging记录日志
    """

    def __init__(self, engine_getter):
        """
        初始化代理

        Args:
            engine_getter: 获取原始引擎的函数
        """
        super().__init__(engine_getter)

    def _log(self, message: str):
        """
        记录日志到Python logger

        Args:
            message: 日志消息
        """
        logger.info(f"[API任务队列] {message}")
