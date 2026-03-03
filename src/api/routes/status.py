"""
状态查询路由
"""

from fastapi import APIRouter
from api.models import StatusResponse
from core.task_engine import get_task_engine

router = APIRouter()


class RequestStatistics:
    """请求统计管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._total_requests = 0
            cls._instance._successful_requests = 0
            cls._instance._failed_requests = 0
            cls._instance._request_log = []
        return cls._instance

    def record_request(self, success: bool):
        """记录请求统计"""
        import time
        self._total_requests += 1
        if success:
            self._successful_requests += 1
            status_icon = "✓"
        else:
            self._failed_requests += 1
            status_icon = "✗"

        self._request_log.append({
            "time": time.strftime("%H:%M:%S"),
            "status": status_icon,
            "success": success
        })

        # 只保留最近100条
        if len(self._request_log) > 100:
            self._request_log.pop(0)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "recent_requests": self._request_log[-20:]  # 最近20条
        }

    def reset(self):
        """重置统计信息"""
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._request_log = []


# 全局统计实例
_stats = RequestStatistics()


@router.get("/status", response_model=StatusResponse)
async def get_status(host: str = "0.0.0.0", port: int = 8848, running: bool = True):
    """
    服务状态查询端点

    返回服务器统计信息和运行状态
    """
    stats = _stats.get_stats()
    # 获取任务引擎状态
    task_engine = get_task_engine()
    engine_status = task_engine.get_status()
    
    return {
        "success": True,
        "host": host,
        "port": port,
        "running": running,
        # 模型状态
        "loaded_model_id": engine_status.get("loaded_model_id"),
        "is_busy": engine_status.get("is_busy", False),
        "queue_size": engine_status.get("queue_size", 0),
        # 请求统计
        **stats
    }


def get_stats():
    """获取统计实例的依赖"""
    return _stats
