"""
状态查询路由
"""

from fastapi import APIRouter, Depends
from api.models import StatusResponse

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
    return {
        "success": True,
        "host": host,
        "port": port,
        "running": running,
        **stats
    }


def get_stats():
    """获取统计实例的依赖"""
    return _stats
