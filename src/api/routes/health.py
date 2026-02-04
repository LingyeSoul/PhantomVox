"""
健康检查路由
"""

from fastapi import APIRouter
from api.models import TTSSuccessResponse

router = APIRouter()


@router.get("/health", response_model=TTSSuccessResponse)
async def health_check():
    """
    健康检查端点

    返回服务状态，用于服务可用性检查
    """
    return {
        "success": True,
        "status": "ok",
        "service": "PhantomVox TTS Service"
    }
