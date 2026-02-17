"""克隆音色查找工具"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)


def find_clone(
    voice_library, clone_id: Optional[str] = None, clone_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    从 VoiceLibrary 查找克隆音色

    Args:
        voice_library: VoiceLibrary 实例
        clone_id: 克隆音色 ID
        clone_name: 克隆音色名称

    Returns:
        克隆音色数据字典

    Raises:
        HTTPException: 音色未找到或库不可用
    """
    if voice_library is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice library not available",
        )

    clone = None

    # 优先按名称查找
    if clone_name:
        for c in voice_library.get_all_clones():
            if c["name"] == clone_name:
                clone = c
                break

        # 如果按名称找到多个或未找到，且提供了 clone_id，则使用 clone_id
        matching_by_name = [
            c for c in voice_library.get_all_clones() if c["name"] == clone_name
        ]
        if clone is None or len(matching_by_name) > 1:
            if clone_id:
                clone = voice_library.get_clone(clone_id)

    elif clone_id:
        clone = voice_library.get_clone(clone_id)

    if not clone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"克隆音色未找到：clone_id={clone_id}, clone_name={clone_name}",
        )

    return clone
