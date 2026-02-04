"""
FastAPI 依赖注入配置

使用 FastAPI 的依赖注入系统管理共享资源
"""

from fastapi import Depends, HTTPException, status
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


# ========== 全局依赖存储 ==========

_tts_engine_getter: Optional[Callable] = None
_voice_library = None
_log_callback: Optional[Callable] = None


# ========== 依赖管理函数 ==========

def initialize_dependencies(
    tts_engine_getter: Callable,
    voice_library=None,
    log_callback: Optional[Callable] = None
):
    """
    初始化全局依赖

    在 lifespan 启动阶段调用

    Args:
        tts_engine_getter: TTS 引擎获取函数
        voice_library: VoiceLibrary 实例
        log_callback: 日志回调函数
    """
    global _tts_engine_getter, _voice_library, _log_callback
    _tts_engine_getter = tts_engine_getter
    _voice_library = voice_library
    _log_callback = log_callback
    logger.info("FastAPI dependencies initialized")


def cleanup_dependencies():
    """清理全局依赖"""
    global _tts_engine_getter, _voice_library, _log_callback
    _tts_engine_getter = None
    _voice_library = None
    _log_callback = None
    logger.info("FastAPI dependencies cleaned up")


# ========== FastAPI 依赖函数 ==========

async def get_tts_engine():
    """
    获取 TTS 引擎实例（依赖注入）

    Returns:
        QwenEngine: TTS 引擎实例

    Raises:
        HTTPException: 当引擎未初始化时
    """
    if _tts_engine_getter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS engine not initialized"
        )

    engine = _tts_engine_getter()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS engine not available"
        )

    return engine


def get_voice_library():
    """
    获取 VoiceLibrary 实例（依赖注入）

    Returns:
        VoiceLibrary: 声音库管理器实例
    """
    return _voice_library


def get_log_callback():
    """
    获取日志回调函数（依赖注入）

    Returns:
        Callable: 日志回调函数
    """
    return _log_callback


# ========== 辅助函数 ==========

def log_message(message: str, level: str = 'info'):
    """
    记录日志的辅助函数

    Args:
        message: 日志消息
        level: 日志级别 (info, error, warning, success)
    """
    if _log_callback:
        try:
            _log_callback(message, level)
        except Exception:
            pass

    # 同时记录到标准日志
    level_upper = level.upper()
    if level_upper == 'INFO':
        logger.info(message)
    elif level_upper == 'ERROR':
        logger.error(message)
    elif level_upper == 'WARNING':
        logger.warning(message)
    elif level_upper == 'SUCCESS':
        logger.info(f"✓ {message}")
    else:
        logger.debug(message)
