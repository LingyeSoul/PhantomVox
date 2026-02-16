"""
FastAPI 依赖注入配置

使用 FastAPI 的依赖注入系统管理共享资源
"""

from fastapi import Depends, HTTPException, status
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ========== 全局依赖存储 ==========

_tts_engine_getter: Optional[Callable] = None
_voice_library = None
_log_callback: Optional[Callable] = None
_api_engine_proxy = None

# 服务配置
_service_config: Dict[str, Any] = {
    "mode": "customvoice",
    "model_id": None,
    "speaker": "Vivian",
    "preset": None,
    "clone_id": None,
    "clone_prompt": None,
}


# ========== 服务配置管理 ==========


def update_service_config(
    mode: Optional[str] = None,
    model_id: Optional[str] = None,
    speaker: Optional[str] = None,
    preset: Optional[str] = None,
    clone_id: Optional[str] = None,
    clone_prompt: Any = None,
):
    global _service_config
    if mode is not None:
        _service_config["mode"] = mode
    if model_id is not None:
        _service_config["model_id"] = model_id
    if speaker is not None:
        _service_config["speaker"] = speaker
    if preset is not None:
        _service_config["preset"] = preset
    if clone_id is not None:
        _service_config["clone_id"] = clone_id
    if clone_prompt is not None:
        _service_config["clone_prompt"] = clone_prompt
    logger.info(
        f"服务配置已更新: mode={_service_config['mode']}, model={_service_config['model_id']}"
    )


def get_service_config() -> Dict[str, Any]:
    """获取当前服务配置"""
    return _service_config.copy()


def get_service_mode() -> str:
    """获取当前服务模式"""
    return _service_config.get("mode", "customvoice")


def get_service_speaker() -> str:
    """获取当前服务说话人"""
    return _service_config.get("speaker", "Vivian")


def get_service_preset() -> Optional[str]:
    """获取当前服务预设"""
    return _service_config.get("preset")


def get_service_clone_prompt() -> Any:
    """获取当前服务克隆提示词"""
    return _service_config.get("clone_prompt")


# ========== 依赖管理函数 ==========


def initialize_dependencies(
    tts_engine_getter: Callable,
    voice_library=None,
    log_callback: Optional[Callable] = None,
):
    """
    初始化全局依赖

    在 lifespan 启动阶段调用

    Args:
        tts_engine_getter: TTS 引擎获取函数
        voice_library: VoiceLibrary 实例
        log_callback: 日志回调函数
    """
    global _tts_engine_getter, _voice_library, _log_callback, _api_engine_proxy
    _tts_engine_getter = tts_engine_getter
    _voice_library = voice_library
    _log_callback = log_callback

    # 创建API引擎代理
    from api.engine_proxy import APIEngineProxy

    _api_engine_proxy = APIEngineProxy(tts_engine_getter)

    logger.info("FastAPI dependencies initialized with task engine integration")


def cleanup_dependencies():
    """清理全局依赖"""
    global _tts_engine_getter, _voice_library, _log_callback, _api_engine_proxy
    _tts_engine_getter = None
    _voice_library = None
    _log_callback = None
    _api_engine_proxy = None
    logger.info("FastAPI dependencies cleaned up")


# ========== FastAPI 依赖函数 ==========


async def get_tts_engine():
    """
    获取 TTS 引擎实例（返回代理）

    Returns:
        APIEngineProxy: TTS 引擎代理实例

    Raises:
        HTTPException: 当引擎未初始化时
    """
    if _api_engine_proxy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS engine not initialized",
        )

    return _api_engine_proxy


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


def log_message(message: str, level: str = "info"):
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
    if level_upper == "INFO":
        logger.info(message)
    elif level_upper == "ERROR":
        logger.error(message)
    elif level_upper == "WARNING":
        logger.warning(message)
    elif level_upper == "SUCCESS":
        logger.info(f"✓ {message}")
    else:
        logger.debug(message)
