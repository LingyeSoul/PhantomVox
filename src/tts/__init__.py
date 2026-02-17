"""
PhantomVox TTS 模块

提供基于 Qwen3-TTS 的文本转语音功能

子模块：
- qwen_engine: 主引擎封装
- audio_loader: 音频加载功能
- model_loader: 模型加载和管理
- prompt_manager: Voice Clone Prompt 管理
- audio_manager: 音频管理
- exceptions: 异常定义
"""

from .qwen_engine import QwenEngine
from .audio_manager import AudioManager

__all__ = ["QwenEngine", "AudioManager"]
