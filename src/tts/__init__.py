"""
PhantomVox TTS 模块

提供基于 Qwen3-TTS 的文本转语音功能
"""

from .qwen_engine import QwenEngine
from .audio_manager import AudioManager

__all__ = ['QwenEngine', 'AudioManager']
