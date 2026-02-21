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
- srt_parser: SRT字幕解析
- srt_batch_engine: SRT批量推理引擎
"""

from .qwen_engine import QwenEngine
from .audio_manager import AudioManager
from .srt_parser import SRTParser, SRTEntry, ScheduledEntry
from .srt_batch_engine import SRTBatchEngine, SRTBatchResult
from .srt_config_models import CustomVoiceConfig, VoiceDesignConfig, VoiceCloneConfig
from .timeline_scheduler import TimelineScheduler
from .audio_assembler import AudioAssembler

__all__ = [
    "QwenEngine",
    "AudioManager",
    "SRTParser",
    "SRTEntry",
    "ScheduledEntry",
    "SRTBatchEngine",
    "SRTBatchResult",
    "CustomVoiceConfig",
    "VoiceDesignConfig",
    "VoiceCloneConfig",
    "TimelineScheduler",
    "AudioAssembler",
]
