"""
PhantomVox UI 基础组件

提供语音生成页面的共享基类和组件
"""

from ui.components.base.text_panel import TextPanel
from ui.components.base.audio_control_panel import AudioControlPanel
from ui.components.base.base_voice_view import BaseVoiceView

__all__ = [
    "TextPanel",
    "AudioControlPanel",
    "BaseVoiceView",
]
