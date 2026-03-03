"""
SRT批量推理配置模型

定义三种TTS模式的配置数据结构
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CustomVoiceConfig:
    """Custom Voice 模式配置"""

    speaker: str = "Vivian"
    language: str = "Chinese"
    instruct: str = ""  # 情感指令（可选）

    # 模型参数
    speed_factor: float = 1.0
    pitch_factor: float = 1.0

    def to_generation_kwargs(self) -> dict:
        """转换为生成参数字典"""
        return {
            "speaker": self.speaker,
            "language": self.language,
            "instruct": self.instruct if self.instruct else None,
            "speed_factor": self.speed_factor,
            "pitch_factor": self.pitch_factor,
        }


@dataclass
class VoiceDesignConfig:
    """Voice Design 模式配置"""

    design_prompt: str = ""  # 声音设计描述（必填）
    language: str = "Chinese"

    # 模型参数
    speed_factor: float = 1.0
    pitch_factor: float = 1.0

    def __post_init__(self):
        if not self.design_prompt or not self.design_prompt.strip():
            raise ValueError("Voice Design模式需要提供design_prompt")

    def to_generation_kwargs(self) -> dict:
        """转换为生成参数字典"""
        return {
            "design_prompt": self.design_prompt,
            "language": self.language,
            "speed_factor": self.speed_factor,
            "pitch_factor": self.pitch_factor,
        }


@dataclass
class VoiceCloneConfig:
    """Voice Clone 模式配置"""

    mode: str = "new"  # "new" | "saved"

    # 新音频模式
    ref_audio_path: str = ""
    ref_text: str = ""
    x_vector_only: bool = False

    # 已保存克隆模式
    clone_id: str = ""
    clone_prompt: Any = None  # VoiceClonePromptItem

    def __post_init__(self):
        if self.mode not in ("new", "saved"):
            raise ValueError(f"不支持的克隆模式: {self.mode}")

    def validate(self) -> tuple[bool, str]:
        """验证配置有效性"""
        if self.mode == "new":
            if not self.ref_audio_path:
                return False, "新音频模式需要提供参考音频路径"
            if not self.ref_text:
                return False, "新音频模式需要提供参考文本"
        else:  # saved
            if not self.clone_id:
                return False, "已保存克隆模式需要选择克隆"
            if self.clone_prompt is None:
                return False, "已保存克隆模式需要提供clone_prompt"

        return True, ""

    def to_generation_kwargs(self) -> dict:
        """转换为生成参数字典"""
        if self.mode == "new":
            return {
                "ref_audio": self.ref_audio_path,
                "ref_text": self.ref_text,
                "x_vector_only": self.x_vector_only,
            }
        else:
            return {
                "clone_prompt": self.clone_prompt,
            }


# 配置类型联合
SRTConfig = CustomVoiceConfig | VoiceDesignConfig | VoiceCloneConfig
