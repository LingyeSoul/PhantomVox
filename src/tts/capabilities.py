"""
TTS引擎能力声明

提供TTS引擎能力描述的声明系统，支持查询引擎支持的合成模式、功能特性和参数范围
"""

from enum import Enum, Flag, auto
from dataclasses import dataclass, field
from typing import List, Set


class SynthesisMode(Enum):
    """合成模式枚举"""

    CUSTOM_VOICE = "CustomVoice"
    VOICE_DESIGN = "VoiceDesign"
    VOICE_CLONE = "VoiceClone"


class FeatureFlag(Flag):
    """功能特性标志（支持位运算组合）"""

    STREAMING = auto()
    BATCH_STREAMING = auto()
    VOICE_CLONE_PROMPT_EXTRACTION = auto()
    SYNC_SYNTHESIS = auto()
    ASYNC_SYNTHESIS = auto()


@dataclass
class ParameterCapabilities:
    """参数能力"""

    supported_languages: List[str] = field(default_factory=list)
    supported_speakers: List[str] = field(default_factory=list)
    supported_sample_rates: List[int] = field(default_factory=list)
    supported_output_formats: List[str] = field(default_factory=list)


@dataclass
class EngineCapabilities:
    """引擎能力聚合"""

    modes: Set[SynthesisMode] = field(default_factory=set)
    features: Set[FeatureFlag] = field(default_factory=set)
    supported_languages: List[str] = field(default_factory=list)
    supported_speakers: List[str] = field(default_factory=list)
    supported_sample_rates: List[int] = field(default_factory=list)
    supported_output_formats: List[str] = field(default_factory=list)

    def supports_mode(self, mode: SynthesisMode) -> bool:
        """检查是否支持指定合成模式"""
        return mode in self.modes

    def supports_feature(self, feature: FeatureFlag) -> bool:
        """检查是否支持指定功能特性"""
        return feature in self.features

    def supports_language(self, language: str) -> bool:
        """检查是否支持指定语言"""
        if not self.supported_languages:
            return False
        return language in self.supported_languages

    def supports_sample_rate(self, rate: int) -> bool:
        """检查是否支持指定采样率"""
        if not self.supported_sample_rates:
            return False
        return rate in self.supported_sample_rates
