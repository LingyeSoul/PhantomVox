"""
Pydantic 数据模型定义

FastAPI 请求/响应数据验证模型
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List

from api.constants import DEFAULT_TTS_TIMEOUT, MAX_TEXT_LENGTH


# ========== 基础响应模型 ==========


class TTSSuccessResponse(BaseModel):
    """TTS 成功响应基类"""

    success: bool = True


class TTSErrorResponse(BaseModel):
    """TTS 错误响应模型"""

    success: bool = False
    error: str


# ========== TTS 请求/响应模型 ==========


class TTSRequest(BaseModel):
    """TTS 合成请求模型（支持三种模式）"""

    text: str = Field(..., min_length=1, description="要合成的文本")
    mode: Literal["custom_voice", "voice_design", "voice_clone"] = Field(
        default="custom_voice", description="TTS 模式"
    )

    # Custom Voice 参数
    speaker: str = Field(default="Vivian", description="说话人名称")
    language: str = Field(default="Chinese", description="语言")
    instruct: str = Field(default="", description="情感指令")

    # Voice Design 参数
    design_prompt: str = Field(default="", description="声音设计描述")

    # Voice Clone 参数（使用已保存的克隆音色）
    clone_id: Optional[str] = Field(default=None, description="克隆音色 ID")
    clone_name: Optional[str] = Field(default=None, description="克隆音色名称")

    # 通用可选参数
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="语速因子")
    pitch_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="音高因子")
    timeout: Optional[int] = Field(
        default=DEFAULT_TTS_TIMEOUT, ge=10, le=3600, description="超时时间(秒)"
    )
    max_length: Optional[int] = Field(
        default=MAX_TEXT_LENGTH, ge=10, le=MAX_TEXT_LENGTH, description="最大文本长度"
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """验证文本不为空"""
        if not v or not v.strip():
            raise ValueError("文本不能为空")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode_parameters(cls, v: str, info) -> str:
        """验证模式对应的必需参数"""
        values = info.data if hasattr(info, "data") else {}

        if v == "voice_design":
            if not values.get("design_prompt"):
                raise ValueError("voice_design 模式需要 design_prompt 参数")
        elif v == "voice_clone":
            if not values.get("clone_id") and not values.get("clone_name"):
                raise ValueError("voice_clone 模式需要 clone_id 或 clone_name 参数")
        return v


class TTSResponse(TTSSuccessResponse):
    """TTS 合成响应"""

    audio: str = Field(..., description="音频数据（base64编码）")
    format: str = Field(..., description="音频格式")
    sample_rate: int = Field(..., description="采样率")
    duration: float = Field(..., description="音频时长（秒）")


# ========== 元数据响应模型 ==========


class SpeakersResponse(TTSSuccessResponse):
    """说话人列表响应"""

    speakers: List[str] = Field(default_factory=list, description="支持的说话人列表")


class LanguagesResponse(TTSSuccessResponse):
    """语言列表响应"""

    languages: dict = Field(default_factory=dict, description="支持的语言字典")


class ClonesResponse(TTSSuccessResponse):
    """克隆音色列表响应"""

    clones: List[dict] = Field(default_factory=list, description="保存的克隆音色列表")


class DesignPresetsResponse(TTSSuccessResponse):
    """设计预设列表响应"""

    presets: dict = Field(default_factory=dict, description="语音设计预设字典")


# ========== 状态响应模型 ==========


class StatusResponse(TTSSuccessResponse):
    """服务状态响应"""

    host: str = Field(..., description="监听地址")
    port: int = Field(..., description="监听端口")
    running: bool = Field(..., description="是否运行中")
    # 模型状态
    loaded_model_id: Optional[str] = Field(
        default=None, description="当前加载的模型ID"
    )
    is_busy: bool = Field(default=False, description="任务引擎是否繁忙")
    queue_size: int = Field(default=0, description="任务队列长度")
    # 请求统计
    total_requests: int = Field(default=0, description="总请求数")
    successful_requests: int = Field(default=0, description="成功请求数")
    failed_requests: int = Field(default=0, description="失败请求数")
    recent_requests: List[dict] = Field(
        default_factory=list, description="最近请求记录"
    )


# ========== OpenAI 兼容模型 ==========


class OpenAITTSRequest(BaseModel):
    """OpenAI TTS API 兼容请求模型"""

    model: str = Field(default="tts-1", description="模型名称")
    input: str = Field(..., min_length=1, description="要转换的文本")
    voice: str = Field(default="alloy", description="说话人")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="pcm",  # 改为默认使用 PCM，更适合流式播放
        description="音频格式",
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速")

    @field_validator("input")
    @classmethod
    def input_must_not_be_empty(cls, v: str) -> str:
        """验证输入不为空"""
        if not v or not v.strip():
            raise ValueError("input 不能为空")
        return v
