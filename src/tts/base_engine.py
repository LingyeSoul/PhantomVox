"""
TTS引擎基类

定义所有TTS引擎必须实现的抽象接口
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, AsyncGenerator

import numpy as np

from .capabilities import EngineCapabilities


class BaseTTSEngine(ABC):
    """
    TTS引擎抽象基类

    定义所有TTS引擎必须实现的接口，包括：
    - 生命周期管理（加载/卸载模型）
    - 非流式合成
    - 流式合成
    - 批量流式合成
    - 提示词管理
    - 能力查询
    """

    # 类属性：引擎标识符
    engine_id: str = ""
    engine_name: str = ""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass

    @property
    @abstractmethod
    def device(self) -> object:
        """获取引擎运行的设备"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> object:
        """获取模型类型"""
        pass

    # ========== 生命周期方法 ==========

    @abstractmethod
    def load_model(self, force_reload: bool = False) -> None:
        """
        加载TTS模型

        Args:
            force_reload: 是否强制重新加载
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """卸载TTS模型，释放资源"""
        pass

    # ========== 非流式合成方法 ==========

    @abstractmethod
    async def custom_voice_synthesize_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Custom Voice 非流式异步合成

        Args:
            text: 要合成的文本
            speaker: 说话人名称
            language: 语言
            instruct: 指令文本
            **kwargs: 其他参数

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        pass

    @abstractmethod
    async def voice_design_synthesize_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Voice Design 非流式异步合成

        Args:
            text: 要合成的文本
            design_prompt: 设计提示词
            language: 语言
            **kwargs: 其他参数

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        pass

    @abstractmethod
    async def voice_clone_synthesize_async(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Voice Clone 非流式异步合成

        Args:
            text: 要合成的文本
            ref_audio: 参考音频文件路径
            ref_text: 参考文本
            clone_prompt: 克隆提示词
            x_vector_only: 是否仅使用x-vector
            **kwargs: 其他参数

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        pass

    # ========== 流式合成方法 ==========

    @abstractmethod
    async def custom_voice_synthesize_streaming_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Custom Voice 流式异步合成

        Args:
            text: 要合成的文本
            speaker: 说话人名称
            language: 语言
            instruct: 指令文本
            emit_every_frames: 每隔多少帧发射一次
            decode_window_frames: 解码窗口帧数
            overlap_samples: 重叠样本数
            **kwargs: 其他参数

        Yields:
            Tuple[np.ndarray, int]: (音频数据块, 采样率)
        """
        pass

    @abstractmethod
    async def voice_design_synthesize_streaming_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Voice Design 流式异步合成

        Args:
            text: 要合成的文本
            design_prompt: 设计提示词
            language: 语言
            emit_every_frames: 每隔多少帧发射一次
            decode_window_frames: 解码窗口帧数
            overlap_samples: 重叠样本数
            **kwargs: 其他参数

        Yields:
            Tuple[np.ndarray, int]: (音频数据块, 采样率)
        """
        pass

    @abstractmethod
    async def voice_clone_synthesize_streaming_async(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Voice Clone 流式异步合成

        Args:
            text: 要合成的文本
            ref_audio: 参考音频文件路径
            ref_text: 参考文本
            clone_prompt: 克隆提示词
            x_vector_only: 是否仅使用x-vector
            emit_every_frames: 每隔多少帧发射一次
            decode_window_frames: 解码窗口帧数
            overlap_samples: 重叠样本数
            **kwargs: 其他参数

        Yields:
            Tuple[np.ndarray, int]: (音频数据块, 采样率)
        """
        pass

    # ========== 批量流式合成方法 ==========

    @abstractmethod
    async def custom_voice_batch_stream_synthesize_async(
        self,
        texts: list,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
        **kwargs,
    ) -> AsyncGenerator[Tuple[list, int], None]:
        """
        Custom Voice 批量流式异步合成

        Args:
            texts: 要合成的文本列表
            speaker: 说话人名称
            language: 语言
            instruct: 指令文本
            emit_every_frames: 每隔多少帧发射一次
            decode_window_frames: 解码窗口帧数
            first_chunk_emit_every: 首块发射间隔
            first_chunk_decode_window: 首块解码窗口
            first_chunk_frames: 首块帧数
            **kwargs: 其他参数

        Yields:
            Tuple[list, int]: (每个文本的音频块列表, 采样率)
        """
        pass

    @abstractmethod
    async def voice_design_batch_stream_synthesize_async(
        self,
        texts: list,
        design_prompt: str,
        language: str = "Chinese",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
        **kwargs,
    ) -> AsyncGenerator[Tuple[list, int], None]:
        """
        Voice Design 批量流式异步合成

        Args:
            texts: 要合成的文本列表
            design_prompt: 设计提示词
            language: 语言
            emit_every_frames: 每隔多少帧发射一次
            decode_window_frames: 解码窗口帧数
            first_chunk_emit_every: 首块发射间隔
            first_chunk_decode_window: 首块解码窗口
            first_chunk_frames: 首块帧数
            **kwargs: 其他参数

        Yields:
            Tuple[list, int]: (每个文本的音频块列表, 采样率)
        """
        pass

    @abstractmethod
    async def voice_clone_batch_stream_synthesize_async(
        self,
        texts: list,
        clone_prompt,
        language: str = "Auto",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
        **kwargs,
    ) -> AsyncGenerator[Tuple[list, int], None]:
        """
        Voice Clone 批量流式异步合成

        Args:
            texts: 要合成的文本列表
            clone_prompt: 克隆提示词
            language: 语言
            emit_every_frames: 每隔多少帧发射一次
            decode_window_frames: 解码窗口帧数
            first_chunk_emit_every: 首块发射间隔
            first_chunk_decode_window: 首块解码窗口
            first_chunk_frames: 首块帧数
            **kwargs: 其他参数

        Yields:
            Tuple[list, int]: (每个文本的音频块列表, 采样率)
        """
        pass

    # ========== 提示词管理方法 ==========

    @abstractmethod
    async def create_voice_clone_prompt_async(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False,
    ):
        """
        创建声音克隆提示词

        Args:
            ref_audio: 参考音频文件路径
            ref_text: 参考文本
            x_vector_only: 是否仅使用x-vector

        Returns:
            提示词结果
        """
        pass

    # ========== 能力查询方法 ==========

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """
        获取引擎能力

        Returns:
            EngineCapabilities: 引擎能力对象
        """
        pass

    @abstractmethod
    def get_supported_speakers(self) -> list:
        """
        获取支持的说话人列表

         Returns:
             list: 说话人名称列表
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list:
        """
        获取支持的语言列表

        Returns:
            list: 语言代码列表
        """
        pass

    # ========== 优化控制方法 ==========

    def ensure_optimization_mode(self, mode: str) -> None:
        """
        确保引擎处于指定优化模式

        默认实现为空操作，某些引擎可能不需要此功能

        Args:
            mode: 优化模式
        """
        pass
