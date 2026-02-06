"""
TTS引擎代理基类

提供统一的任务队列集成，防止并发模型访问冲突
"""

import logging
from typing import AsyncGenerator, Tuple, Optional, Callable
import numpy as np

from core.task_engine import get_task_engine, TaskType

logger = logging.getLogger(__name__)


class BaseEngineProxy:
    """
    TTS引擎代理基类

    所有TTS操作都通过任务引擎执行，确保并发安全。
    子类通过实现_log方法来定制日志行为。
    """

    def __init__(self, engine_getter: Callable, task_engine=None):
        """
        初始化代理基类

        Args:
            engine_getter: 获取原始TTS引擎的函数
            task_engine: 任务引擎实例（可选，默认使用全局单例）
        """
        self._engine_getter = engine_getter
        self._task_engine = task_engine or get_task_engine()

    def _get_engine(self):
        """获取原始TTS引擎实例（带 None 检查）"""
        engine = self._engine_getter()
        if engine is None:
            raise RuntimeError(
                "TTS 引擎不可用 - 可能未初始化或已卸载。"
                "请确保模型已正确加载。"
            )
        return engine

    def _log(self, message: str):
        """
        记录日志（子类应覆盖此方法）

        Args:
            message: 日志消息
        """
        logger.info(message)

    # ========== 非流式合成方法 ==========

    async def custom_voice_synthesize_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Custom Voice 异步合成

        Args:
            text: 要合成的文本
            speaker: 说话人名称
            language: 语言
            instruct: 指令文本
            **kwargs: 其他参数

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        self._log(f"提交 Custom Voice 任务: {text[:30]}...")

        engine = self._get_engine()
        return await self._task_engine.submit(
            task_type=TaskType.GENERATE,
            func=engine.custom_voice_synthesize_async,
            args=(text, speaker, language, instruct),
            kwargs=kwargs,
            description=f"Custom Voice: {text[:30]}",
            priority=5
        )

    async def voice_design_synthesize_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Voice Design 异步合成

        Args:
            text: 要合成的文本
            design_prompt: 设计提示词
            language: 语言
            **kwargs: 其他参数

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        self._log(f"提交 Voice Design 任务: {text[:30]}...")

        engine = self._get_engine()
        return await self._task_engine.submit(
            task_type=TaskType.GENERATE,
            func=engine.voice_design_synthesize_async,
            args=(text, design_prompt, language),
            kwargs=kwargs,
            description=f"Voice Design: {text[:30]}",
            priority=5
        )

    async def voice_clone_synthesize_async(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Voice Clone 异步合成

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
        self._log(f"提交 Voice Clone 任务: {text[:30]}...")

        engine = self._get_engine()
        return await self._task_engine.submit(
            task_type=TaskType.GENERATE,
            func=engine.voice_clone_synthesize_async,
            args=(text, ref_audio, ref_text, clone_prompt, x_vector_only),
            kwargs=kwargs,
            description=f"Voice Clone: {text[:30]}",
            priority=5
        )

    async def create_voice_clone_prompt_async(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False
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
        self._log("提交创建 Voice Clone Prompt 任务")

        engine = self._get_engine()
        return await self._task_engine.submit(
            task_type=TaskType.GENERATE,
            func=engine.create_voice_clone_prompt_async,
            args=(ref_audio, ref_text, x_vector_only),
            description="创建 Voice Clone Prompt",
            priority=5
        )

    # ========== 流式合成方法 ==========

    async def custom_voice_synthesize_streaming_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,
        **kwargs
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Custom Voice 流式合成

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
        self._log(f"提交流式 Custom Voice 任务: {text[:30]}...")

        engine = self._get_engine()
        stream_gen = await self._task_engine.submit_streaming(
            task_type=TaskType.GENERATE,
            func=engine.custom_voice_synthesize_streaming_async,
            args=(text, speaker, language, instruct),
            kwargs={
                "emit_every_frames": emit_every_frames,
                "decode_window_frames": decode_window_frames,
                "overlap_samples": overlap_samples,
                **kwargs
            },
            description=f"流式 Custom Voice: {text[:30]}"
        )

        async for chunk, sr in stream_gen:
            yield chunk, sr

    async def voice_design_synthesize_streaming_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,
        **kwargs
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Voice Design 流式合成

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
        self._log(f"提交流式 Voice Design 任务: {text[:30]}...")

        engine = self._get_engine()
        stream_gen = await self._task_engine.submit_streaming(
            task_type=TaskType.GENERATE,
            func=engine.voice_design_synthesize_streaming_async,
            args=(text, design_prompt, language),
            kwargs={
                "emit_every_frames": emit_every_frames,
                "decode_window_frames": decode_window_frames,
                "overlap_samples": overlap_samples,
                **kwargs
            },
            description=f"流式 Voice Design: {text[:30]}"
        )

        async for chunk, sr in stream_gen:
            yield chunk, sr

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
        **kwargs
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Voice Clone 流式合成

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
        self._log(f"提交流式 Voice Clone 任务: {text[:30]}...")

        engine = self._get_engine()
        stream_gen = await self._task_engine.submit_streaming(
            task_type=TaskType.GENERATE,
            func=engine.voice_clone_synthesize_streaming_async,
            args=(text, ref_audio, ref_text, clone_prompt, x_vector_only),
            kwargs={
                "emit_every_frames": emit_every_frames,
                "decode_window_frames": decode_window_frames,
                "overlap_samples": overlap_samples,
                **kwargs
            },
            description=f"流式 Voice Clone: {text[:30]}"
        )

        async for chunk, sr in stream_gen:
            yield chunk, sr

    # ========== 引擎管理方法 ==========

    async def unload_async(self):
        """
        异步卸载引擎（通过任务引擎）

        卸载操作会排队等待当前任务完成后执行
        """
        self._log("提交卸载引擎任务")

        engine = self._get_engine()
        await self._task_engine.submit(
            task_type=TaskType.UNLOAD,
            func=engine.unload,
            description="卸载 TTS 引擎"
        )

    def unload(self):
        """
        同步卸载（直接调用引擎）

        注意：此方法不通过任务队列，谨慎使用
        """
        engine = self._get_engine()
        engine.unload()

    # ========== 辅助方法 ==========

    def get_supported_speakers(self) -> list:
        """获取支持的说话人列表"""
        engine = self._get_engine()
        return engine.get_supported_speakers() if engine else []

    def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        engine = self._get_engine()
        return engine.get_supported_languages() if engine else []

    # ========== 属性代理 ==========

    @property
    def enable_streaming(self):
        """是否启用流式输出"""
        engine = self._get_engine()
        return engine.enable_streaming if engine else False

    @property
    def model(self):
        """获取当前模型"""
        engine = self._get_engine()
        return engine.model if engine else None

    @property
    def device(self):
        """获取当前设备"""
        engine = self._get_engine()
        return engine.device if engine else None

    @property
    def model_type(self):
        """获取当前模型类型"""
        engine = self._get_engine()
        return engine.model_type if engine else None
