"""
Qwen3-TTS 引擎封装（重构版）

提供文本转语音的核心功能，支持三种模式：
1. Custom Voice - 使用预设说话人 + 情感指令
2. Voice Design - 通过自然语言描述设计声音
3. Voice Clone - 使用参考音频克隆声音

所有三种模式都支持流式输出（基于修改版qwen-tts的 stream_generate_pcm API）
"""

import logging
import numpy as np
import asyncio
from typing import Tuple, Optional, AsyncGenerator, Generator, List
import torch

from tts.base_engine import BaseTTSEngine
from tts.capabilities import EngineCapabilities, SynthesisMode, FeatureFlag
from tts.exceptions import (
    TTSError,
    TTSModelNotLoadedError,
    TTSInvalidParameterError,
)

# 导入新模块
from tts.engines.qwen.audio_loader import apply_librosa_patch
from tts.engines.qwen.model_loader import ModelLoader
from tts.engines.qwen.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class QwenEngine(BaseTTSEngine):
    """Qwen3-TTS 引擎（使用 ModelLoader 和 PromptManager）"""

    MODEL_CUSTOM_VOICE = "CustomVoice"
    MODEL_VOICE_DESIGN = "VoiceDesign"
    MODEL_BASE = "Base"

    # 类属性：引擎标识符
    engine_id: str = "qwen"
    engine_name: str = "Qwen3-TTS"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        attn_implementation: Optional[str] = None,
        shared_tokenizer_path: Optional[str] = None,
        enable_streaming: bool = True,
        streaming_decode_window: int = 80,
        lazy_load: bool = False,
        smart_vram_enabled: bool = True,
        delay_cleanup_seconds: int = 60,
    ):
        self.model_path = model_path
        self._model_type = model_type
        self._device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.shared_tokenizer_path = shared_tokenizer_path
        self.enable_streaming = enable_streaming
        self.streaming_decode_window = streaming_decode_window
        self.smart_vram_enabled = smart_vram_enabled
        self.delay_cleanup_seconds = delay_cleanup_seconds

        self.model_loader = ModelLoader(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            shared_tokenizer_path=shared_tokenizer_path,
            enable_streaming=enable_streaming,
            streaming_decode_window=streaming_decode_window,
            smart_vram_enabled=smart_vram_enabled,
            delay_cleanup_seconds=delay_cleanup_seconds,
        )
        self.model = None
        self.prompt_manager = None

        if not lazy_load:
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def device(self):
        """获取引擎运行的设备"""
        return self._device

    @property
    def model_type(self):
        """获取模型类型"""
        return self._model_type

    @property
    def _current_optimization_mode(self):
        return self.model_loader._current_optimization_mode

    @property
    def _optimization_lock(self):
        return self.model_loader._optimization_lock

    def _load_model(self):
        """加载 Qwen3-TTS 模型"""
        apply_librosa_patch()
        self.model_loader.load()
        self.model = self.model_loader.model
        self.prompt_manager = PromptManager(self.model, self.device)

    def load_model(self, force_reload=False):
        """手动加载模型（用于延迟加载模式）"""
        if self.model is not None and not force_reload:
            raise RuntimeError("模型已加载，请勿重复加载")
        self._load_model()

    def _apply_optimizations(self, mode: str):
        """应用指定模式的优化配置"""
        self.model_loader._apply_optimizations(mode)

    async def _ensure_optimization_mode(self, mode: str):
        """确保模型使用指定的优化模式"""
        await self.model_loader.ensure_optimization_mode(mode)

    # ========== BaseTTSEngine 实现 ==========

    def get_capabilities(self) -> EngineCapabilities:
        """
        获取引擎能力

        Returns:
            EngineCapabilities: 引擎能力对象
        """
        return EngineCapabilities(
            modes={
                SynthesisMode.CUSTOM_VOICE,
                SynthesisMode.VOICE_DESIGN,
                SynthesisMode.VOICE_CLONE,
            },
            features={
                FeatureFlag.STREAMING,
                FeatureFlag.BATCH_STREAMING,
                FeatureFlag.VOICE_CLONE_PROMPT_EXTRACTION,
                FeatureFlag.SYNC_SYNTHESIS,
                FeatureFlag.ASYNC_SYNTHESIS,
            },
            supported_languages=["Chinese", "English", "Japanese", "Korean", "Auto"],
            supported_speakers=[
                "Vivian",
                "Serena",
                "Uncle_Fu",
                "Dylan",
                "Eric",
                "Ryan",
                "Aiden",
                "Ono_Anna",
                "Sohee",
            ],
            supported_sample_rates=[24000],
            supported_output_formats=["wav"],
        )

    # ========== Custom Voice 模式 ==========

    def custom_voice_synthesize(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Custom Voice 同步生成"""
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Custom Voice): {text[:50]}...")

            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                **kwargs,
            )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    async def custom_voice_synthesize_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Custom Voice 异步生成（非流式优化）"""
        # 确保使用非流式优化
        await self._ensure_optimization_mode("non_streaming")

        return await asyncio.to_thread(
            self.custom_voice_synthesize,
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            **kwargs,
        )

    async def custom_voice_synthesize_streaming_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,  # 10ms @ 24kHz = 240 samples
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Custom Voice 流式生成（流式优化）

        使用统一的 stream_generate_pcm API
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 确保使用流式优化
        await self._ensure_optimization_mode("streaming")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        logger.info(f"[Streaming] 正在流式生成语音 (Custom Voice): {text[:50]}...")

        # 准备 tokens - 使用正确的方法
        assistant_text = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        input = self.model.processor(
            text=assistant_text, return_tensors="pt", padding=True
        )
        input_ids = [input["input_ids"].to(self.model.device)]
        if input_ids[0].dim() == 1:
            input_ids[0] = input_ids[0].unsqueeze(0)

        instruct_ids = None
        if instruct and instruct.strip():
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_input = self.model.processor(
                text=instruct_text, return_tensors="pt", padding=True
            )
            instruct_id = instruct_input["input_ids"].to(self.model.device)
            if instruct_id.dim() == 1:
                instruct_id = instruct_id.unsqueeze(0)
            instruct_ids = [instruct_id]

        def stream_generator() -> Generator[Tuple[np.ndarray, int], None, None]:
            """同步生成器"""
            for chunk, sr in self.model.model.stream_generate_pcm(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                speakers=[speaker],
                languages=[language],
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                overlap_samples=overlap_samples,
            ):
                # 过滤空音频块
                if len(chunk) > 0:
                    yield chunk, sr

        # 真正的流式：每个 chunk 生成后立即 yield
        gen = stream_generator()
        loop = asyncio.get_event_loop()

        def get_next_chunk(gen):
            """Get next chunk from generator (moved outside loop for performance)"""
            try:
                return next(gen), False  # (chunk, is_stop)
            except StopIteration:
                return None, True  # (None, is_stop)

        while True:
            chunk_info, is_stop = await loop.run_in_executor(
                None, lambda: get_next_chunk(gen)
            )
            if is_stop:
                break
            yield chunk_info

        logger.info("✓ [Streaming] 流式生成完成")

    async def custom_voice_batch_stream_synthesize_async(
        self,
        texts: List[str],
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
        **kwargs,
    ) -> AsyncGenerator[Tuple[List[np.ndarray], int], None]:
        """
        Custom Voice 批量流式生成

        使用底层 batch_stream_generate_pcm 方法，
        实现多个文本的并行流式生成。所有文本共享 KV cache，同步推进。

        Args:
            texts: 文本列表
            speaker: 说话人（广播到所有文本）
            language: 语言（广播到所有文本）
            instruct: 情感指令（广播到所有文本，可选）
            emit_every_frames: 音频块发射间隔（帧）
            decode_window_frames: 解码窗口大小（帧）
            first_chunk_emit_every: 首块发射间隔
            first_chunk_decode_window: 首块解码窗口
            first_chunk_frames: 首块帧数
            **kwargs: 其他参数

        Yields:
            Tuple[List[np.ndarray], int]: (每个文本的音频块列表, 采样率)
                - 每个 yield 返回所有文本当前生成的音频块
                - 已完成的文本返回空数组
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 确保使用流式优化
        await self._ensure_optimization_mode("streaming")

        if not texts:
            return

        # 验证文本
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise TTSInvalidParameterError(f"文本 {i + 1} 为空")

        logger.info(
            f"[Batch Streaming] 正在批量流式生成 {len(texts)} 个文本 (Custom Voice)..."
        )

        # 为每个文本构建 input_ids（批量处理优化）
        assistant_texts = [
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
            for text in texts
        ]
        batch_input = self.model.processor(
            text=assistant_texts, return_tensors="pt", padding=True
        )
        batch_ids = batch_input["input_ids"].to(self.model.device)
        # Split batch into individual tensors
        input_ids = []
        for i in range(len(texts)):
            ids = (
                batch_ids[i : i + 1] if batch_ids.dim() > 1 else batch_ids.unsqueeze(0)
            )
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            input_ids.append(ids)

        # 构建可选的 instruct_ids
        instruct_ids = None
        if instruct and instruct.strip():
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_inp = self.model.processor(
                text=instruct_text, return_tensors="pt", padding=True
            )
            instruct_id = instruct_inp["input_ids"].to(self.model.device)
            if instruct_id.dim() == 1:
                instruct_id = instruct_id.unsqueeze(0)
            instruct_ids = [instruct_id] * len(texts)

        def batch_stream_generator() -> Generator[
            Tuple[List[np.ndarray], int], None, None
        ]:
            """同步批量流式生成器"""
            with torch.no_grad():  # 禁用梯度计算，减少显存占用
                for chunks_list, sr in self.model.model.batch_stream_generate_pcm(
                    input_ids=input_ids,
                    instruct_ids=instruct_ids,
                    speakers=[speaker],
                    languages=[language] * len(texts),
                    voice_clone_prompt=None,
                    emit_every_frames=emit_every_frames,
                    decode_window_frames=decode_window_frames,
                    first_chunk_emit_every=first_chunk_emit_every,
                    first_chunk_decode_window=first_chunk_decode_window,
                    first_chunk_frames=first_chunk_frames,
                ):
                    # 过滤全空的块
                    has_content = any(chunk.size > 0 for chunk in chunks_list)
                    if has_content:
                        yield chunks_list, sr

        # 异步迭代
        gen = batch_stream_generator()
        loop = asyncio.get_event_loop()

        def get_next_batch(gen):
            """Get next batch from generator (moved outside loop for performance)"""
            try:
                return next(gen), False
            except StopIteration:
                return None, True

        while True:
            batch_info, is_stop = await loop.run_in_executor(
                None, lambda: get_next_batch(gen)
            )
            if is_stop:
                break
            yield batch_info

        logger.info("✓ [Batch Streaming] Custom Voice 批量流式生成完成")

    # ========== Voice Design 模式 ==========

    def voice_design_synthesize(
        self, text: str, design_prompt: str, language: str = "Chinese", **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Voice Design 同步生成"""
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Voice Design): {text[:50]}...")

            wavs, sr = self.model.generate_voice_design(
                text=text, language=language, instruct=design_prompt, **kwargs
            )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    async def voice_design_synthesize_async(
        self, text: str, design_prompt: str, language: str = "Chinese", **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Voice Design 异步生成（非流式优化）"""
        # 确保使用非流式优化
        await self._ensure_optimization_mode("non_streaming")

        return await asyncio.to_thread(
            self.voice_design_synthesize,
            text=text,
            design_prompt=design_prompt,
            language=language,
            **kwargs,
        )

    async def voice_design_synthesize_streaming_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,  # 10ms @ 24kHz = 240 samples
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Voice Design 流式生成（流式优化）

        使用统一的 stream_generate_pcm API
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 确保使用流式优化
        await self._ensure_optimization_mode("streaming")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        logger.info(f"[Streaming] 正在流式生成语音 (Voice Design): {text[:50]}...")

        # 准备 tokens - 使用正确的方法
        assistant_text = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        input = self.model.processor(
            text=assistant_text, return_tensors="pt", padding=True
        )
        input_ids = [input["input_ids"].to(self.model.device)]
        if input_ids[0].dim() == 1:
            input_ids[0] = input_ids[0].unsqueeze(0)

        instruct_text = f"<|im_start|>user\n{design_prompt}<|im_end|>\n"
        instruct_input = self.model.processor(
            text=instruct_text, return_tensors="pt", padding=True
        )
        instruct_ids = [instruct_input["input_ids"].to(self.model.device)]
        if instruct_ids[0].dim() == 1:
            instruct_ids[0] = instruct_ids[0].unsqueeze(0)

        def stream_generator() -> Generator[Tuple[np.ndarray, int], None, None]:
            """同步生成器"""
            for chunk, sr in self.model.model.stream_generate_pcm(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                languages=[language],
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                overlap_samples=overlap_samples,
            ):
                # 过滤空音频块
                if len(chunk) > 0:
                    yield chunk, sr

        # 真正的流式：每个 chunk 生成后立即 yield
        gen = stream_generator()
        loop = asyncio.get_event_loop()

        while True:
            # 使用自定义函数包装 next，避免 StopIteration 在 Future 中的问题
            def get_next_chunk():
                try:
                    return next(gen), False  # (chunk, is_stop)
                except StopIteration:
                    return None, True  # (None, is_stop)

            chunk_info, is_stop = await loop.run_in_executor(None, get_next_chunk)
            if is_stop:
                break
            yield chunk_info

        logger.info("✓ [Streaming] 流式生成完成")

    async def voice_design_batch_stream_synthesize_async(
        self,
        texts: List[str],
        design_prompt: str,
        language: str = "Chinese",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
        **kwargs,
    ) -> AsyncGenerator[Tuple[List[np.ndarray], int], None]:
        """
        Voice Design 批量流式生成

        使用底层 batch_stream_generate_pcm 方法，
        实现多个文本的并行流式生成。所有文本共享 KV cache，同步推进。

        Args:
            texts: 文本列表
            design_prompt: 声音设计描述（作为 instruct，广播到所有文本）
            language: 语言（广播到所有文本）
            emit_every_frames: 音频块发射间隔（帧）
            decode_window_frames: 解码窗口大小（帧）
            first_chunk_emit_every: 首块发射间隔
            first_chunk_decode_window: 首块解码窗口
            first_chunk_frames: 首块帧数
            **kwargs: 其他参数

        Yields:
            Tuple[List[np.ndarray], int]: (每个文本的音频块列表, 采样率)
                - 每个 yield 返回所有文本当前生成的音频块
                - 已完成的文本返回空数组
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 确保使用流式优化
        await self._ensure_optimization_mode("streaming")

        if not texts:
            return

        # 验证文本
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise TTSInvalidParameterError(f"文本 {i + 1} 为空")

        logger.info(
            f"[Batch Streaming] 正在批量流式生成 {len(texts)} 个文本 (Voice Design)..."
        )

        # 为每个文本构建 input_ids
        input_ids = []
        for text in texts:
            assistant_text = (
                f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
            )
            inp = self.model.processor(
                text=assistant_text, return_tensors="pt", padding=True
            )
            ids = inp["input_ids"].to(self.model.device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            input_ids.append(ids)

        # 从 design_prompt 构建 instruct_ids
        instruct_text = f"<|im_start|>user\n{design_prompt}<|im_end|>\n"
        instruct_inp = self.model.processor(
            text=instruct_text, return_tensors="pt", padding=True
        )
        instruct_id = instruct_inp["input_ids"].to(self.model.device)
        if instruct_id.dim() == 1:
            instruct_id = instruct_id.unsqueeze(0)
        instruct_ids = [instruct_id] * len(texts)

        def batch_stream_generator() -> Generator[
            Tuple[List[np.ndarray], int], None, None
        ]:
            """同步批量流式生成器"""
            with torch.no_grad():  # 禁用梯度计算，减少显存占用
                for chunks_list, sr in self.model.model.batch_stream_generate_pcm(
                    input_ids=input_ids,
                    instruct_ids=instruct_ids,
                    speakers=None,  # Voice Design 不使用 speakers
                    languages=[language] * len(texts),
                    voice_clone_prompt=None,
                    emit_every_frames=emit_every_frames,
                    decode_window_frames=decode_window_frames,
                    first_chunk_emit_every=first_chunk_emit_every,
                    first_chunk_decode_window=first_chunk_decode_window,
                    first_chunk_frames=first_chunk_frames,
                ):
                    # 过滤全空的块
                    has_content = any(chunk.size > 0 for chunk in chunks_list)
                    if has_content:
                        yield chunks_list, sr

        # 异步迭代
        gen = batch_stream_generator()
        loop = asyncio.get_event_loop()

        while True:

            def get_next_batch():
                try:
                    return next(gen), False
                except StopIteration:
                    return None, True

            batch_info, is_stop = await loop.run_in_executor(None, get_next_batch)
            if is_stop:
                break
            yield batch_info

        logger.info("✓ [Batch Streaming] Voice Design 批量流式生成完成")

    # ========== Voice Clone 模式 ==========

    def _convert_prompt_to_prompt_items(self, clone_prompt):
        """将各种格式的 clone_prompt 转换为 VoiceClonePromptItem 对象列表"""
        return self.prompt_manager.convert_prompt_to_prompt_items(clone_prompt)

    def voice_clone_synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Voice Clone 同步生成

        Args:
            text: 要合成的文本
            ref_audio: 参考音频路径（使用 clone_prompt 时可选）
            ref_text: 参考文本（使用 clone_prompt 时可选）
            clone_prompt: 预计算的 prompt 特征（可选）
            x_vector_only: 是否仅使用 x_vector 模式
            **kwargs: 其他生成参数

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        # 验证参数：如果不使用 clone_prompt，必须提供 ref_audio 和 ref_text
        if not clone_prompt and (not ref_audio or not ref_text):
            raise TTSInvalidParameterError(
                "不使用 clone_prompt 时，必须提供 ref_audio 和 ref_text"
            )

        try:
            logger.info(f"正在生成语音 (Voice Clone): {text[:50]}...")

            if clone_prompt:
                # 转换为 VoiceClonePromptItem 对象列表（模型期望的格式）
                prompt_items = self._convert_prompt_to_prompt_items(clone_prompt)

                # 使用 voice_clone_prompt 时，不需要 ref_audio, ref_text 等参数
                # 过滤掉不应该传递给模型的参数
                excluded_params = [
                    "timeout",
                    "ref_audio",
                    "ref_text",
                    "x_vector_only_mode",
                    "x_vector_only",
                ]
                model_kwargs = {
                    k: v for k, v in kwargs.items() if k not in excluded_params
                }

                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=prompt_items,  # ← 传递列表而不是字典
                    **model_kwargs,
                )
            else:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                    **kwargs,
                )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    async def voice_clone_synthesize_async(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Voice Clone 异步生成（非流式优化）"""
        # 确保使用非流式优化
        await self._ensure_optimization_mode("non_streaming")

        return await asyncio.to_thread(
            self.voice_clone_synthesize,
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            clone_prompt=clone_prompt,
            x_vector_only=x_vector_only,
            **kwargs,
        )

    async def voice_clone_synthesize_streaming_async(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 0,
        **kwargs,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Voice Clone 流式生成（流式优化）

        使用统一的 stream_generate_pcm API
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 确保使用流式优化
        await self._ensure_optimization_mode("streaming")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        # 验证参数：如果不使用 clone_prompt，必须提供 ref_audio 和 ref_text
        if not clone_prompt and (not ref_audio or not ref_text):
            raise TTSInvalidParameterError(
                "不使用 clone_prompt 时，必须提供 ref_audio 和 ref_text"
            )

        logger.info(f"[Streaming] 正在流式生成语音 (Voice Clone): {text[:50]}...")

        # 创建或使用已有的 voice_clone_prompt
        if clone_prompt is None:
            clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only
            )

        # 转换为 VoiceClonePromptItem 对象列表（确保格式正确）
        prompt_items = self._convert_prompt_to_prompt_items(clone_prompt)

        # 准备 tokens - 使用正确的方法
        assistant_text = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        input = self.model.processor(
            text=assistant_text, return_tensors="pt", padding=True
        )
        input_ids = [input["input_ids"].to(self.model.device)]
        if input_ids[0].dim() == 1:
            input_ids[0] = input_ids[0].unsqueeze(0)

        # 使用模型方法转换为 dict 格式（包含所有必要信息）
        prompt_dict = self.model._prompt_items_to_voice_clone_prompt(prompt_items)

        # 构建 ref_ids（从 prompt_items 中提取 ref_text）
        ref_ids = None
        if prompt_items[0].ref_text and not prompt_items[0].x_vector_only_mode:
            ref_text_formatted = (
                f"<|im_start|>assistant\n{prompt_items[0].ref_text}<|im_end|>\n"
            )
            ref_input = self.model.processor(
                text=ref_text_formatted, return_tensors="pt", padding=True
            )
            ref_id = ref_input["input_ids"].to(self.model.device)
            if ref_id.dim() == 1:
                ref_id = ref_id.unsqueeze(0)
            ref_ids = [ref_id]

        def stream_generator() -> Generator[Tuple[np.ndarray, int], None, None]:
            """同步生成器"""
            for chunk, sr in self.model.model.stream_generate_pcm(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=prompt_dict,
                languages=["Auto"],
                emit_every_frames=emit_every_frames,
                decode_window_frames=decode_window_frames,
                overlap_samples=overlap_samples,
            ):
                # 过滤空音频块
                if len(chunk) > 0:
                    yield chunk, sr

        # 真正的流式：每个 chunk 生成后立即 yield
        gen = stream_generator()
        loop = asyncio.get_event_loop()

        while True:
            # 使用自定义函数包装 next，避免 StopIteration 在 Future 中的问题
            def get_next_chunk():
                try:
                    return next(gen), False  # (chunk, is_stop)
                except StopIteration:
                    return None, True  # (None, is_stop)

            chunk_info, is_stop = await loop.run_in_executor(None, get_next_chunk)
            if is_stop:
                break
            yield chunk_info

        logger.info("✓ [Streaming] 流式生成完成")

    async def voice_clone_batch_stream_synthesize_async(
        self,
        texts: List[str],
        clone_prompt,
        language: str = "Auto",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        first_chunk_emit_every: int = 5,
        first_chunk_decode_window: int = 48,
        first_chunk_frames: int = 48,
        **kwargs,
    ) -> AsyncGenerator[Tuple[List[np.ndarray], int], None]:
        """
        Voice Clone 批量流式生成

        使用 qwen-tts 的原生 batch_stream_generate_voice_clone 方法，
        实现多个文本的并行流式生成。所有文本共享 KV cache，同步推进。

        Args:
            texts: 文本列表
            clone_prompt: VoiceClonePromptItem 或 List[VoiceClonePromptItem]
            language: 语言（广播到所有文本）
            emit_every_frames: 音频块发射间隔（帧）
            decode_window_frames: 解码窗口大小（帧）
            first_chunk_emit_every: 首块发射间隔
            first_chunk_decode_window: 首块解码窗口
            first_chunk_frames: 首块帧数
            **kwargs: 其他参数

        Yields:
            Tuple[List[np.ndarray], int]: (每个文本的音频块列表, 采样率)
                - 每个 yield 返回所有文本当前生成的音频块
                - 已完成的文本返回空数组
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 确保使用流式优化
        await self._ensure_optimization_mode("streaming")

        if not texts:
            return

        # 验证文本
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise TTSInvalidParameterError(f"文本 {i + 1} 为空")

        logger.info(f"[Batch Streaming] 正在批量流式生成 {len(texts)} 个文本...")

        # 转换 clone_prompt 为 VoiceClonePromptItem 列表
        prompt_items = self._convert_prompt_to_prompt_items(clone_prompt)

        # 确保模型支持批量流式（必须是 Base 模型）
        if hasattr(self.model, "model") and hasattr(self.model.model, "tts_model_type"):
            if self.model.model.tts_model_type != "base":
                raise TTSError(
                    f"批量流式推理仅支持 Base 模型，当前模型类型: {self.model.model.tts_model_type}"
                )

        def batch_stream_generator() -> Generator[
            Tuple[List[np.ndarray], int], None, None
        ]:
            """同步批量流式生成器"""
            with torch.no_grad():  # 禁用梯度计算，减少显存占用
                for chunks_list, sr in self.model.batch_stream_generate_voice_clone(
                    text=texts,
                    language=language,
                    voice_clone_prompt=prompt_items[0]
                    if len(prompt_items) == 1
                    else prompt_items,
                    emit_every_frames=emit_every_frames,
                    decode_window_frames=decode_window_frames,
                    first_chunk_emit_every=first_chunk_emit_every,
                    first_chunk_decode_window=first_chunk_decode_window,
                    first_chunk_frames=first_chunk_frames,
                ):
                    # 过滤全空的块
                    has_content = any(chunk.size > 0 for chunk in chunks_list)
                    if has_content:
                        yield chunks_list, sr

        # 异步迭代
        gen = batch_stream_generator()
        loop = asyncio.get_event_loop()

        while True:

            def get_next_batch():
                try:
                    return next(gen), False
                except StopIteration:
                    return None, True

            batch_info, is_stop = await loop.run_in_executor(None, get_next_batch)
            if is_stop:
                break
            yield batch_info

        logger.info("✓ [Batch Streaming] 批量流式生成完成")

    def create_voice_clone_prompt(
        self, ref_audio: str, ref_text: str, x_vector_only: bool = False
    ):
        """创建可重用的声音克隆 prompt"""
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        try:
            logger.info("正在提取声音特征...")

            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only
            )

            logger.info("✓ 声音特征提取完成")
            return prompt_items

        except Exception as e:
            logger.error(f"✗ 特征提取失败: {str(e)}")
            raise

    async def create_voice_clone_prompt_async(
        self, ref_audio: str, ref_text: str, x_vector_only: bool = False
    ):
        """创建可重用的声音克隆 prompt（异步版本）"""
        return await asyncio.to_thread(
            self.create_voice_clone_prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
        )

    def create_and_save_prompt_features(
        self, ref_audio: str, ref_text: str, save_path: str, x_vector_only: bool = False
    ) -> bool:
        """
        创建并保存 prompt 特征到文件

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            save_path: 保存路径（.pt 文件）
            x_vector_only: 是否仅使用 x_vector

        Returns:
            bool: 是否成功
        """
        try:
            # 创建特征
            prompt_item = self.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only=x_vector_only
            )

            # 保存特征
            from tts.prompt_serializer import save_prompt_features

            metadata = {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only": x_vector_only,
            }

            return save_prompt_features(prompt_item, save_path, metadata)

        except Exception as e:
            logger.error(f"创建并保存特征失败: {e}")
            return False

    def unload(self):
        """卸载模型并释放资源"""
        # 根据 smart_vram 配置选择卸载方式
        if self.smart_vram_enabled:
            # 智能显存模式：调用异步方法（需要事件循环）
            # 检查是否有事件循环
            try:
                import asyncio

                asyncio.get_running_loop()
                # 在同步上下文中调用异步方法
                asyncio.create_task(self.model_loader.unload_async())
                logger.info("已启动智能显存卸载任务")
            except RuntimeError:
                # 没有运行中的事件循环，使用同步方式
                logger.warning("无法获取事件循环，使用同步卸载")
                self.model_loader.unload()
        else:
            # 非智能模式：直接同步清理
            self.model_loader.unload()
        self.model = None
        self.prompt_manager = None

    # ========== 辅助方法 ==========

    def get_supported_speakers(self) -> list:
        """获取支持的说话人列表"""
        if self.model:
            try:
                return self.model.get_supported_speakers()
            except Exception as e:
                logger.warning(f"获取说话人列表失败: {e}")
        return [
            "Vivian",
            "Serena",
            "Uncle_Fu",
            "Dylan",
            "Eric",
            "Ryan",
            "Aiden",
            "Ono_Anna",
            "Sohee",
        ]

    def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        if self.model:
            try:
                return self.model.get_supported_languages()
            except Exception as e:
                logger.warning(f"获取语言列表失败: {e}")
        return ["Chinese", "English", "Japanese", "Korean", "Auto"]

    # ========== 兼容旧 API ==========

    def synthesize(self, text, voice="default", speed=1.0, pitch=1.0):
        """基础合成方法 (向后兼容)"""
        speaker_map = {"default": "Vivian", "female": "Serena", "male": "Uncle_Fu"}
        speaker = speaker_map.get(voice, "Vivian")

        audio_data, _ = self.custom_voice_synthesize(
            text=text,
            speaker=speaker,
            language="Chinese",
            speed_factor=speed,
            pitch_factor=pitch,
        )
        return audio_data

    def get_available_voices(self):
        """获取可用的声音列表 (向后兼容)"""
        return ["default", "female", "male", "child", "elderly"]
