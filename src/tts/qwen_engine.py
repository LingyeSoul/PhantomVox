"""
Qwen3-TTS 引擎封装

提供文本转语音的核心功能，支持三种模式：
1. Custom Voice - 使用预设说话人 + 情感指令
2. Voice Design - 通过自然语言描述设计声音
3. Voice Clone - 使用参考音频克隆声音

支持流式输出（通过 monkey patch qwen-tts）
"""

from qwen_tts import Qwen3TTSModel
import logging
import os
import numpy as np
import asyncio
import threading
from typing import Tuple, Optional, AsyncGenerator, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .thread_pool_manager import TTSThreadPoolManager
from .exceptions import (
    TTSError,
    TTSModelNotLoadedError,
    TTSInvalidParameterError,
    TTSTimeoutError,
    TTSSynthesisError
)
from .apply_streaming_patch import apply_streaming_patch_to_qwen_tts

logger = logging.getLogger(__name__)


# ========== 共享 Tokenizer 支持 ==========

_SHARED_TOKENIZER_DIR = None  # 共享 tokenizer 路径


def _patch_tokenizer_loading():
    """
    Patch transformers 的 cached_file 和 cached_files 以支持共享 tokenizer。

    关键：直接在 transformers.utils.hub 模块上 patch，确保所有后续导入都能获取 patched 版本
    """
    from transformers.utils import hub as transformers_hub
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    # 保存原始函数
    original_cached_file = transformers_hub.cached_file
    original_cached_files = transformers_hub.cached_files

    logger.info(f"[PATCH] 开始应用共享 tokenizer patch，共享目录: {_SHARED_TOKENIZER_DIR}")

    # ========== Patch cached_files（必须先 patch，因为 cached_file 内部调用它）==========
    def patched_cached_files(path_or_repo_id, filenames, **kwargs):
        global _SHARED_TOKENIZER_DIR

        # 检查是否请求 speech_tokenizer 相关文件
        if _SHARED_TOKENIZER_DIR and filenames:
            first_filename = filenames[0] if filenames else ""

            if first_filename.startswith("speech_tokenizer/"):
                logger.info(f"[PATCH] cached_files 拦截到请求: {filenames}")
                # 重定向到共享 tokenizer 目录
                new_filenames = [
                    os.path.join(_SHARED_TOKENIZER_DIR, f.replace("speech_tokenizer/", ""))
                    for f in filenames
                ]

                existing_files = [f for f in new_filenames if os.path.exists(f)]
                if existing_files:
                    logger.info(f"[PATCH] ✓ 返回共享文件: {existing_files}")
                    return existing_files
                else:
                    logger.warning(f"[PATCH] ✗ 共享目录中找不到文件: {new_filenames}")

        # 其他情况使用原始逻辑
        return original_cached_files(path_or_repo_id, filenames, **kwargs)

    # 直接替换模块属性
    transformers_hub.cached_files = patched_cached_files

    # ========== Patch cached_file ==========
    def patched_cached_file(pretrained_model_name_or_path, filename, *args, **kwargs):
        global _SHARED_TOKENIZER_DIR

        # 拦截 speech_tokenizer 相关请求
        if _SHARED_TOKENIZER_DIR and filename.startswith("speech_tokenizer/"):
            logger.info(f"[PATCH] cached_file 拦截到请求: {filename}")
            # 重定向到共享目录
            new_path = os.path.join(_SHARED_TOKENIZER_DIR, filename.replace("speech_tokenizer/", ""))
            if os.path.exists(new_path):
                logger.info(f"[PATCH] ✓ 返回共享文件: {new_path}")
                return new_path
            else:
                logger.warning(f"[PATCH] ✗ 共享目录中找不到文件: {new_path}")

        return original_cached_file(pretrained_model_name_or_path, filename, *args, **kwargs)

    # 直接替换模块属性
    transformers_hub.cached_file = patched_cached_file

    # ========== Patch Qwen3TTSTokenizer.from_pretrained ==========
    original_from_pretrained = Qwen3TTSTokenizer.from_pretrained.__func__

    def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        global _SHARED_TOKENIZER_DIR

        # 如果路径包含 speech_tokenizer，重定向到共享目录
        if _SHARED_TOKENIZER_DIR and "speech_tokenizer" in str(pretrained_model_name_or_path):
            logger.info(f"[PATCH] Qwen3TTSTokenizer 拦截到请求: {pretrained_model_name_or_path}")
            return original_from_pretrained(cls, _SHARED_TOKENIZER_DIR, **kwargs)

        return original_from_pretrained(cls, pretrained_model_name_or_path, **kwargs)

    Qwen3TTSTokenizer.from_pretrained = classmethod(patched_from_pretrained)

    # ========== Patch qwen-tts 模块中的引用 ==========
    # 因为 qwen-tts 在模块导入时就已经 cached_file 的引用
    # 我们需要直接替换模块命名空间中的引用
    try:
        from qwen_tts.core.models import modeling_qwen3_tts
        modeling_qwen3_tts.cached_file = patched_cached_file
        logger.info("[PATCH] ✓ 已更新 qwen-tts 模块中的 cached_file 引用")
    except Exception as e:
        logger.warning(f"[PATCH] ⚠ 无法更新 qwen-tts 模块引用: {e}")

    logger.info("[PATCH] ✓ 所有 Patch 应用完成")


# =========================================


class QwenEngine:
    """Qwen3-TTS 引擎封装"""

    # 模型类型常量
    MODEL_CUSTOM_VOICE = "CustomVoice"
    MODEL_VOICE_DESIGN = "VoiceDesign"
    MODEL_BASE = "Base"

    # 默认超时常量
    DEFAULT_TTS_TIMEOUT = 300.0  # 默认TTS超时时间（秒）

    def __init__(
        self,
        model_path=None,
        model_type=None,
        device="cuda:0",
        dtype=None,
        attn_implementation=None,
        shared_tokenizer_path=None,
        enable_streaming: bool = True,
        streaming_chunk_size: int = 32,
    ):
        """
        初始化 Qwen TTS 引擎

        Args:
            model_path: 模型路径
            model_type: 模型类型 (CustomVoice/VoiceDesign/Base)
            device: 运行设备 ("cpu", "cuda", 或 "cuda:0")
            dtype: 数据类型（可选，传递给 qwen-tts）
            attn_implementation: 注意力实现（可选，传递给 qwen-tts）
            shared_tokenizer_path: 共享 tokenizer 路径（可选，如未指定则自动查找）
            enable_streaming: 是否启用流式输出（默认 True）
            streaming_chunk_size: 流式输出块大小（token 数量，默认 32）
        """
        self.model = None
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.shared_tokenizer_path = shared_tokenizer_path
        self.enable_streaming = enable_streaming
        self.streaming_chunk_size = streaming_chunk_size
        self._executor: Optional[ThreadPoolExecutor] = None
        self._load_model()

    def _load_model(self):
        """加载 Qwen3-TTS 模型"""
        global _SHARED_TOKENIZER_DIR

        try:
            logger.info(f"正在加载 Qwen3-TTS 模型 ({self.model_type or '默认'})...")
            logger.info(f"设备: {self.device}")

            # 设置共享 tokenizer 路径
            tokenizer_dir = self.shared_tokenizer_path

            logger.info(f"[DEBUG] shared_tokenizer_path (参数): {tokenizer_dir}")
            logger.info(f"[DEBUG] model_path: {self.model_path}")

            if tokenizer_dir is None and self.model_path:
                # 自动查找：模型路径父目录/tokenizer-12hz
                # 只在本地路径（非 HuggingFace Hub ID）时查找
                if os.path.isdir(self.model_path) or os.path.exists(self.model_path):
                    model_parent_dir = os.path.dirname(self.model_path)
                    tokenizer_dir = os.path.join(model_parent_dir, "tokenizer-12hz")
                    logger.info(f"[DEBUG] 自动计算的 tokenizer_dir: {tokenizer_dir}")
                    logger.info(f"[DEBUG] tokenizer_dir 是否存在: {os.path.exists(tokenizer_dir)}")
                else:
                    logger.info(f"[DEBUG] model_path 是 HuggingFace Hub ID，跳过本地 tokenizer 查找")
                    tokenizer_dir = None

            if tokenizer_dir and os.path.exists(tokenizer_dir):
                _SHARED_TOKENIZER_DIR = tokenizer_dir
                logger.info(f"使用共享 tokenizer: {tokenizer_dir}")
                # 应用 monkey-patch
                _patch_tokenizer_loading()
            else:
                if tokenizer_dir:
                    logger.warning(f"未找到共享 tokenizer ({tokenizer_dir})，使用模型内置 tokenizer")
                    logger.info(f"提示: 如果您有本地 tokenizer，请使用 shared_tokenizer_path 参数指定")
                _SHARED_TOKENIZER_DIR = None

            # 构建模型加载参数
            model_kwargs = {"device_map": self.device}

            # 如果指定了 dtype，添加到参数中
            if self.dtype:
                model_kwargs["dtype"] = self.dtype
                logger.info(f"数据类型: {self.dtype}")

            # 如果指定了注意力实现，添加到参数中
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation
                logger.info(f"注意力实现: {self.attn_implementation}")

            if self.model_path:
                # 使用自定义路径
                logger.info(f"加载模型: {self.model_path}")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            else:
                # 使用 HuggingFace 模型 ID
                model_id = f"Qwen/Qwen3-TTS-12Hz-1.7B-{self.model_type}" if self.model_type else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                logger.info(f"使用模型 ID: {model_id}")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    **model_kwargs
                )

            logger.info("✓ Qwen3-TTS 模型加载完成")

            # 应用流式生成 patch
            if self.enable_streaming:
                try:
                    # 应用第三方 streaming 项目的修改
                    apply_streaming_patch_to_qwen_tts()
                    logger.info(f"✓ 流式生成 patch 已应用 (chunk_size={self.streaming_chunk_size})")
                except Exception as e:
                    logger.warning(f"流式生成 patch 应用失败: {e}")
                    logger.warning("将使用非流式模式")
                    self.enable_streaming = False

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {str(e)}")
            raise

    # ========== Custom Voice 模式 ==========

    def custom_voice_synthesize(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Custom Voice 模式生成语音

        Args:
            text: 输入文本
            speaker: 说话人 (Vivian/Serena/Uncle_Fu/Dylan/Eric/Ryan/Aiden/Ono_Anna/Sohee)
            language: 语言 (Chinese/English/Japanese/Korean/Auto)
            instruct: 情感指令
            **kwargs: 其他参数 (speed_factor, pitch_factor 等)

        Returns:
            (audio_data, sample_rate)
        """
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
                **kwargs
            )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    def get_supported_speakers(self) -> list:
        """获取支持的说话人列表"""
        if self.model:
            try:
                return self.model.get_supported_speakers()
            except:
                pass
        return ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

    def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        if self.model:
            try:
                return self.model.get_supported_languages()
            except:
                pass
        return ["Chinese", "English", "Japanese", "Korean", "Auto"]

    # ========== Voice Design 模式 ==========

    def voice_design_synthesize(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Voice Design 模式生成语音

        Args:
            text: 输入文本
            design_prompt: 声音设计描述
            language: 语言
            **kwargs: 其他参数

        Returns:
            (audio_data, sample_rate)
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Voice Design): {text[:50]}...")

            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=design_prompt,
                **kwargs
            )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    # ========== Voice Clone 模式 ==========

    def voice_clone_synthesize(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Voice Clone 模式生成语音

        Args:
            text: 输入文本
            ref_audio: 参考音频路径
            ref_text: 参考文本
            clone_prompt: 已保存的 clone_prompt (可选)
            x_vector_only: 是否仅使用 x_vector (快速模式)
            **kwargs: 其他参数

        Returns:
            (audio_data, sample_rate)
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Voice Clone): {text[:50]}...")

            if clone_prompt:
                # 使用已保存的 clone_prompt
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=clone_prompt,
                    **kwargs
                )
            else:
                # 使用新的参考音频
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                    **kwargs
                )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    def create_voice_clone_prompt(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False
    ):
        """
        创建可重用的声音克隆 prompt

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            x_vector_only: 是否仅使用 x_vector

        Returns:
            prompt_items (可传递给 generate_voice_clone 的 voice_clone_prompt 参数)
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        try:
            logger.info("正在提取声音特征...")

            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )

            logger.info("✓ 声音特征提取完成")
            return prompt_items

        except Exception as e:
            logger.error(f"✗ 特征提取失败: {str(e)}")
            raise

    # ========== 异步API ==========

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取或创建线程池执行器

        Returns:
            ThreadPoolExecutor: 线程池执行器实例
        """
        if self._executor is None:
            self._executor = TTSThreadPoolManager().get_executor()
        return self._executor

    async def _execute_async_tts(
        self,
        operation_name: str,
        sync_func,
        timeout: float = DEFAULT_TTS_TIMEOUT,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """通用的异步TTS执行包装器

        Args:
            operation_name: 操作名称（用于日志）
            sync_func: 要执行的同步函数
            timeout: 超时时间（秒）
            **kwargs: 传递给同步函数的参数

        Returns:
            (audio_data, sample_rate)

        Raises:
            TTSModelNotLoadedError: 模型未加载
            TTSInvalidParameterError: 输入文本为空
            TTSTimeoutError: 合成超时
            TTSSynthesisError: 合成失败
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        # 验证文本参数（如果存在）
        text = kwargs.get('text', '')
        if text and not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        logger.info(f"[Async] 正在生成语音 ({operation_name}): {text[:50]}...")

        loop = asyncio.get_running_loop()
        executor = self._get_executor()

        try:
            wavs, sr = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: sync_func(**kwargs)
                ),
                timeout=timeout
            )
            logger.info(f"✓ [Async] 语音生成成功 ({operation_name})")
            return wavs[0], sr

        except asyncio.TimeoutError:
            logger.error(f"✗ [Async] 语音生成超时 (>{timeout}s)")
            raise TTSTimeoutError(f"操作超时: {operation_name}")
        except Exception as e:
            logger.error(f"✗ [Async] 语音生成失败 ({operation_name}): {str(e)}")
            raise TTSSynthesisError(f"语音合成失败: {operation_name}") from e

    async def custom_voice_synthesize_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        timeout: float = DEFAULT_TTS_TIMEOUT,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Custom Voice 模式生成语音 (异步版本)

        Args:
            text: 输入文本
            speaker: 说话人 (Vivian/Serena/Uncle_Fu/Dylan/Eric/Ryan/Aiden/Ono_Anna/Sohee)
            language: 语言 (Chinese/English/Japanese/Korean/Auto)
            instruct: 情感指令
            timeout: 超时时间（秒），默认300秒
            **kwargs: 其他参数 (speed_factor, pitch_factor 等)

        Returns:
            (audio_data, sample_rate)

        Raises:
            TTSModelNotLoadedError: 模型未加载
            TTSInvalidParameterError: 输入文本为空
            TTSTimeoutError: 合成超时
        """
        return await self._execute_async_tts(
            operation_name="Custom Voice",
            sync_func=self.model.generate_custom_voice,
            timeout=timeout,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            **kwargs
        )

    async def voice_design_synthesize_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        timeout: float = DEFAULT_TTS_TIMEOUT,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Voice Design 模式生成语音 (异步版本)

        Args:
            text: 输入文本
            design_prompt: 声音设计描述
            language: 语言
            timeout: 超时时间（秒），默认300秒
            **kwargs: 其他参数

        Returns:
            (audio_data, sample_rate)

        Raises:
            TTSModelNotLoadedError: 模型未加载
            TTSInvalidParameterError: 输入文本为空
            TTSTimeoutError: 合成超时
        """
        return await self._execute_async_tts(
            operation_name="Voice Design",
            sync_func=self.model.generate_voice_design,
            timeout=timeout,
            text=text,
            language=language,
            instruct=design_prompt,
            **kwargs
        )

    async def voice_clone_synthesize_async(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        clone_prompt=None,
        x_vector_only: bool = False,
        timeout: float = DEFAULT_TTS_TIMEOUT,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Voice Clone 模式生成语音 (异步版本)

        Args:
            text: 输入文本
            ref_audio: 参考音频路径
            ref_text: 参考文本
            clone_prompt: 已保存的 clone_prompt (可选)
            x_vector_only: 是否仅使用 x_vector (快速模式)
            timeout: 超时时间（秒），默认300秒
            **kwargs: 其他参数

        Returns:
            (audio_data, sample_rate)

        Raises:
            TTSModelNotLoadedError: 模型未加载
            TTSInvalidParameterError: 输入文本为空
            TTSTimeoutError: 合成超时
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        # 构建参数
        params = {
            "text": text,
            "language": "Auto",
            **kwargs
        }

        if clone_prompt:
            params["voice_clone_prompt"] = clone_prompt
        else:
            params.update({
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only
            })

        logger.info(f"[Async] 正在生成语音 (Voice Clone): {text[:50]}...")

        loop = asyncio.get_running_loop()
        executor = self._get_executor()

        try:
            wavs, sr = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: self.model.generate_voice_clone(**params)
                ),
                timeout=timeout
            )

            logger.info("✓ [Async] 语音生成成功")
            return wavs[0], sr

        except asyncio.TimeoutError:
            logger.error(f"✗ [Async] 语音生成超时 (>{timeout}s)")
            raise TTSTimeoutError(f"操作超时: Voice Clone")
        except Exception as e:
            logger.error(f"✗ [Async] 语音生成失败: {str(e)}")
            raise TTSSynthesisError(f"语音合成失败: Voice Clone") from e

    async def create_voice_clone_prompt_async(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False,
        timeout: float = DEFAULT_TTS_TIMEOUT
    ) -> Tuple[np.ndarray, int]:
        """
        创建可重用的声音克隆 prompt (异步版本)

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            x_vector_only: 是否仅使用 x_vector
            timeout: 超时时间（秒），默认300秒

        Returns:
            prompt_items (可传递给 generate_voice_clone 的 voice_clone_prompt 参数)

        Raises:
            TTSModelNotLoadedError: 模型未加载
            TTSTimeoutError: 特征提取超时
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        logger.info("[Async] 正在提取声音特征...")

        loop = asyncio.get_running_loop()
        executor = self._get_executor()

        try:
            prompt_items = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: self.model.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        x_vector_only_mode=x_vector_only
                    )
                ),
                timeout=timeout
            )

            logger.info("✓ [Async] 声音特征提取完成")
            return prompt_items

        except asyncio.TimeoutError:
            logger.error(f"✗ [Async] 特征提取超时 (>{timeout}s)")
            raise TTSTimeoutError(f"操作超时: Voice Clone Prompt")
        except Exception as e:
            logger.error(f"✗ [Async] 特征提取失败: {str(e)}")
            raise TTSSynthesisError(f"特征提取失败") from e

    # ========== 兼容旧 API ==========

    def synthesize(self, text, voice="default", speed=1.0, pitch=1.0):
        """
        基础合成方法 (向后兼容)

        内部调用 custom_voice_synthesize

        Returns:
            audio_data (numpy array) - 只返回音频数据，采样率固定为 24000
        """
        # 将旧参数映射到新参数
        speaker_map = {
            "default": "Vivian",
            "female": "Serena",
            "male": "Uncle_Fu"
        }
        speaker = speaker_map.get(voice, "Vivian")

        audio_data, _ = self.custom_voice_synthesize(
            text=text,
            speaker=speaker,
            language="Chinese",
            speed_factor=speed,
            pitch_factor=pitch
        )
        return audio_data

    def get_available_voices(self):
        """获取可用的声音列表 (向后兼容)"""
        return ["default", "female", "male", "child", "elderly"]

    def clone_voice(self, audio_samples, text):
        """
        声音克隆（向后兼容，已废弃）

        建议使用 voice_clone_synthesize 方法
        """
        logger.warning("clone_voice 方法已废弃，建议使用 voice_clone_synthesize")
        # 尝试使用新方法
        return self.voice_clone_synthesize(
            text=text,
            ref_audio=audio_samples,
            ref_text="",  # 旧版本没有参考文本
            x_vector_only=True
        )

    # ============================================
    # 真正的流式合成 API（基于 monkey patch）
    # ============================================

    async def custom_voice_synthesize_streaming_async(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        使用 Custom Voice 模式生成语音并流式返回（真正的流式输出）

        通过 monkey patch 拦截底层生成过程，实现边生成边解码边输出

        Args:
            text: 输入文本
            speaker: 说话人
            language: 语言
            instruct: 情感指令
            **kwargs: 其他参数

        Yields:
            Dict[str, Any]: 包含以下键的字典:
                - 'type': 'audio_chunk', 'done', 或 'error'
                - 'audio': np.ndarray - 音频数据 (当 type='audio_chunk')
                - 'sample_rate': int - 采样率
                - 'is_final': bool - 是否为最后一块
                - 'progress': str - 进度信息 (如 "128/256 tokens")

        Raises:
            TTSModelNotLoadedError: 模型未加载或流式 patch 未应用
            TTSInvalidParameterError: 输入文本为空
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not self.enable_streaming:
            raise TTSModelNotLoadedError("流式输出未启用，请在初始化时设置 enable_streaming=True")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        logger.info(f"[Streaming] 正在流式生成语音 (Custom Voice): {text[:50]}...")

        # 使用 Queue 实现真正的实时流式输出
        result_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        thread_started = threading.Event()

        def run_streaming_generation():
            # 在工作线程中运行流式生成并实时放入队列
            try:
                thread_started.set()

                params = self._prepare_custom_voice_params(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    **kwargs
                )

                # 调用官方 streaming 实现的 stream_generate_pcm 方法
                # 直接返回 PCM chunks，不再使用旧的 wrapper
                for audio_chunk, sample_rate in self.model.model.stream_generate_pcm(
                    emit_every_frames=self.streaming_chunk_size,
                    decode_window_frames=80,
                    overlap_samples=0,
                    **params
                ):
                    # 将 PCM chunk 包装成标准格式
                    result = {
                        'type': 'audio_chunk',
                        'audio': audio_chunk,
                        'sample_rate': sample_rate,
                        'is_final': False,
                    }
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put(result),
                        loop
                    )

                # 发送完成消息
                asyncio.run_coroutine_threadsafe(
                    result_queue.put({'type': 'done', 'sample_rate': 24000}),
                    loop
                )

            except Exception as e:
                logger.error(f"✗ [Streaming] 生成线程异常: {str(e)}", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    result_queue.put({'type': 'error', 'error': str(e)}),
                    loop
                )

        # 直接启动线程（不使用 run_in_executor）
        thread = threading.Thread(target=run_streaming_generation, daemon=True)
        thread.start()
        thread_started.wait()  # 等待线程启动

        # 从队列读取并 yield 结果（真正的实时流式）
        try:
            while True:
                result = await result_queue.get()
                yield result

                if result.get('type') == 'done':
                    logger.info("✓ [Streaming] 流式生成完成")
                    break
                elif result.get('type') == 'error':
                    error_msg = result.get('error', '未知错误')
                    logger.error(f"✗ [Streaming] 流式生成失败: {error_msg}")
                    raise TTSSynthesisError(f"流式生成失败: {error_msg}")

        except TTSSynthesisError:
            raise
        except Exception as e:
            logger.error(f"✗ [Streaming] 流式生成异常: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'error': str(e)
            }

    async def voice_design_synthesize_streaming_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        使用 Voice Design 模式生成语音并流式返回（真正的流式输出）

        Args:
            text: 输入文本
            design_prompt: 声音设计描述
            language: 语言
            **kwargs: 其他参数

        Yields:
            Dict[str, Any]: 流式生成结果字典
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not self.enable_streaming:
            raise TTSModelNotLoadedError("流式输出未启用")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        logger.info(f"[Streaming] 正在流式生成语音 (Voice Design): {text[:50]}...")

        # 使用 Queue 实现真正的实时流式输出
        result_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        thread_started = threading.Event()

        def run_streaming_generation():
            try:
                thread_started.set()

                params = self._prepare_voice_design_params(
                    text=text,
                    design_prompt=design_prompt,
                    language=language,
                    **kwargs
                )

                for result in self.model.model.generate_streaming_v4(**params):
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put(result),
                        loop
                    )

            except Exception as e:
                logger.error(f"✗ [Streaming] 生成线程异常: {str(e)}", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    result_queue.put({'type': 'error', 'error': str(e)}),
                    loop
                )

        # 直接启动线程
        thread = threading.Thread(target=run_streaming_generation, daemon=True)
        thread.start()
        thread_started.wait()

        try:
            while True:
                result = await result_queue.get()
                yield result

                if result.get('type') == 'done':
                    logger.info("✓ [Streaming] 流式生成完成")
                    break
                elif result.get('type') == 'error':
                    error_msg = result.get('error', '未知错误')
                    logger.error(f"✗ [Streaming] 流式生成失败: {error_msg}")
                    raise TTSSynthesisError(f"流式生成失败: {error_msg}")

        except TTSSynthesisError:
            raise
        except Exception as e:
            logger.error(f"✗ [Streaming] 流式生成异常: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'error': str(e)
            }

    async def voice_clone_synthesize_streaming_async(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        使用 Voice Clone 模式生成语音并流式返回（真正的流式输出）

        Args:
            text: 输入文本
            ref_audio: 参考音频路径
            ref_text: 参考文本
            clone_prompt: 已保存的 clone_prompt (可选)
            x_vector_only: 是否仅使用 x_vector (快速模式)
            **kwargs: 其他参数

        Yields:
            Dict[str, Any]: 流式生成结果字典
        """
        if not self.model:
            raise TTSModelNotLoadedError("模型未加载")

        if not self.enable_streaming:
            raise TTSModelNotLoadedError("流式输出未启用")

        if not text or not text.strip():
            raise TTSInvalidParameterError("输入文本不能为空")

        logger.info(f"[Streaming] 正在流式生成语音 (Voice Clone): {text[:50]}...")

        # 使用 Queue 实现真正的实时流式输出
        result_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        thread_started = threading.Event()

        def run_streaming_generation():
            try:
                thread_started.set()

                params = self._prepare_voice_clone_params(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    clone_prompt=clone_prompt,
                    x_vector_only=x_vector_only,
                    **kwargs
                )

                for result in self.model.model.generate_streaming_v4(**params):
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put(result),
                        loop
                    )

            except Exception as e:
                logger.error(f"✗ [Streaming] 生成线程异常: {str(e)}", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    result_queue.put({'type': 'error', 'error': str(e)}),
                    loop
                )

        # 直接启动线程
        thread = threading.Thread(target=run_streaming_generation, daemon=True)
        thread.start()
        thread_started.wait()

        try:
            while True:
                result = await result_queue.get()
                yield result

                if result.get('type') == 'done':
                    logger.info("✓ [Streaming] 流式生成完成")
                    break
                elif result.get('type') == 'error':
                    error_msg = result.get('error', '未知错误')
                    logger.error(f"✗ [Streaming] 流式生成失败: {error_msg}")
                    raise TTSSynthesisError(f"流式生成失败: {error_msg}")

        except TTSSynthesisError:
            raise
        except Exception as e:
            logger.error(f"✗ [Streaming] 流式生成异常: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'error': str(e)
            }

    # ============================================
    # 辅助方法：准备生成参数
    # ============================================

    def _prepare_custom_voice_params(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str,
        **kwargs
    ) -> Dict[str, Any]:
        """准备 Custom Voice 生成参数"""
        # 构建输入文本（参考 qwen-tts 的格式）
        input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        input = self.model.processor(text=input_text, return_tensors="pt", padding=True)
        input_id = input["input_ids"].to(self.device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

        # 构建 instruct_ids
        instruct_ids = None
        if instruct and instruct.strip():
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_input = self.model.processor(text=instruct_text, return_tensors="pt", padding=True)
            instruct_id = instruct_input["input_ids"].to(self.device)
            instruct_id = instruct_id.unsqueeze(0) if instruct_id.dim() == 1 else instruct_id
            instruct_ids = [instruct_id]

        return {
            "input_ids": [input_id],
            "instruct_ids": instruct_ids,
            "languages": [language],
            "speakers": [speaker],
            "non_streaming_mode": True,
            **kwargs
        }

    def _prepare_voice_design_params(
        self,
        text: str,
        design_prompt: str,
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """准备 Voice Design 生成参数"""
        input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        input = self.model.processor(text=input_text, return_tensors="pt", padding=True)
        input_id = input["input_ids"].to(self.device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

        instruct_ids = None
        if design_prompt and design_prompt.strip():
            instruct_text = f"<|im_start|>user\n{design_prompt}<|im_end|>\n"
            instruct_input = self.model.processor(text=instruct_text, return_tensors="pt", padding=True)
            instruct_id = instruct_input["input_ids"].to(self.device)
            instruct_id = instruct_id.unsqueeze(0) if instruct_id.dim() == 1 else instruct_id
            instruct_ids = [instruct_id]

        return {
            "input_ids": [input_id],
            "instruct_ids": instruct_ids,
            "languages": [language],
            "non_streaming_mode": True,
            **kwargs
        }

    def _prepare_voice_clone_params(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        clone_prompt,
        x_vector_only: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """准备 Voice Clone 生成参数"""
        input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        input = self.model.processor(text=input_text, return_tensors="pt", padding=True)
        input_id = input["input_ids"].to(self.device)
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

        # 准备 voice_clone_prompt
        if clone_prompt is None:
            # 创建新的 prompt
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )
            voice_clone_prompt = {
                "ref_code": [it.ref_code for it in prompt_items],
                "ref_spk_embedding": [it.ref_spk_embedding for it in prompt_items],
                "x_vector_only_mode": [it.x_vector_only_mode for it in prompt_items],
                "icl_mode": [it.icl_mode for it in prompt_items],
            }
        else:
            voice_clone_prompt = clone_prompt

        # 准备 ref_ids
        ref_ids = None
        if not x_vector_only and ref_text:
            ref_text_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
            ref_input = self.model.processor(text=ref_text_formatted, return_tensors="pt", padding=True)
            ref_id = ref_input["input_ids"].to(self.device)
            ref_id = ref_id.unsqueeze(0) if ref_id.dim() == 1 else ref_id
            ref_ids = [ref_id]

        return {
            "input_ids": [input_id],
            "ref_ids": ref_ids,
            "voice_clone_prompt": voice_clone_prompt,
            "languages": ["Auto"],
            "non_streaming_mode": False,  # Voice Clone 使用非流式模式
            **kwargs
        }

