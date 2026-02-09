"""
Qwen3-TTS 引擎封装（重构版）

提供文本转语音的核心功能，支持三种模式：
1. Custom Voice - 使用预设说话人 + 情感指令
2. Voice Design - 通过自然语言描述设计声音
3. Voice Clone - 使用参考音频克隆声音

所有三种模式都支持流式输出（基于修改版qwen-tts的 stream_generate_pcm API）
"""

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
import logging
import os
import numpy as np
import asyncio
from typing import Tuple, Optional, AsyncGenerator, Dict, Any, Generator, List
import torch

from .exceptions import (
    TTSError,
    TTSModelNotLoadedError,
    TTSInvalidParameterError,
    TTSSynthesisError
)

logger = logging.getLogger(__name__)


# ========== 共享 Tokenizer 支持 ==========

_SHARED_TOKENIZER_DIR = None  # 共享 tokenizer 路径


# ========== 音频加载 Patch ==========

_original_librosa_load = None  # 保存原始的 librosa.load 函数


def _load_audio_with_pydub(audio_path: str, sr: int = None, mono: bool = True):
    """
    使用 pydub (ffmpeg) 加载音频数据（直接返回音频数组，内存操作）

    Args:
        audio_path: 音频文件路径
        sr: 目标采样率（None 表示使用原采样率）
        mono: 是否转换为单声道

    Returns:
        tuple: (音频数据, 采样率)
    """
    import numpy as np
    logger = logging.getLogger(__name__)

    try:
        from pydub import AudioSegment
    except ImportError:
        logger.debug("pydub 未安装")
        return None

    try:
        # 安全验证：解析并规范化路径
        abs_audio_path = os.path.abspath(audio_path)
        normalized_path = os.path.normpath(abs_audio_path)

        # 验证文件存在
        if not os.path.exists(normalized_path):
            logger.warning(f"音频文件不存在: {audio_path}")
            return None

        # 验证是文件而不是目录
        if not os.path.isfile(normalized_path):
            logger.warning(f"路径不是文件: {audio_path}")
            return None

        # 验证文件大小
        from .audio_utils import MAX_AUDIO_FILE_SIZE
        file_size = os.path.getsize(normalized_path)
        if file_size == 0:
            logger.warning(f"音频文件为空: {audio_path}")
            return None
        if file_size > MAX_AUDIO_FILE_SIZE:
            logger.warning(f"音频文件过大: {file_size / 1024 / 1024:.2f} MB")
            return None

        logger.debug(f"使用 pydub 加载音频: {normalized_path}")

        # 使用 pydub 加载音频
        audio = AudioSegment.from_file(normalized_path)

        # 验证音频时长（防止资源耗尽攻击）
        from .audio_utils import MAX_AUDIO_DURATION_SECONDS
        duration_seconds = len(audio) / 1000.0
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            logger.warning(f"音频时长过长: {duration_seconds / 60:.2f} 分钟")
            return None

        # 记录原始音频信息
        original_sr = audio.frame_rate
        original_channels = audio.channels
        logger.debug(f"原始音频: sr={original_sr}, channels={original_channels}, duration={len(audio)}ms")

        # 设置输出格式
        if sr:
            audio = audio.set_frame_rate(sr)
        if mono:
            audio = audio.set_channels(1)

        # 转换为 numpy 数组并正确归一化
        # AudioSegment 的 sample_width 决定音频位深度：
        #   - 1 byte = 8-bit (unsigned, 范围 0-255)
        #   - 2 bytes = 16-bit (有符号, 范围 -32768 到 32767)
        #   - 3 bytes = 24-bit (有符号, 范围 -8388608 到 8388607)
        #   - 4 bytes = 32-bit (有符号, 范围 -2147483648 到 2147483647)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # 根据实际位深度计算归一化因子
        sample_width = audio.sample_width

        # 验证 sample_width 范围（防止意外的值导致错误）
        if sample_width < 1 or sample_width > 4:
            logger.warning(f"不支持的音频位深度: sample_width={sample_width}")
            return None

        if sample_width == 1:  # 8-bit unsigned
            # 先转换为有符号（以 128 为中心），再归一化
            samples = (samples - 128.0) / 128.0
        else:
            # 对于有符号整数，归一化因子为 2^(8*sample_width - 1)
            max_value = float(2 ** (8 * sample_width - 1))
            samples /= max_value  # 归一化到 [-1, 1]

        logger.debug(f"音频位深度: {8*sample_width}-bit, 归一化后范围: [{samples.min():.3f}, {samples.max():.3f}]")

        actual_sr = audio.frame_rate

        logger.info(f"✓ pydub 成功加载音频: {normalized_path}, shape={samples.shape}, sr={actual_sr}")
        return samples, actual_sr

    # 只捕获可恢复的异常，让严重错误向上传播
    except ImportError as e:
        logger.debug(f"pydub 未安装: {e}")
        return None
    except (FileNotFoundError, OSError, IOError) as e:
        logger.warning(f"音频文件读取失败: {type(e).__name__}: {e}")
        return None
    except RuntimeError as e:
        # pydub 处理失败（如不支持的视频编码）
        logger.warning(f"pydub 音频处理失败: {e}")
        return None


def _patched_librosa_load(path, *args, **kwargs):
    """
    Patched 版本的 librosa.load

    对于非 WAV 文件，优先使用 pydub (ffmpeg) 加载（避免 mpg123 错误）
    如果 pydub 不可用或失败，回退到原始的 librosa.load
    """
    # 如果是 WAV 文件，直接使用原始方法
    if path.lower().endswith('.wav'):
        return _original_librosa_load(path, *args, **kwargs)

    logger.debug(f"[PATCH] 加载非 WAV 音频: {path}")

    # 尝试使用 pydub 加载
    result = _load_audio_with_pydub(path,
                                     sr=kwargs.get('sr'),
                                     mono=kwargs.get('mono', True))

    if result is not None:
        audio, sr = result
        logger.info(f"✓ 使用 pydub (ffmpeg) 成功加载音频: {path}")
        return audio, sr

    # pydub 失败，回退到原始 librosa.load
    logger.debug(f"回退到 librosa.load: {path}")
    return _original_librosa_load(path, *args, **kwargs)


def _apply_librosa_patch():
    """应用 librosa.load patch"""
    global _original_librosa_load

    try:
        import librosa

        # 保存原始函数
        if _original_librosa_load is None:
            _original_librosa_load = librosa.load

        # 应用 patch
        librosa.load = _patched_librosa_load

        logger.info("[PATCH] ✓ 已应用 librosa.load patch（使用 pydub/ffmpeg 加载非 WAV 音频）")

    except ImportError:
        logger.warning("[PATCH] ⚠ librosa 未安装，跳过 patch")
    except Exception as e:
        logger.error(f"[PATCH] ✗ 应用 librosa.patch 失败: {e}")


# =========================================


def _patch_tokenizer_loading():
    """
    Patch transformers 的 cached_file 和 cached_files 以支持共享 tokenizer。
    """
    from transformers.utils import hub as transformers_hub
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    original_cached_file = transformers_hub.cached_file
    original_cached_files = transformers_hub.cached_files

    logger.info(f"[PATCH] 开始应用共享 tokenizer patch，共享目录: {_SHARED_TOKENIZER_DIR}")

    def patched_cached_files(path_or_repo_id, filenames, **kwargs):
        global _SHARED_TOKENIZER_DIR

        if _SHARED_TOKENIZER_DIR and filenames:
            first_filename = filenames[0] if filenames else ""

            if first_filename.startswith("speech_tokenizer/"):
                logger.info(f"[PATCH] cached_files 拦截到请求: {filenames}")
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

        return original_cached_files(path_or_repo_id, filenames, **kwargs)

    transformers_hub.cached_files = patched_cached_files

    def patched_cached_file(pretrained_model_name_or_path, filename, *args, **kwargs):
        global _SHARED_TOKENIZER_DIR

        if _SHARED_TOKENIZER_DIR and filename.startswith("speech_tokenizer/"):
            logger.info(f"[PATCH] cached_file 拦截到请求: {filename}")
            new_path = os.path.join(_SHARED_TOKENIZER_DIR, filename.replace("speech_tokenizer/", ""))
            if os.path.exists(new_path):
                logger.info(f"[PATCH] ✓ 返回共享文件: {new_path}")
                return new_path
            else:
                logger.warning(f"[PATCH] ✗ 共享目录中找不到文件: {new_path}")

        return original_cached_file(pretrained_model_name_or_path, filename, *args, **kwargs)

    transformers_hub.cached_file = patched_cached_file

    original_from_pretrained = Qwen3TTSTokenizer.from_pretrained.__func__

    def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        global _SHARED_TOKENIZER_DIR

        if _SHARED_TOKENIZER_DIR and "speech_tokenizer" in str(pretrained_model_name_or_path):
            logger.info(f"[PATCH] Qwen3TTSTokenizer 拦截到请求: {pretrained_model_name_or_path}")
            return original_from_pretrained(cls, _SHARED_TOKENIZER_DIR, **kwargs)

        return original_from_pretrained(cls, pretrained_model_name_or_path, **kwargs)

    Qwen3TTSTokenizer.from_pretrained = classmethod(patched_from_pretrained)

    try:
        from qwen_tts.core.models import modeling_qwen3_tts
        modeling_qwen3_tts.cached_file = patched_cached_file
        logger.info("[PATCH] ✓ 已更新 qwen-tts 模块中的 cached_file 引用")
    except Exception as e:
        logger.warning(f"[PATCH] ⚠ 无法更新 qwen-tts 模块引用: {e}")

    logger.info("[PATCH] ✓ 所有 Patch 应用完成")


# =========================================

class QwenEngine:
    """简化的 Qwen3-TTS 引擎"""

    # 模型类型常量
    MODEL_CUSTOM_VOICE = "CustomVoice"
    MODEL_VOICE_DESIGN = "VoiceDesign"
    MODEL_BASE = "Base"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        device: str = "cuda:0",
        dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        shared_tokenizer_path: Optional[str] = None,
        enable_streaming: bool = True,
        streaming_decode_window: int = 80,
    ):
        """
        初始化 Qwen TTS 引擎

        Args:
            model_path: 模型路径（可选，默认使用HuggingFace Hub）
            model_type: 模型类型 (CustomVoice/VoiceDesign/Base)
            device: 运行设备 ("cpu", "cuda", 或 "cuda:0")
            dtype: 数据类型（默认 torch.bfloat16）
            attn_implementation: 注意力实现（可选）
            shared_tokenizer_path: 共享 tokenizer 路径（可选）
            enable_streaming: 是否启用流式输出（默认 True）
            streaming_decode_window: 流式解码窗口大小（默认 80）
        """
        self.model: Optional[Qwen3TTSModel] = None
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.shared_tokenizer_path = shared_tokenizer_path
        self.enable_streaming = enable_streaming
        self.streaming_decode_window = streaming_decode_window

        # 当前激活的优化模式（'streaming' 或 'non_streaming'）
        self._current_optimization_mode = None
        self._optimization_lock = asyncio.Lock()  # 保护优化切换的锁

        self._load_model()

    def _load_model(self):
        """加载 Qwen3-TTS 模型"""
        global _SHARED_TOKENIZER_DIR

        # 应用 librosa patch（在模型加载前）
        _apply_librosa_patch()

        try:
            logger.info(f"正在加载 Qwen3-TTS 模型 ({self.model_type or '默认'})...")
            logger.info(f"设备: {self.device}")

            # 设置共享 tokenizer 路径
            tokenizer_dir = self.shared_tokenizer_path

            if tokenizer_dir is None and self.model_path:
                # 自动查找共享 tokenizer
                if os.path.isdir(self.model_path) or os.path.exists(self.model_path):
                    model_parent_dir = os.path.dirname(self.model_path)
                    tokenizer_dir = os.path.join(model_parent_dir, "tokenizer-12hz")
                    logger.info(f"[DEBUG] 自动计算的 tokenizer_dir: {tokenizer_dir}")
                else:
                    logger.info(f"[DEBUG] model_path 是 HuggingFace Hub ID，跳过本地 tokenizer 查找")
                    tokenizer_dir = None

            if tokenizer_dir and os.path.exists(tokenizer_dir):
                _SHARED_TOKENIZER_DIR = tokenizer_dir
                logger.info(f"使用共享 tokenizer: {tokenizer_dir}")
                _patch_tokenizer_loading()
            else:
                if tokenizer_dir:
                    logger.warning(f"未找到共享 tokenizer ({tokenizer_dir})，使用模型内置 tokenizer")
                _SHARED_TOKENIZER_DIR = None

            # 构建模型加载参数
            model_kwargs = {"device_map": self.device}

            if self.dtype:
                model_kwargs["dtype"] = self.dtype
                logger.info(f"数据类型: {self.dtype}")

            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation
                logger.info(f"注意力实现: {self.attn_implementation}")

            # 加载模型
            if self.model_path:
                logger.info(f"加载模型: {self.model_path}")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            else:
                model_id = f"Qwen/Qwen3-TTS-12Hz-1.7B-{self.model_type}" if self.model_type else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                logger.info(f"使用模型 ID: {model_id}")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    **model_kwargs
                )

            logger.info("✓ Qwen3-TTS 模型加载完成")

            # 默认使用非流式优化初始化（UI 和非流式 API 使用）
            self._apply_optimizations("non_streaming")
            self._current_optimization_mode = "non_streaming"

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {str(e)}")
            raise

    def _apply_optimizations(self, mode: str):
        """
        应用指定模式的优化配置

        Args:
            mode: 'streaming' 或 'non_streaming'
        """
        if not self.model:
            return

        try:
            if mode == "streaming":
                # 流式模式
                logger.info("正在启用流式生成优化...")
                self.model.enable_streaming_optimizations(
                    decode_window_frames=self.streaming_decode_window,
                    use_compile=False,  
                    use_cuda_graphs=False,
                    use_fast_codebook=True,  # ✅ 快速 codebook（2-3x 加速）
                )
                logger.info(f"✓ 流式优化已启用 (decode_window={self.streaming_decode_window}, 快速 codebook)")

            elif mode == "non_streaming":
                # 非流式模式：使用完整优化
                logger.info("正在启用非流式生成优化...")
                self.model.enable_streaming_optimizations(
                    decode_window_frames=300,  # 非流式使用更大的窗口
                    use_compile=False,  
                    use_cuda_graphs=False,  
                    use_fast_codebook=True,  # ✅ 快速 codebook（2-3x 加速）
                    compile_codebook_predictor=True,  # ✅ 编译 codebook predictor
                )
                logger.info("✓ 非流式优化已启用 (快速 codebook)")

        except Exception as e:
            logger.error(f"✗ 优化功能应用失败: {e}")
            raise

    async def _ensure_optimization_mode(self, mode: str):
        """
        确保模型使用指定的优化模式

        如果当前模式不是目标模式，则切换优化配置。

        Args:
            mode: 'streaming' 或 'non_streaming'
        """
        if self._current_optimization_mode == mode:
            return  # 已经是目标模式，无需切换

        async with self._optimization_lock:
            # 双重检查，避免在获取锁后重复切换
            if self._current_optimization_mode == mode:
                return

            logger.info(f"正在切换优化模式: {self._current_optimization_mode} → {mode}")

            # 在线程池中执行优化切换（可能是 CPU 密集型操作）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._apply_optimizations, mode)

            self._current_optimization_mode = mode
            logger.info(f"✓ 优化模式已切换至: {mode}")

    # ========== Custom Voice 模式 ==========

    def custom_voice_synthesize(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs
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
                **kwargs
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
        **kwargs
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
            **kwargs
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
        **kwargs
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
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input = self.model.processor(text=assistant_text, return_tensors="pt", padding=True)
        input_ids = [input["input_ids"].to(self.model.device)]
        if input_ids[0].dim() == 1:
            input_ids[0] = input_ids[0].unsqueeze(0)

        instruct_ids = None
        if instruct and instruct.strip():
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_input = self.model.processor(text=instruct_text, return_tensors="pt", padding=True)
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

    # ========== Voice Design 模式 ==========

    def voice_design_synthesize(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Voice Design 同步生成"""
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

    async def voice_design_synthesize_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Voice Design 异步生成（非流式优化）"""
        # 确保使用非流式优化
        await self._ensure_optimization_mode("non_streaming")

        return await asyncio.to_thread(
            self.voice_design_synthesize,
            text=text,
            design_prompt=design_prompt,
            language=language,
            **kwargs
        )

    async def voice_design_synthesize_streaming_async(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        emit_every_frames: int = 8,
        decode_window_frames: int = 80,
        overlap_samples: int = 240,  # 10ms @ 24kHz = 240 samples
        **kwargs
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
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input = self.model.processor(text=assistant_text, return_tensors="pt", padding=True)
        input_ids = [input["input_ids"].to(self.model.device)]
        if input_ids[0].dim() == 1:
            input_ids[0] = input_ids[0].unsqueeze(0)

        instruct_text = f"<|im_start|>user\n{design_prompt}<|im_end|>\n"
        instruct_input = self.model.processor(text=instruct_text, return_tensors="pt", padding=True)
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

    # ========== Voice Clone 模式 ==========

    def _convert_prompt_to_prompt_items(self, clone_prompt) -> List[VoiceClonePromptItem]:
        """
        将各种格式的 clone_prompt 转换为 VoiceClonePromptItem 对象列表

        Args:
            clone_prompt: 可以是以下格式之一:
                - VoiceClonePromptItem 对象
                - List[VoiceClonePromptItem]
                - dict (从 load_prompt_features 加载的格式)

        Returns:
            List[VoiceClonePromptItem]: 模型期望的对象列表
        """
        # 如果已经是列表，验证格式
        if isinstance(clone_prompt, list):
            if len(clone_prompt) == 0:
                raise TTSInvalidParameterError("clone_prompt 列表为空")
            if isinstance(clone_prompt[0], VoiceClonePromptItem):
                return clone_prompt
            else:
                # 列表中的元素不是 VoiceClonePromptItem，尝试转换
                raise TTSInvalidParameterError(
                    "clone_prompt 列表中的元素必须是 VoiceClonePromptItem 对象"
                )

        # 如果是单个 VoiceClonePromptItem 对象，包装成列表
        if isinstance(clone_prompt, VoiceClonePromptItem):
            return [clone_prompt]

        # 如果是字典（从 load_prompt_features 加载的格式）
        if isinstance(clone_prompt, dict):
            try:
                # 提取列表格式的数据
                ref_code_list = clone_prompt.get("ref_code")  # 可能是 None 或 [tensor]
                ref_spk_embedding_list = clone_prompt["ref_spk_embedding"]  # [tensor]
                x_vector_only_mode_list = clone_prompt["x_vector_only_mode"]  # [bool]
                icl_mode_list = clone_prompt["icl_mode"]  # [bool]
                ref_text_list = clone_prompt.get("ref_text")  # 可能是 None 或 [str]

                # 确保所有字段都是列表且长度一致
                n = len(ref_spk_embedding_list)

                # 构建 VoiceClonePromptItem 列表
                prompt_items = []
                for i in range(n):
                    ref_code = ref_code_list[i] if ref_code_list is not None else None
                    ref_spk_embedding = ref_spk_embedding_list[i]
                    x_vector_only_mode = x_vector_only_mode_list[i]
                    icl_mode = icl_mode_list[i]
                    ref_text = ref_text_list[i] if ref_text_list is not None else None

                    # 确保张量在正确的设备上
                    if hasattr(ref_spk_embedding, 'to'):
                        ref_spk_embedding = ref_spk_embedding.to(self.device)
                    if ref_code is not None and hasattr(ref_code, 'to'):
                        ref_code = ref_code.to(self.device)

                    prompt_items.append(VoiceClonePromptItem(
                        ref_code=ref_code,
                        ref_spk_embedding=ref_spk_embedding,
                        x_vector_only_mode=x_vector_only_mode,
                        icl_mode=icl_mode,
                        ref_text=ref_text
                    ))

                logger.info(f"✓ 已转换 {len(prompt_items)} 个 VoiceClonePromptItem 对象")
                return prompt_items

            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"转换 clone_prompt 字典失败: {e}")
                raise TTSInvalidParameterError(
                    f"clone_prompt 字典格式无效: {e}"
                )

        # 不支持的格式
        logger.error(f"不支持的 clone_prompt 类型: {type(clone_prompt)}")
        raise TTSInvalidParameterError(
            "clone_prompt 必须是 VoiceClonePromptItem、List[VoiceClonePromptItem] 或 dict"
        )

    def voice_clone_synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs
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
                excluded_params = ['timeout', 'ref_audio', 'ref_text', 'x_vector_only_mode', 'x_vector_only']
                model_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}

                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=prompt_items,  # ← 传递列表而不是字典
                    **model_kwargs
                )
            else:
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

    async def voice_clone_synthesize_async(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs
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
            **kwargs
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
        **kwargs
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
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )

        # 转换为 VoiceClonePromptItem 对象列表（确保格式正确）
        prompt_items = self._convert_prompt_to_prompt_items(clone_prompt)

        # 准备 tokens - 使用正确的方法
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input = self.model.processor(text=assistant_text, return_tensors="pt", padding=True)
        input_ids = [input["input_ids"].to(self.model.device)]
        if input_ids[0].dim() == 1:
            input_ids[0] = input_ids[0].unsqueeze(0)

        # 使用模型方法转换为 dict 格式（包含所有必要信息）
        prompt_dict = self.model._prompt_items_to_voice_clone_prompt(prompt_items)

        # 构建 ref_ids（从 prompt_items 中提取 ref_text）
        ref_ids = None
        if prompt_items[0].ref_text and not prompt_items[0].x_vector_only_mode:
            ref_text_formatted = f"<|im_start|>assistant\n{prompt_items[0].ref_text}<|im_end|>\n"
            ref_input = self.model.processor(text=ref_text_formatted, return_tensors="pt", padding=True)
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

    def create_voice_clone_prompt(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False
    ):
        """创建可重用的声音克隆 prompt"""
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

    async def create_voice_clone_prompt_async(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False
    ):
        """创建可重用的声音克隆 prompt（异步版本）"""
        return await asyncio.to_thread(
            self.create_voice_clone_prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only
        )

    def create_and_save_prompt_features(
        self,
        ref_audio: str,
        ref_text: str,
        save_path: str,
        x_vector_only: bool = False
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
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only=x_vector_only
            )

            # 保存特征
            from tts.prompt_serializer import save_prompt_features

            metadata = {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only": x_vector_only
            }

            return save_prompt_features(prompt_item, save_path, metadata)

        except Exception as e:
            logger.error(f"创建并保存特征失败: {e}")
            return False

    def unload(self):
        """
        卸载模型并释放资源

        释放模型占用的显存和内存
        """
        if self.model is None:
            return

        try:
            logger.info("正在卸载 TTS 模型...")

            # 释放模型引用
            if hasattr(self.model, 'model'):
                del self.model.model

            if hasattr(self.model, 'processor'):
                del self.model.processor

            del self.model
            self.model = None

            # 清理 CUDA 缓存（如果使用 CUDA）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✓ CUDA 缓存已清理")

            logger.info("✓ TTS 模型已卸载")

        except Exception as e:
            logger.error(f"✗ 模型卸载失败: {str(e)}")

    # ========== 辅助方法 ==========

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

    # ========== 兼容旧 API ==========

    def synthesize(self, text, voice="default", speed=1.0, pitch=1.0):
        """基础合成方法 (向后兼容)"""
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
