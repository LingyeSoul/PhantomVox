"""
模型加载器模块

提供 Qwen3-TTS 模型的加载、优化和卸载功能
"""

import os
import logging
import asyncio
import gc
from typing import Optional, Literal

import torch

OptimizationMode = Literal["streaming", "non_streaming"]

logger = logging.getLogger(__name__)

_SHARED_TOKENIZER_DIR = None


def _patch_tokenizer_loading():
    """Patch transformers 的 cached_file 以支持共享 tokenizer"""
    from transformers.utils import hub as transformers_hub
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    original_cached_file = transformers_hub.cached_file
    original_cached_files = transformers_hub.cached_files

    logger.info(
        f"[PATCH] 开始应用共享 tokenizer patch，共享目录: {_SHARED_TOKENIZER_DIR}"
    )

    def patched_cached_files(path_or_repo_id, filenames, **kwargs):
        global _SHARED_TOKENIZER_DIR

        if _SHARED_TOKENIZER_DIR and filenames:
            first_filename = filenames[0] if filenames else ""

            if first_filename.startswith("speech_tokenizer/"):
                logger.info(f"[PATCH] cached_files 拦截到请求: {filenames}")
                new_filenames = [
                    os.path.join(
                        _SHARED_TOKENIZER_DIR, f.replace("speech_tokenizer/", "")
                    )
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
            new_path = os.path.join(
                _SHARED_TOKENIZER_DIR, filename.replace("speech_tokenizer/", "")
            )
            if os.path.exists(new_path):
                logger.info(f"[PATCH] ✓ 返回共享文件: {new_path}")
                return new_path
            else:
                logger.warning(f"[PATCH] ✗ 共享目录中找不到文件: {new_path}")

        return original_cached_file(
            pretrained_model_name_or_path, filename, *args, **kwargs
        )

    transformers_hub.cached_file = patched_cached_file

    original_from_pretrained = Qwen3TTSTokenizer.from_pretrained.__func__

    def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        global _SHARED_TOKENIZER_DIR

        if _SHARED_TOKENIZER_DIR and "speech_tokenizer" in str(
            pretrained_model_name_or_path
        ):
            logger.info(
                f"[PATCH] Qwen3TTSTokenizer 拦截到请求: {pretrained_model_name_or_path}"
            )
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


class ModelLoader:
    """Qwen3-TTS 模型加载器
    
    提供模型的加载、卸载、优化模式切换和智能显存管理功能。
    
    Attributes:
        model_path: 模型路径或 HuggingFace Hub ID
        model_type: 模型类型标识
        device: 加载设备 (如 "cuda:0", "cpu")
        smart_vram_enabled: 是否启用智能显存管理
        delay_cleanup_seconds: 延迟清理的等待秒数
    
    Example:
        loader = ModelLoader(model_path="/path/to/model", device="cuda:0")
        loader.load()
        # ... 使用模型 ...
        await loader.unload_async()
    """

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
        smart_vram_enabled: bool = True,
        delay_cleanup_seconds: int = 60,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.shared_tokenizer_path = shared_tokenizer_path
        self.enable_streaming = enable_streaming
        self.streaming_decode_window = streaming_decode_window
        self.smart_vram_enabled = smart_vram_enabled
        self.delay_cleanup_seconds = delay_cleanup_seconds

        self.model = None
        self._current_optimization_mode: Optional[OptimizationMode] = None
        self._optimization_lock = asyncio.Lock()
        self._cleanup_timer = None  # 定时清理任务
        self._model_on_cpu = False  # 标记模型是否已移到CPU

    def load(self):
        """加载 Qwen3-TTS 模型"""
        global _SHARED_TOKENIZER_DIR

        from .audio_loader import apply_librosa_patch

        apply_librosa_patch()

        try:
            logger.info(f"正在加载 Qwen3-TTS 模型 ({self.model_type or '默认'})...")
            logger.info(f"设备: {self.device}")

            tokenizer_dir = self.shared_tokenizer_path

            if tokenizer_dir is None and self.model_path:
                if os.path.isdir(self.model_path) or os.path.exists(self.model_path):
                    model_parent_dir = os.path.dirname(self.model_path)
                    tokenizer_dir = os.path.join(model_parent_dir, "tokenizer-12hz")
                    logger.info(f"[DEBUG] 自动计算的 tokenizer_dir: {tokenizer_dir}")
                else:
                    logger.info(
                        "[DEBUG] model_path 是 HuggingFace Hub ID，跳过本地 tokenizer 查找"
                    )
                    tokenizer_dir = None

            if tokenizer_dir and os.path.exists(tokenizer_dir):
                _SHARED_TOKENIZER_DIR = tokenizer_dir
                logger.info(f"使用共享 tokenizer: {tokenizer_dir}")
                _patch_tokenizer_loading()
            else:
                if tokenizer_dir:
                    logger.warning(
                        f"未找到共享 tokenizer ({tokenizer_dir})，使用模型内置 tokenizer"
                    )
                _SHARED_TOKENIZER_DIR = None

            from qwen_tts import Qwen3TTSModel

            model_kwargs = {"device_map": self.device}

            if self.dtype:
                model_kwargs["dtype"] = self.dtype
                logger.info(f"数据类型: {self.dtype}")

            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation
                logger.info(f"注意力实现: {self.attn_implementation}")

            if self.model_path:
                logger.info(f"加载模型: {self.model_path}")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_path, **model_kwargs
                )
            else:
                model_id = (
                    f"Qwen/Qwen3-TTS-12Hz-1.7B-{self.model_type}"
                    if self.model_type
                    else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                )
                logger.info(f"使用模型 ID: {model_id}")
                self.model = Qwen3TTSModel.from_pretrained(model_id, **model_kwargs)

            logger.info("✓ Qwen3-TTS 模型加载完成")

            self._apply_optimizations("non_streaming")
            self._current_optimization_mode = "non_streaming"

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {str(e)}")
            raise

    def _apply_optimizations(self, mode: OptimizationMode):
        """应用指定模式的优化配置"""
        if not self.model:
            return

        try:
            if mode == "streaming":
                logger.info("正在启用流式生成优化...")
                self.model.enable_streaming_optimizations(
                    decode_window_frames=self.streaming_decode_window,
                    use_compile=False,
                    use_cuda_graphs=False,
                    use_fast_codebook=True,
                )
                logger.info(
                    f"✓ 流式优化已启用 (decode_window={self.streaming_decode_window}, 快速 codebook)"
                )

            elif mode == "non_streaming":
                logger.info("正在启用非流式生成优化...")
                self.model.enable_streaming_optimizations(
                    decode_window_frames=300,
                    use_compile=False,
                    use_cuda_graphs=False,
                    use_fast_codebook=True,
                    compile_codebook_predictor=True,
                )
                logger.info("✓ 非流式优化已启用 (快速 codebook)")

        except Exception as e:
            logger.error(f"✗ 优化功能应用失败: {e}")
            raise

    async def ensure_optimization_mode(self, mode: OptimizationMode):
        """确保模型使用指定的优化模式"""
        if self._current_optimization_mode == mode:
            return

        async with self._optimization_lock:
            if self._current_optimization_mode == mode:
                return

            logger.info(f"正在切换优化模式: {self._current_optimization_mode} → {mode}")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._apply_optimizations, mode)

            self._current_optimization_mode = mode
            logger.info(f"✓ 优化模式已切换至: {mode}")

    async def unload_async(self):
        """卸载模型并释放资源（异步方法，支持智能显存管理）"""
        if self.model is None:
            return
        try:
            if self._cleanup_timer and not self._cleanup_timer.done():
                self._cleanup_timer.cancel()
                logger.info("✓ 已取消之前的定时清理任务")
            logger.info("正在卸载 TTS 模型...")
            if self.smart_vram_enabled:
                if hasattr(self.model, "model") and self.model.model is not None:
                    try:
                        self.model.model.cpu()
                        self._model_on_cpu = True
                        logger.info("✓ 模型已移至 CPU")
                    except Exception as e:
                        logger.warning(f"将模型移至 CPU 失败: {e}")
                self._cleanup_cuda_memory()
                self._cleanup_timer = asyncio.create_task(self._delayed_cleanup())
                logger.info("✓ 显存已释放，定时清理任务已启动")
            else:
                self._force_cleanup()
                logger.info("✓ 模型已直接清除（非智能模式）")
        except Exception as e:
            logger.error(f"✗ 模型卸载失败: {str(e)}")

    def _recursive_delete(self, module):
        """删除模块的所有子模块、参数和缓冲区（迭代方式）"""
        stack = [module]
        while stack:
            current = stack.pop()
            for name, child in list(current.named_children()):
                stack.append(child)
                delattr(current, name)
            for name, param in list(current.named_parameters()):
                if param is not None:
                    del param
            for name, buffer in list(current.named_buffers()):
                if buffer is not None:
                    del buffer

    def _force_cleanup(self):
        """强制清理模型资源"""
        if hasattr(self.model, "model"):
            self._recursive_delete(self.model.model)
        if hasattr(self.model, "processor"):
            del self.model.processor
        del self.model
        self.model = None
        self._model_on_cpu = False
        self._cleanup_cuda_memory()

    async def _delayed_cleanup(self):
        """延迟清理模型（智能显存模式）"""
        cleanup_completed = False
        try:
            await asyncio.sleep(self.delay_cleanup_seconds)
            if self.model is None:
                cleanup_completed = True
                return
            logger.info("✓ 定时清理触发，正在清除模型...")
            self._force_cleanup()
            cleanup_completed = True
            logger.info("✓ 智能显存模式：模型已延迟清除")
        except asyncio.CancelledError:
            if not cleanup_completed:
                logger.info("✓ 定时清理任务已被取消（模型重新加载）")
            raise
        except Exception as e:
            logger.error(f"✗ 延迟清理失败: {str(e)}")

    def unload(self):
        """卸载模型并释放资源"""
        if self.model is None:
            return

        try:
            logger.info("正在卸载 TTS 模型...")

            if hasattr(self.model, "model") and self.model.model is not None:
                try:
                    self.model.model.cpu()
                except Exception:
                    pass

            if hasattr(self.model, "model"):
                self._recursive_delete(self.model.model)

            if hasattr(self.model, "processor"):
                del self.model.processor

            del self.model
            self.model = None

            logger.info("✓ TTS 模型已卸载")

        except Exception as e:
            logger.error(f"✗ 模型卸载失败: {str(e)}")
        finally:
            self._cleanup_cuda_memory()

    def _cleanup_cuda_memory(self):
        """
        彻底清理 CUDA 显存
        
        执行所有可能的 PyTorch 显存清理操作
        """
        if not torch.cuda.is_available():
            return

        try:
            logger.info("正在清理 CUDA 显存...")

            # 强制垃圾回收 Python 对象
            gc.collect()
            logger.info("✓ 垃圾回收完成 (已收集对象)")

            # 清空 PyTorch CUDA 缓存
            torch.cuda.empty_cache()
            logger.info("✓ CUDA 缓存已清空")

            # 同步 CUDA 操作（确保所有操作完成）
            torch.cuda.synchronize()
            logger.info("✓ CUDA 操作已同步")

            # 获取清理后的显存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            logger.info(f"✓ 清理后显存使用: 已分配 {allocated:.2f}MB, 已预留 {reserved:.2f}MB")


        except Exception as e:
            logger.error(f"✗ CUDA 内存清理失败: {str(e)}")
