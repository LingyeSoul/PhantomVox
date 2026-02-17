"""
音频加载模块

提供音频加载功能，支持 pydub/ffmpeg 和 librosa 两种方式
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 保存原始的 librosa.load 函数
_original_librosa_load = None


def _load_audio_with_pydub(
    audio_path: str, sr: int = None, mono: bool = True
) -> Optional[Tuple[np.ndarray, int]]:
    """
    使用 pydub (ffmpeg) 加载音频数据

    Args:
        audio_path: 音频文件路径
        sr: 目标采样率（None 表示使用原采样率）
        mono: 是否转换为单声道

    Returns:
        tuple: (音频数据, 采样率) 或 None
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.debug("pydub 未安装")
        return None

    try:
        abs_audio_path = os.path.abspath(audio_path)
        normalized_path = os.path.normpath(abs_audio_path)

        if not os.path.exists(normalized_path):
            logger.warning(f"音频文件不存在: {audio_path}")
            return None

        if not os.path.isfile(normalized_path):
            logger.warning(f"路径不是文件: {audio_path}")
            return None

        from .audio_utils import MAX_AUDIO_FILE_SIZE

        file_size = os.path.getsize(normalized_path)
        if file_size == 0:
            logger.warning(f"音频文件为空: {audio_path}")
            return None
        if file_size > MAX_AUDIO_FILE_SIZE:
            logger.warning(f"音频文件过大: {file_size / 1024 / 1024:.2f} MB")
            return None

        logger.debug(f"使用 pydub 加载音频: {normalized_path}")

        audio = AudioSegment.from_file(normalized_path)

        from .audio_utils import MAX_AUDIO_DURATION_SECONDS

        duration_seconds = len(audio) / 1000.0
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            logger.warning(f"音频时长过长: {duration_seconds / 60:.2f} 分钟")
            return None

        original_sr = audio.frame_rate
        original_channels = audio.channels
        logger.debug(
            f"原始音频: sr={original_sr}, channels={original_channels}, duration={len(audio)}ms"
        )

        if sr:
            audio = audio.set_frame_rate(sr)
        if mono:
            audio = audio.set_channels(1)

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        sample_width = audio.sample_width

        if sample_width < 1 or sample_width > 4:
            logger.warning(f"不支持的音频位深度: sample_width={sample_width}")
            return None

        if sample_width == 1:
            samples = (samples - 128.0) / 128.0
        else:
            max_value = float(2 ** (8 * sample_width - 1))
            samples /= max_value

        logger.debug(
            f"音频位深度: {8 * sample_width}-bit, 归一化后范围: [{samples.min():.3f}, {samples.max():.3f}]"
        )

        actual_sr = audio.frame_rate

        logger.info(
            f"✓ pydub 成功加载音频: {normalized_path}, shape={samples.shape}, sr={actual_sr}"
        )
        return samples, actual_sr

    except ImportError as e:
        logger.debug(f"pydub 未安装: {e}")
        return None
    except (FileNotFoundError, OSError, IOError) as e:
        logger.warning(f"音频文件读取失败: {type(e).__name__}: {e}")
        return None
    except RuntimeError as e:
        logger.warning(f"pydub 音频处理失败: {e}")
        return None


def _patched_librosa_load(path, *args, **kwargs):
    """
    Patched 版本的 librosa.load

    对于非 WAV 文件，优先使用 pydub (ffmpeg) 加载
    如果 pydub 不可用或失败，回退到原始的 librosa.load
    """
    global _original_librosa_load

    if _original_librosa_load is None:
        return None

    if path.lower().endswith(".wav"):
        return _original_librosa_load(path, *args, **kwargs)

    logger.debug(f"[PATCH] 加载非 WAV 音频: {path}")

    result = _load_audio_with_pydub(
        path, sr=kwargs.get("sr"), mono=kwargs.get("mono", True)
    )

    if result is not None:
        audio, sr = result
        logger.info(f"✓ 使用 pydub (ffmpeg) 成功加载音频: {path}")
        return audio, sr

    logger.debug(f"回退到 librosa.load: {path}")
    return _original_librosa_load(path, *args, **kwargs)


def apply_librosa_patch():
    """
    应用 librosa.load patch

    将 librosa.load 替换为支持 pydub 的版本
    """
    global _original_librosa_load

    try:
        import librosa

        if _original_librosa_load is None:
            _original_librosa_load = librosa.load

        librosa.load = _patched_librosa_load

        logger.info(
            "[PATCH] ✓ 已应用 librosa.load patch（使用 pydub/ffmpeg 加载非 WAV 音频）"
        )

    except ImportError:
        logger.warning("[PATCH] ⚠ librosa 未安装，跳过 patch")
    except Exception as e:
        logger.error(f"[PATCH] ✗ 应用 librosa.patch 失败: {e}")


def get_original_librosa_load():
    """获取原始的 librosa.load 函数"""
    global _original_librosa_load
    return _original_librosa_load


def set_original_librosa_load(func):
    """设置原始的 librosa.load 函数"""
    global _original_librosa_load
    _original_librosa_load = func
