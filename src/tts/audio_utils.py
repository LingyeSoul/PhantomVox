"""
音频处理工具模块

提供安全的音频加载、转换和验证功能。
所有音频文件操作都应通过此模块进行，以确保安全性和一致性。
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# 常量定义
QWEN_TTS_SAMPLE_RATE = 24000  # Qwen-TTS 标准采样率
MAX_AUDIO_FILE_SIZE = 100 * 1024 * 1024  # 音频文件最大大小：100 MB
MAX_AUDIO_DURATION_SECONDS = 600  # 音频最大时长：10 分钟
MAX_SAMPLE_RATE = 48000  # 最大采样率
MAX_CHANNELS = 2  # 最大声道数

# 允许的音频 MIME 类型
ALLOWED_AUDIO_MIMES = [
    'audio/mpeg',      # MP3
    'audio/wav',       # WAV
    'audio/x-wav',     # WAV (alternative)
    'audio/mp4',       # M4A
    'audio/x-m4a',     # M4A (alternative)
    'audio/ogg',       # OGG
    'audio/flac',      # FLAC
]

# 允许的音频文件扩展名
ALLOWED_AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'
}


class AudioValidationError(Exception):
    """音频验证失败异常"""
    pass


class AudioSecurityError(Exception):
    """音频安全相关异常"""
    pass


def validate_audio_path(
    audio_path: str,
    allowed_directories: Optional[List[str]] = None
) -> str:
    """
    验证音频文件路径是否安全（防止路径遍历攻击）

    Args:
        audio_path: 用户提供的音频文件路径
        allowed_directories: 允许的基础目录列表（None 表示跳过目录检查）

    Returns:
        str: 规范化后的绝对路径

    Raises:
        AudioSecurityError: 路径不安全或超出允许范围
    """
    if not audio_path:
        raise AudioSecurityError("音频路径不能为空")

    # 解析为绝对路径
    abs_path = os.path.abspath(audio_path)

    # 规范化路径，移除 ../ 等相对路径
    normalized_path = os.path.normpath(abs_path)

    # 如果提供了允许的目录列表，验证路径在这些目录内
    # 如果 allowed_directories 为 None，跳过目录检查（允许任意路径）
    if allowed_directories is not None:
        is_allowed = False
        for allowed_dir in allowed_directories:
            allowed_abs = os.path.abspath(allowed_dir)
            if normalized_path.startswith(allowed_abs + os.sep) or normalized_path == allowed_abs:
                is_allowed = True
                break

        if not is_allowed:
            raise AudioSecurityError(
                f"音频路径不在允许的目录内: {audio_path}\n"
                f"允许的目录: {allowed_directories}"
            )

    return normalized_path


def validate_file_exists_and_readable(file_path: str) -> None:
    """
    验证文件存在且可读

    Args:
        file_path: 文件路径

    Raises:
        FileNotFoundError: 文件不存在
        AudioSecurityError: 文件不可读
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    if not os.path.isfile(file_path):
        raise AudioSecurityError(f"路径不是文件: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise AudioSecurityError(f"文件不可读: {file_path}")


def validate_file_size(file_path: str) -> int:
    """
    验证文件大小是否在允许范围内

    Args:
        file_path: 文件路径

    Returns:
        int: 文件大小（字节）

    Raises:
        AudioValidationError: 文件为空或过大
    """
    file_size = os.path.getsize(file_path)

    if file_size == 0:
        raise AudioValidationError(f"音频文件为空: {file_path}")

    if file_size > MAX_AUDIO_FILE_SIZE:
        raise AudioValidationError(
            f"音频文件过大: {file_size / 1024 / 1024:.2f} MB "
            f"(最大允许 {MAX_AUDIO_FILE_SIZE / 1024 / 1024:.0f} MB)"
        )

    return file_size


def validate_audio_extension(file_path: str) -> None:
    """
    验证文件扩展名是否在允许列表中

    Args:
        file_path: 文件路径

    Raises:
        AudioValidationError: 文件扩展名不支持
    """
    ext = Path(file_path).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise AudioValidationError(
            f"不支持的音频格式: {ext}\n"
            f"支持的格式: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )


def load_audio_with_pydub(
    audio_path: str,
    sr: Optional[int] = None,
    mono: bool = True,
    validate_duration: bool = True
) -> Optional[Tuple[np.ndarray, int]]:
    """
    使用 pydub (ffmpeg) 加载音频数据

    Args:
        audio_path: 音频文件路径
        sr: 目标采样率（None 表示使用原采样率）
        mono: 是否转换为单声道
        validate_duration: 是否验证音频时长

    Returns:
        tuple: (音频数据, 采样率) 或 None（失败时）

    Raises:
        AudioValidationError: 音频参数验证失败
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.debug("pydub 未安装")
        return None

    try:
        logger.debug(f"使用 pydub 加载音频: {audio_path}")

        # 加载音频
        audio = AudioSegment.from_file(audio_path)

        # 验证音频属性
        if validate_duration:
            duration_seconds = len(audio) / 1000.0
            if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
                raise AudioValidationError(
                    f"音频时长过长: {duration_seconds / 60:.2f} 分钟 "
                    f"(最大允许 {MAX_AUDIO_DURATION_SECONDS / 60:.1f} 分钟)"
                )

        if audio.frame_rate > MAX_SAMPLE_RATE:
            raise AudioValidationError(
                f"采样率过高: {audio.frame_rate} Hz "
                f"(最大允许 {MAX_SAMPLE_RATE} Hz)"
            )

        if audio.channels > MAX_CHANNELS:
            raise AudioValidationError(
                f"声道数过多: {audio.channels} "
                f"(最大允许 {MAX_CHANNELS})"
            )

        # 记录原始音频信息
        original_sr = audio.frame_rate
        original_channels = audio.channels
        logger.debug(
            f"原始音频: sr={original_sr}, channels={original_channels}, "
            f"duration={len(audio)/1000:.2f}s"
        )

        # 设置输出格式
        if sr:
            audio = audio.set_frame_rate(sr)
        if mono:
            audio = audio.set_channels(1)

        # 转换为 numpy 数组并归一化
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # 根据实际位深度计算归一化因子
        sample_width = audio.sample_width

        # 验证 sample_width 范围
        if sample_width < 1 or sample_width > 4:
            raise AudioValidationError(
                f"不支持的音频位深度: sample_width={sample_width}"
            )

        # 归一化到 [-1, 1]
        if sample_width == 1:  # 8-bit unsigned
            samples = (samples - 128.0) / 128.0
        else:  # 有符号整数
            max_value = float(2 ** (8 * sample_width - 1))
            samples /= max_value

        logger.debug(
            f"音频位深度: {8*sample_width}-bit, "
            f"归一化后范围: [{samples.min():.3f}, {samples.max():.3f}]"
        )

        actual_sr = audio.frame_rate

        logger.info(
            f"✓ pydub 成功加载音频: {audio_path}, "
            f"shape={samples.shape}, sr={actual_sr}"
        )
        return samples, actual_sr

    except (FileNotFoundError, OSError, IOError) as e:
        logger.warning(f"音频文件读取失败: {type(e).__name__}: {e}")
        return None
    except RuntimeError as e:
        logger.warning(f"pydub 音频处理失败: {e}")
        return None


def convert_to_wav(
    source_path: str,
    target_path: str,
    sample_rate: int = QWEN_TTS_SAMPLE_RATE,
    allowed_directories: Optional[List[str]] = None
) -> None:
    """
    安全地将音频文件转换为 WAV 格式

    Args:
        source_path: 源音频文件路径
        target_path: 目标 WAV 文件路径
        sample_rate: 输出采样率
        allowed_directories: 允许的源文件目录列表（用于路径验证）

    Raises:
        AudioSecurityError: 路径不安全
        FileNotFoundError: 源文件不存在
        AudioValidationError: 音频验证失败
        RuntimeError: 转换失败
    """
    # 路径安全验证（allowed_directories=None 表示允许任意路径）
    safe_source_path = validate_audio_path(source_path, allowed_directories)

    # 文件存在性和可读性验证
    validate_file_exists_and_readable(safe_source_path)

    # 文件大小验证
    file_size = validate_file_size(safe_source_path)

    # 文件扩展名验证
    validate_audio_extension(safe_source_path)

    logger.info(f"开始转换音频: {safe_source_path} ({file_size / 1024:.2f} KB)")

    try:
        from pydub import AudioSegment

        # 加载音频（包含时长、采样率等验证）
        audio = AudioSegment.from_file(safe_source_path)

        # 验证音频时长
        duration_seconds = len(audio) / 1000.0
        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise AudioValidationError(
                f"音频时长过长: {duration_seconds / 60:.2f} 分钟"
            )

        # 验证采样率
        if audio.frame_rate > MAX_SAMPLE_RATE:
            raise AudioValidationError(
                f"采样率过高: {audio.frame_rate} Hz"
            )

        # 验证声道数
        if audio.channels > MAX_CHANNELS:
            raise AudioValidationError(
                f"声道数过多: {audio.channels}"
            )

        # 转换为目标格式
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(1)

        # 确保目标目录存在
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        # 导出到临时文件，然后原子性重命名
        temp_path = target_path + '.tmp'
        audio.export(temp_path, format="wav")

        # 原子性重命名
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)

        logger.info(f"✓ 音频转换成功: {target_path}")

    except ImportError:
        raise RuntimeError(
            "pydub 未安装。请安装: pip install pydub"
        )
    except Exception as e:
        temp_path = target_path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        raise RuntimeError(f"音频转换失败: {e}")


def normalize_audio_samples(audio) -> np.ndarray:
    """
    归一化 AudioSegment 样本到 [-1, 1] 范围

    Args:
        audio: pydub AudioSegment 对象

    Returns:
        np.ndarray: 归一化后的音频样本
    """
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_width = audio.sample_width

    # 验证 sample_width
    if sample_width < 1 or sample_width > 4:
        raise ValueError(f"不支持的 sample_width: {sample_width}")

    if sample_width == 1:  # 8-bit unsigned
        samples = (samples - 128.0) / 128.0
    else:  # 有符号整数
        max_value = float(2 ** (8 * sample_width - 1))
        samples /= max_value

    return samples
