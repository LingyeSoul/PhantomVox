"""
音频拼装器模块

将多个生成的音频片段按时间轴拼装成完整音频
"""

import logging
import os
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from .srt_parser import ScheduledEntry

logger = logging.getLogger(__name__)


class AudioAssembler:
    """音频拼装器"""

    def __init__(
        self, sample_rate: int = 24000, allowed_dirs: Optional[List[str]] = None
    ):
        self.sample_rate = sample_rate
        if allowed_dirs is None:
            cwd = Path.cwd()
            self.allowed_dirs = [str(cwd), str(cwd / "temp"), str(cwd / "output")]
        else:
            self.allowed_dirs = [os.path.abspath(d) for d in allowed_dirs]

    def _validate_output_path(self, output_path: str) -> Path:
        """
        验证输出路径是否安全（防止路径遍历攻击）

        Args:
            output_path: 输出文件路径

        Returns:
            Path: 规范化后的绝对路径

        Raises:
            ValueError: 路径不在允许的目录内
        """
        path = Path(output_path).resolve()

        for allowed_dir in self.allowed_dirs:
            if str(path).startswith(str(allowed_dir)):
                return path

        raise ValueError(
            f"输出路径不允许: {output_path}\n允许的目录: {self.allowed_dirs}"
        )

    def assemble(
        self, scheduled_entries: List[ScheduledEntry], output_path: str
    ) -> str:
        """
        拼装音频

        流程：
        1. 验证输出路径
        2. 确定总时长
        3. 创建空白音频容器
        4. 按actual_start插入各段音频
        5. 填充静音间隙
        6. 导出文件

        Args:
            scheduled_entries: 调度后的字幕条目（包含音频数据）
            output_path: 输出文件路径

        Returns:
            str: 输出文件路径

        Raises:
            ValueError: 输出路径不允许或条目为空
        """
        validated_path = self._validate_output_path(output_path)

        if not scheduled_entries:
            raise ValueError("没有可拼装的音频条目")

        total_duration = self._get_total_duration(scheduled_entries)
        total_samples = int(total_duration * self.sample_rate)

        logger.info(f"创建音频容器: {total_duration:.2f}s ({total_samples} samples)")

        container = np.zeros(total_samples, dtype=np.float32)

        for entry in scheduled_entries:
            if entry.audio_data is None:
                logger.warning(f"字幕 {entry.entry.index} 没有音频数据，跳过")
                continue

            start_sample = int(entry.actual_start * self.sample_rate)
            audio = entry.audio_data

            if isinstance(audio, tuple):
                audio = audio[0]

            end_sample = start_sample + len(audio)

            if end_sample > len(container):
                logger.warning(
                    f"音频超出容器边界，截断: {end_sample} > {len(container)}"
                )
                audio = audio[: len(container) - start_sample]
                end_sample = len(container)

            container[start_sample:end_sample] = audio
            logger.debug(
                f"插入音频: 字幕{entry.entry.index} @ {entry.actual_start:.2f}s "
                f"({start_sample}-{end_sample})"
            )

        self._normalize_audio(container)

        import soundfile as sf

        sf.write(str(validated_path), container, self.sample_rate)

        logger.info(f"音频拼装完成: {validated_path}")
        return str(validated_path)

    def assemble_to_memory(self, scheduled_entries: List[ScheduledEntry]) -> np.ndarray:
        """
        拼装音频到内存（不保存文件）

        Args:
            scheduled_entries: 调度后的字幕条目

        Returns:
            np.ndarray: 拼装后的音频数据
        """
        if not scheduled_entries:
            return np.array([], dtype=np.float32)

        total_duration = self._get_total_duration(scheduled_entries)
        total_samples = int(total_duration * self.sample_rate)

        container = np.zeros(total_samples, dtype=np.float32)

        for entry in scheduled_entries:
            if entry.audio_data is None:
                continue

            start_sample = int(entry.actual_start * self.sample_rate)
            audio = entry.audio_data

            if isinstance(audio, tuple):
                audio = audio[0]

            end_sample = min(start_sample + len(audio), len(container))
            actual_len = end_sample - start_sample

            container[start_sample:end_sample] = audio[:actual_len]

        self._normalize_audio(container)

        return container

    def create_silence(self, duration: float) -> np.ndarray:
        """
        生成静音片段

        Args:
            duration: 静音时长（秒）

        Returns:
            np.ndarray: 静音音频数据
        """
        samples = int(duration * self.sample_rate)
        return np.zeros(samples, dtype=np.float32)

    def _get_total_duration(self, entries: List[ScheduledEntry]) -> float:
        """获取总时长"""
        if not entries:
            return 0.0
        return max(entry.actual_end for entry in entries)

    def _normalize_audio(self, audio: np.ndarray) -> None:
        """
        归一化音频到 [-1, 1] 范围

        Args:
            audio: 音频数据（原地修改）
        """
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            logger.info(f"音频归一化: max={max_val:.4f}")
            audio /= max_val
