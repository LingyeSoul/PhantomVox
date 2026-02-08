"""
音频临时文件管理器

负责创建、管理和清理音频临时文件
"""

import os
import logging
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AudioTempManager:
    """音频临时文件管理器"""

    def __init__(self, project_root: str = None):
        """
        初始化临时文件管理器

        Args:
            project_root: 项目根目录，如果为 None 则自动检测
        """
        if project_root is None:
            # 自动检测项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent

        self.project_root = Path(project_root)
        self.temp_dir = self.project_root / "temp"

        # 确保 temp 目录存在
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"音频临时文件目录: {self.temp_dir}")

    def get_temp_path(self, prefix: str = "audio") -> str:
        """
        获取一个新的临时文件路径

        Args:
            prefix: 文件名前缀

        Returns:
            str: 临时文件的完整路径
        """
        import time
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        filename = f"{prefix}_{timestamp}.wav"
        return str(self.temp_dir / filename)

    def save_audio(self, audio_data: np.ndarray, sample_rate: int, prefix: str = "audio") -> str:
        """
        保存音频到临时文件

        Args:
            audio_data: 音频数据 (numpy array)
            sample_rate: 采样率
            prefix: 文件名前缀

        Returns:
            str: 临时文件的完整路径
        """
        temp_path = self.get_temp_path(prefix)

        try:
            # 标准化音频数据
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            # 保存为 WAV 文件
            sf.write(temp_path, audio_data, sample_rate)

            logger.info(f"✓ 音频已保存到临时文件: {Path(temp_path).name}")
            return temp_path

        except Exception as e:
            logger.error(f"✗ 保存临时文件失败: {str(e)}")
            raise

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        从临时文件加载音频

        Args:
            file_path: 音频文件路径

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        try:
            audio_data, sr = sf.read(file_path, dtype='float32')

            # 如果是立体声，转换为单声道
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            logger.info(f"✓ 从文件加载音频: {Path(file_path).name}, 采样率: {sr}")
            return audio_data, sr

        except Exception as e:
            logger.error(f"✗ 加载音频文件失败: {str(e)}")
            raise

    def cleanup_file(self, file_path: str) -> bool:
        """
        删除指定的临时文件

        Args:
            file_path: 要删除的文件路径

        Returns:
            bool: 是否成功删除
        """
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"已删除临时文件: {Path(file_path).name}")
                return True
            return False
        except Exception as e:
            logger.error(f"✗ 删除临时文件失败: {str(e)}")
            return False

    def cleanup_all(self) -> int:
        """
        清理所有临时音频文件

        Returns:
            int: 成功删除的文件数量
        """
        count = 0
        try:
            for file_path in self.temp_dir.glob("*.wav"):
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"删除文件 {file_path.name} 失败: {e}")

            logger.info(f"✓ 已清理 {count} 个临时音频文件")
            return count

        except Exception as e:
            logger.error(f"✗ 清理临时文件失败: {str(e)}")
            return count

    def get_temp_dir(self) -> Path:
        """获取临时文件目录"""
        return self.temp_dir

    def file_exists(self, file_path: str) -> bool:
        """检查临时文件是否存在"""
        return file_path and os.path.exists(file_path)

    def save_to_persistent(self, temp_file_path: str, save_dir: str, prefix: str = "audio", custom_filename: str = None) -> str:
        """
        将临时文件保存到持久化目录

        Args:
            temp_file_path: 临时文件路径
            save_dir: 保存目录
            prefix: 文件名前缀
            custom_filename: 自定义文件名（不含扩展名），如果为 None 则使用时间戳

        Returns:
            str: 保存后的文件完整路径

        Raises:
            ValueError: 如果临时文件不存在
            OSError: 如果保存失败
        """
        if not temp_file_path or not os.path.exists(temp_file_path):
            raise ValueError(f"临时文件不存在: {temp_file_path}")

        import shutil
        from datetime import datetime

        # 使用 Path 对象规范化路径，确保斜杠方向一致
        save_path = Path(save_dir)

        # 确保保存目录存在
        save_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        if custom_filename:
            # 使用自定义文件名，确保有 .wav 扩展名
            if not custom_filename.endswith('.wav'):
                custom_filename += '.wav'
            filename = custom_filename
        else:
            # 使用默认的时间戳文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.wav"

        # 使用 Path 构建完整路径，自动处理斜杠
        full_path = save_path / filename

        # 复制临时文件到目标位置
        shutil.copy2(temp_file_path, str(full_path))

        # 返回规范化的路径字符串
        normalized_path = str(full_path)
        logger.info(f"✓ 音频已保存到: {normalized_path}")
        return normalized_path
