"""
音频临时文件管理器

负责创建、管理和清理音频临时文件
"""

import os
import logging
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np

logger = logging.getLogger(__name__)

# 支持的音频输出格式
AudioFormat = Literal["wav", "mp3", "ogg"]
SUPPORTED_FORMATS = {"wav", "mp3", "ogg"}


class AudioTempManager:
    """音频临时文件管理器"""

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            current_file = Path(__file__).resolve()
            project_root = str(current_file.parent.parent.parent)

        self.project_root = Path(project_root)
        self.temp_dir = self.project_root / "temp"

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
        return bool(file_path and os.path.exists(file_path))

    def convert_audio_format(
        self,
        source_path: str,
        target_path: str,
        target_format: AudioFormat
    ) -> str:
        if target_format not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的音频格式: {target_format}，支持: {SUPPORTED_FORMATS}")

        if not os.path.exists(source_path):
            raise ValueError(f"源文件不存在: {source_path}")

        if target_format == "wav":
            import shutil
            full_path = f"{target_path}.wav"
            shutil.copy2(source_path, full_path)
            return full_path

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(source_path)
            full_path = f"{target_path}.{target_format}"

            format_kwargs = {"format": target_format}
            if target_format == "mp3":
                format_kwargs["bitrate"] = "192k"

            audio.export(full_path, **format_kwargs)
            logger.info(f"✓ 音频格式转换成功: {Path(full_path).name}")
            return full_path

        except ImportError:
            raise RuntimeError("pydub 未安装，无法进行格式转换。请安装: pip install pydub")
        except Exception as e:
            raise RuntimeError(f"音频格式转换失败: {str(e)}")

    def save_audio_to_format(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        save_dir: str,
        prefix: str = "audio",
        output_format: AudioFormat = "wav",
        target_path: Optional[str] = None
    ) -> str:
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        save_dir: str,
        prefix: str = "audio",
        output_format: AudioFormat = "wav"
    ) -> str:
        if target_path:
            target_file = target_path
            actual_format = Path(target_file).suffix.lstrip('.')
        else:
        from datetime import datetime
                logger.warning(f"不支持的格式 {output_format}，使用默认 wav")
            output_format = "wav"
        save_path.mkdir(parents=True, exist_ok=True)
            filename_base = f"{prefix}_{timestamp}"
            target_file = str(save_path / f"{filename_base}.{output_format}")
            actual_format = output_format

        full_path_no_ext = str(Path(target_file).with_suffix(''))
        temp_wav = full_path_no_ext + ".wav"

        try:
            sf.write(temp_wav, audio_data, sample_rate)
            if actual_format == "wav":
                return temp_wav
            else:
                return self.convert_audio_format(temp_wav, full_path_no_ext, actual_format)
        finally:
            if actual_format != "wav" and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def get_persistent_target_path(
        self,
        save_dir: str,
        prefix: str = "audio",
        custom_filename: Optional[str] = None,
        output_format: AudioFormat = "wav"
    ) -> str:
        """
        获取持久化保存的目标文件路径（不实际保存）

        用于在保存前检查文件是否已存在

        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
            custom_filename: 自定义文件名（不含扩展名）
            output_format: 输出格式 (wav/mp3/ogg)

        Returns:
            str: 目标文件的完整路径
        """
        from datetime import datetime

        if output_format not in SUPPORTED_FORMATS:
            output_format = "wav"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if custom_filename:
            filename_base = custom_filename
            if filename_base.endswith(('.wav', '.mp3', '.ogg')):
                filename_base = Path(filename_base).stem
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{prefix}_{timestamp}"

        return str(save_path / f"{filename_base}.{output_format}")

    def get_audio_to_format_target_path(
        self,
        save_dir: str,
        prefix: str = "audio",
        output_format: AudioFormat = "wav"
    ) -> str:
        """
        获取 save_audio_to_format 的目标文件路径（不实际保存）

        Args:
            save_dir: 保存目录
            prefix: 文件名前缀
            output_format: 输出格式

        Returns:
            str: 目标文件的完整路径
        """
        from datetime import datetime

        if output_format not in SUPPORTED_FORMATS:
            output_format = "wav"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{prefix}_{timestamp}"

        return str(save_path / f"{filename_base}.{output_format}")
    def save_to_persistent(
        self,
        temp_file_path: str,
        save_dir: str,
        prefix: str = "audio",
        custom_filename: Optional[str] = None,
        output_format: AudioFormat = "wav",
        target_path: Optional[str] = None
    ) -> str:
        self,
        temp_file_path: str,
        save_dir: str,
        prefix: str = "audio",
        custom_filename: Optional[str] = None,
        output_format: AudioFormat = "wav"
    ) -> str:
        """
        将临时文件保存到持久化目录

        Args:
            temp_file_path: 临时文件路径
            save_dir: 保存目录
            prefix: 文件名前缀
            custom_filename: 自定义文件名（不含扩展名），如果为 None 则使用时间戳
            output_format: 输出格式 (wav/mp3/ogg)

        Returns:
            str: 保存后的文件完整路径

        Raises:
            ValueError: 如果临时文件不存在或格式不支持
            OSError: 如果保存失败
        """
        if not temp_file_path or not os.path.exists(temp_file_path):
            raise ValueError(f"临时文件不存在: {temp_file_path}")
        if target_path:
            target_file = target_path
        else:
            if output_format not in SUPPORTED_FORMATS:
                logger.warning(f"不支持的格式 {output_format}，使用默认 wav")
            output_format = "wav"
            from datetime import datetime
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
                filename_base = custom_filename
                if filename_base.endswith(('.wav', '.mp3', '.ogg')):
                    filename_base = Path(filename_base).stem
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{prefix}_{timestamp}"
            target_file = str(save_path / f"{filename_base}.{output_format}")

        actual_format = Path(target_file).suffix.lstrip('.')
        full_path_no_ext = str(Path(target_file).with_suffix(''))

        if actual_format == "wav":
            import shutil
            shutil.copy2(temp_file_path, target_file)
            logger.info(f"✓ 音频已保存到: {target_file}")
            return target_file
        else:
            return self.convert_audio_format(temp_file_path, full_path_no_ext, actual_format)
