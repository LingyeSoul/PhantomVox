"""
音频播放和管理模块

提供音频播放、停止、保存等功能
使用 Flet Audio 控制进行音频播放
"""

import soundfile as sf
import logging
import numpy as np
import os
from pathlib import Path
from typing import Optional

import flet as ft
import flet_audio as fta

logger = logging.getLogger(__name__)


class AudioManager:
    """音频播放和管理 - 基于 Flet Audio"""

    def __init__(self, page: ft.Page):
        """
        初始化音频管理器

        Args:
            page: Flet Page 实例，用于添加 Audio 控件
        """
        self.page = page
        self.current_audio = None
        self.sample_rate = 24000  # qwen-tts 默认采样率
        self._audio_control: Optional[fta.Audio] = None
        self._temp_audio_file: Optional[str] = None
        self._is_playing = False
        self._is_paused = False

    def _get_audio_control(self) -> fta.Audio:
        """获取或创建 Audio 控件"""
        if self._audio_control is None:
            self._audio_control = fta.Audio(
                autoplay=False,
                volume=1.0,
                balance=0.0,
                release_mode=fta.ReleaseMode.STOP,
                on_loaded=lambda _: logger.info("✓ 音频已加载"),
                on_state_change=lambda e: self._on_state_change(e),
                on_position_change=lambda e: self._on_position_change(e),
            )
            # 将控件添加到页面（不可见）
            self.page.add(self._audio_control)
        return self._audio_control

    def _on_state_change(self, e):
        """音频状态变化回调"""
        logger.info(f"音频状态: {e.state}")
        if e.state in ["stopped", "completed"]:
            self._is_playing = False
            self._is_paused = False

    def _on_position_change(self, e):
        """音频播放位置变化回调"""
        pass  # 可用于更新进度条

    async def play(self, audio_data):
        """
        播放音频

        Args:
            audio_data: numpy array 格式的音频数据
        """
        try:
            if audio_data is None:
                raise ValueError("音频数据为空")

            logger.info("正在播放音频...")

            # 保存当前音频数据
            self.current_audio = audio_data

            # 将音频数据保存到临时文件
            temp_file = self._save_to_temp_file(audio_data)
            self._temp_audio_file = temp_file

            # 获取 Audio 控件并设置源
            audio = self._get_audio_control()
            audio.src = temp_file

            # 播放音频
            await audio.play()
            self._is_playing = True
            self._is_paused = False

            logger.info("✓ 音频开始播放")

        except Exception as e:
            logger.error(f"✗ 音频播放失败: {str(e)}")
            self._is_playing = False
            raise

    async def stop(self):
        """停止播放"""
        try:
            if self._is_playing or self._is_paused:
                audio = self._get_audio_control()
                await audio.release()
                self._is_playing = False
                self._is_paused = False
                logger.info("✓ 音频播放已停止")

        except Exception as e:
            logger.error(f"✗ 停止播放失败: {str(e)}")

    async def pause(self):
        """暂停播放"""
        try:
            if self._is_playing and not self._is_paused:
                audio = self._get_audio_control()
                await audio.pause()
                self._is_paused = True
                logger.info("✓ 音频已暂停")

        except Exception as e:
            logger.error(f"✗ 暂停失败: {str(e)}")

    async def resume(self):
        """恢复播放"""
        try:
            if self._is_paused:
                audio = self._get_audio_control()
                await audio.resume()
                self._is_paused = False
                logger.info("✓ 音频已恢复播放")

        except Exception as e:
            logger.error(f"✗ 恢复播放失败: {str(e)}")

    async def set_volume(self, volume: float):
        """
        设置音量

        Args:
            volume: 音量值 (0.0 - 1.0)
        """
        try:
            audio = self._get_audio_control()
            audio.volume = max(0.0, min(1.0, volume))
            logger.info(f"✓ 音量已设置为: {audio.volume:.2f}")

        except Exception as e:
            logger.error(f"✗ 设置音量失败: {str(e)}")

    async def seek(self, position_seconds: float):
        """
        跳转到指定位置

        Args:
            position_seconds: 跳转位置（秒）
        """
        try:
            audio = self._get_audio_control()
            await audio.seek(ft.Duration(seconds=position_seconds))
            logger.info(f"✓ 已跳转到: {position_seconds:.2f}秒")

        except Exception as e:
            logger.error(f"✗ 跳转失败: {str(e)}")

    def _get_project_root(self) -> str:
        """
        获取项目根目录

        Returns:
            str: 项目根目录路径
        """
        # 从当前文件向上两级到项目根目录 (src/tts/audio_manager.py -> src/ -> project_root/)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        return str(project_root)

    def _get_temp_dir(self) -> str:
        """
        获取或创建项目临时文件夹

        Returns:
            str: 临时文件夹路径
        """
        temp_dir = os.path.join(self._get_project_root(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _save_to_temp_file(self, audio_data) -> str:
        """
        将音频数据保存到临时文件

        Args:
            audio_data: numpy array 格式的音频数据

        Returns:
            str: 临时文件路径
        """
        try:
            # 获取项目临时文件夹
            temp_dir = self._get_temp_dir()
            temp_file = os.path.join(temp_dir, f"phantomvox_audio_{id(audio_data)}.wav")

            # 数据格式检查和转换
            audio_data = self._normalize_audio_data(audio_data)

            # 保存音频
            sf.write(temp_file, audio_data, self.sample_rate)

            logger.info(f"✓ 临时音频文件已创建: {temp_file}")
            return temp_file

        except Exception as e:
            logger.error(f"✗ 创建临时文件失败: {str(e)}")
            raise

    def _normalize_audio_data(self, audio_data):
        """
        标准化音频数据格式

        Args:
            audio_data: 可能是 numpy array、tuple 或其他格式

        Returns:
            numpy.ndarray: 标准化后的单通道音频数据
        """
        # 检查是否是元组 (audio_data, sample_rate)
        if isinstance(audio_data, tuple):
            logger.warning("接收到 (audio_data, sample_rate) 元组，自动提取 audio_data")
            audio_data = audio_data[0]

        # 检查是否是列表
        if isinstance(audio_data, list):
            logger.warning(f"接收到列表，自动转换为 numpy array")
            audio_data = np.array(audio_data)

        # 确保是 numpy 数组
        if not isinstance(audio_data, np.ndarray):
            raise ValueError(f"不支持的音频数据类型: {type(audio_data)}，期望 numpy.ndarray")

        # 记录原始形状
        original_shape = audio_data.shape
        logger.debug(f"音频数据原始形状: {original_shape}, dtype: {audio_data.dtype}")

        # 处理多维数组
        if len(audio_data.shape) > 1:
            # 如果是立体声或多声道，转换为单声道
            if audio_data.shape[1] > 1:
                logger.info(f"检测到 {audio_data.shape[1]} 声道音频，转换为单声道")
                audio_data = np.mean(audio_data, axis=1)
            else:
                # 形状是 (N, 1)，压缩为 (N,)
                audio_data = audio_data.flatten()

        # 确保是一维数组
        audio_data = audio_data.flatten()

        # 转换数据类型为 float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # 归一化到 [-1, 1] 范围
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            logger.warning(f"音频数据超过 ±1 范围 (max: {max_val:.4f})，进行归一化")
            audio_data = audio_data / max_val

        logger.debug(f"音频数据标准化后形状: {audio_data.shape}, dtype: {audio_data.dtype}")
        return audio_data

    def save(self, audio_data, filename):
        """
        保存音频到文件

        Args:
            audio_data: numpy array 格式的音频数据
            filename: 保存的文件路径
        """
        try:
            if audio_data is None:
                raise ValueError("音频数据为空")

            # 数据格式检查和转换
            audio_data = self._normalize_audio_data(audio_data)

            # 保存音频
            sf.write(filename, audio_data, self.sample_rate)
            logger.info(f"✓ 音频已保存到: {filename}")

        except Exception as e:
            logger.error(f"✗ 保存音频失败: {str(e)}")
            raise

    def get_audio_duration(self, audio_data) -> float:
        """
        获取音频时长（秒）

        Args:
            audio_data: numpy array 格式的音频数据

        Returns:
            float: 音频时长（秒）
        """
        if audio_data is None:
            return 0.0

        # 标准化后获取长度
        try:
            normalized_data = self._normalize_audio_data(audio_data)
            return len(normalized_data) / self.sample_rate
        except:
            return 0.0

    def is_playing(self) -> bool:
        """
        检查是否正在播放

        Returns:
            bool: 是否正在播放
        """
        return self._is_playing and not self._is_paused

    async def cleanup(self):
        """清理资源"""
        try:
            # 停止播放
            if self._is_playing:
                await self.stop()

            # 清理当前临时文件
            if self._temp_audio_file and os.path.exists(self._temp_audio_file):
                try:
                    os.remove(self._temp_audio_file)
                    logger.info(f"✓ 临时文件已删除: {self._temp_audio_file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {str(e)}")

            self._temp_audio_file = None

        except Exception as e:
            logger.error(f"✗ 清理资源失败: {str(e)}")

    def clear_temp_folder(self):
        """
        清理 temp 文件夹中的所有临时音频文件

        可以在程序启动时调用此方法，清理上次运行留下的临时文件
        """
        try:
            temp_dir = self._get_temp_dir()
            if not os.path.exists(temp_dir):
                return

            # 删除所有 .wav 文件
            files_removed = 0
            for filename in os.listdir(temp_dir):
                if filename.endswith('.wav') and filename.startswith('phantomvox_audio_'):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        os.remove(file_path)
                        files_removed += 1
                    except Exception as e:
                        logger.warning(f"删除文件 {filename} 失败: {str(e)}")

            if files_removed > 0:
                logger.info(f"✓ 已清理 {files_removed} 个临时音频文件")

        except Exception as e:
            logger.error(f"✗ 清理临时文件夹失败: {str(e)}")
