"""
音频播放和管理模块

提供音频播放、停止、保存、进度显示等功能
使用 sounddevice 进行音频播放，支持进度条显示和拖动跳转
"""

import soundfile as sf
import logging
import numpy as np
import threading
import time
import asyncio
from pathlib import Path
from typing import Optional, Callable, Tuple
from enum import Enum
from api.constants import SAMPLE_RATE


import flet as ft

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class AudioState(Enum):
    """音频状态枚举"""
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


class AudioManager:
    """音频播放和管理 - 基于 sounddevice 实现"""

    def __init__(self, page: ft.Page):
        """
        初始化音频管理器

        Args:
            page: Flet Page 实例
        """
        self.page = page
        self.current_audio_data = None
        self.sample_rate = SAMPLE_RATE  # qwen-tts 默认采样率

        # sounddevice 相关状态
        self._stream: Optional[sd.OutputStream] = None
        self._audio_data: Optional[np.ndarray] = None
        self._current_frame: int = 0
        self._total_frames: int = 0
        self._is_playing: bool = False
        self._is_paused: bool = False

        # 进度更新相关
        self._progress_callback: Optional[Callable] = None
        self._update_thread: Optional[threading.Thread] = None
        self._stop_update_thread = threading.Event()
        self._completion_callback: Optional[Callable] = None

        # 音量控制
        self._volume: float = 1.0

    # ==================== 播放控制 ====================

    async def _cleanup_playback(self):
        """完全清理播放资源"""
        try:
            # 停止并关闭音频流
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.debug(f"关闭音频流时出错: {e}")
                finally:
                    self._stream = None

            # 停止进度更新线程
            self._stop_update_thread.set()
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=0.5)
                if self._update_thread.is_alive():
                    logger.warning("进度更新线程未能在超时时间内停止")

            # 重置状态
            self._is_playing = False
            self._is_paused = False
            self._current_frame = 0

            logger.debug("✓ 播放资源已清理")

        except Exception as e:
            logger.error(f"✗ 清理播放资源失败: {str(e)}")

    async def play_from_file(self, file_path: str, position: float = 0):
        """
        从文件播放音频

        Args:
            file_path: 音频文件路径
            position: 开始播放的位置（秒），默认为 0
        """
        try:
            # 读取音频文件
            audio_data, sr = sf.read(file_path, dtype='float32')

            # 如果是立体声，转换为单声道
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)

            logger.info(f"从文件加载音频: {file_path}, 采样率: {sr}")

            # 调用 play 方法播放
            await self.play(audio_data, sr, position)

        except Exception as e:
            logger.error(f"✗ 从文件播放失败: {str(e)}")
            raise

    async def play(self, audio_data, sample_rate: int = None, position: float = 0):
        """
        播放音频

        Args:
            audio_data: numpy array 格式的音频数据
            sample_rate: 采样率（如果为 None，使用默认的 24000）
            position: 开始播放的位置（秒），默认为 0
        """
        try:
            if audio_data is None:
                raise ValueError("音频数据为空")

            # 如果正在播放，先完全停止并清理
            if self._is_playing or self._stream is not None:
                await self._cleanup_playback()

            # 设置采样率
            if sample_rate is not None:
                self.sample_rate = sample_rate

            logger.info("正在播放音频...")

            # 标准化音频数据
            self._audio_data = self._normalize_audio_data(audio_data)
            self._total_frames = len(self._audio_data)

            # 计算起始位置
            start_frame = int(position * self.sample_rate)
            start_frame = max(0, min(start_frame, self._total_frames - 1))
            self._current_frame = start_frame

            # 重置停止事件（必须在启动新线程前清除）
            self._stop_update_thread.clear()

            # 创建音频流
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                finished_callback=self._on_playback_finished
            )

            # 启动播放
            self._stream.start()
            self._is_playing = True
            self._is_paused = False

            # 启动进度更新线程
            self._start_progress_updates()

            logger.info("✓ 音频开始播放")

        except Exception as e:
            logger.error(f"✗ 音频播放失败: {str(e)}")
            self._is_playing = False
            raise

    async def stop(self):
        """停止播放"""
        try:
            # 使用统一的清理方法
            await self._cleanup_playback()

            # 重置进度条
            if self._progress_callback:
                duration = self._total_frames / self.sample_rate if self._total_frames > 0 else 0.0
                async def _reset():
                    try:
                        await self._progress_callback(0.0, 0.0, duration)
                    except Exception as e:
                        logger.error(f"重置进度回调失败: {e}")

                # 检查页面会话是否有效
                if hasattr(self.page, 'session') and self.page.session:
                    try:
                        self.page.run_task(_reset)
                    except RuntimeError:
                        # 页面会话可能已被销毁，忽略错误
                        pass

            logger.info("✓ 音频播放已停止")

        except Exception as e:
            logger.error(f"✗ 停止播放失败: {str(e)}")

    async def pause(self):
        """暂停播放"""
        try:
            if self._is_playing and not self._is_paused and self._stream:
                self._stream.stop()
                self._is_paused = True
                logger.info("✓ 音频已暂停")

        except Exception as e:
            logger.error(f"✗ 暂停失败: {str(e)}")

    async def resume(self):
        """恢复播放"""
        try:
            if self._is_paused and self._stream:
                self._stream.start()
                self._is_paused = False
                logger.info("✓ 音频已恢复播放")

        except Exception as e:
            logger.error(f"✗ 恢复播放失败: {str(e)}")

    async def seek(self, position_seconds: float):
        """
        跳转到指定位置

        Args:
            position_seconds: 跳转位置（秒）
        """
        try:
            if self._audio_data is None:
                return

            # 计算目标帧
            target_frame = int(position_seconds * self.sample_rate)
            target_frame = max(0, min(target_frame, self._total_frames - 1))

            # 保存当前状态
            was_paused = self._is_paused

            if self._is_playing:
                # 停止并关闭当前流
                if self._stream:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None

                # 更新当前位置
                self._current_frame = target_frame

                # 重新创建流
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=self._audio_callback,
                    finished_callback=self._on_playback_finished
                )

                # 启动流
                self._stream.start()

                # 如果之前是暂停状态，需要手动暂停
                if was_paused:
                    # 使用 asyncio.sleep 确保流启动完成
                    await asyncio.sleep(0.01)
                    self._stream.stop()
                    self._is_paused = True

                logger.info(f"✓ 已跳转到: {position_seconds:.2f}秒")
            else:
                # 如果没有播放，只更新位置
                self._current_frame = target_frame

            # 立即更新进度显示
            self._update_progress_ui()

        except Exception as e:
            logger.error(f"✗ 跳转失败: {str(e)}")

    async def set_volume(self, volume: float):
        """
        设置音量

        Args:
            volume: 音量值 (0.0 - 1.0)
        """
        self._volume = max(0.0, min(1.0, volume))
        logger.info(f"✓ 音量已设置为: {self._volume:.2f}")

    # ==================== 进度更新 ====================

    def set_progress_callback(self, callback: Callable):
        """
        设置进度更新回调函数

        Args:
            callback: 回调函数，接收 (progress, current, total) 参数
        """
        self._progress_callback = callback

    def set_completion_callback(self, callback: Callable):
        """
        设置播放完成回调函数

        Args:
            callback: 播放完成时的回调函数
        """
        self._completion_callback = callback

    def get_progress(self) -> Tuple[float, float, float]:
        """
        获取当前播放进度

        Returns:
            Tuple[float, float, float]: (progress, current_position, duration)
        """
        if self._audio_data is None:
            return (0.0, 0.0, 0.0)

        duration = self._total_frames / self.sample_rate
        current = self._current_frame / self.sample_rate
        progress = self._current_frame / self._total_frames

        return (progress, current, duration)

    def _start_progress_updates(self):
        """启动进度更新线程"""
        # 确保停止旧线程
        if self._update_thread and self._update_thread.is_alive():
            self._stop_update_thread.set()
            self._update_thread.join(timeout=0.5)

        # 清除停止事件
        self._stop_update_thread.clear()

        # 创建并启动新线程
        self._update_thread = threading.Thread(
            target=self._update_progress_loop,
            daemon=True
        )
        self._update_thread.start()

    def _update_progress_loop(self):
        """进度更新循环（在后台线程中运行）"""
        while not self._stop_update_thread.is_set() and self._is_playing:
            if not self._is_paused:
                self._update_progress_ui()

            # 每 100ms 更新一次
            time.sleep(0.1)

    def _update_progress_ui(self):
        """更新进度 UI（线程安全）"""
        if self._progress_callback and self._audio_data is not None:
            progress, current, duration = self.get_progress()
            # 使用 run_task 确保在 UI 线程中更新
            async def _update():
                await self._progress_callback(progress, current, duration)
            
            # 检查页面会话是否有效
            if hasattr(self.page, 'session') and self.page.session:
                try:
                    self.page.run_task(_update)
                except RuntimeError:
                    # 页面会话可能已被销毁，忽略错误
                    pass

    # ==================== 音频回调 ====================

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        sounddevice 音频回调函数

        Args:
            outdata: 输出数据缓冲区 (frames, channels)
            frames: 需要填充的帧数
            time_info: 时间信息
            status: 状态信息
        """
        if status:
            logger.warning(f"音频回调状态: {status}")

        # 检查是否暂停或停止
        if not self._is_playing or self._is_paused:
            # 填充零数据（静音）
            outdata.fill(0)
            return

        # 计算需要读取的数据长度
        end_idx = min(self._current_frame + frames, len(self._audio_data))

        # 获取当前片段
        chunk = self._audio_data[self._current_frame:end_idx]

        # 如果片段长度小于所需帧数，用零填充
        if len(chunk) < frames:
            if chunk.ndim == 1:
                chunk = np.pad(chunk, (0, frames - len(chunk)), mode='constant')
            else:
                chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)), mode='constant')

        # 将数据复制到输出缓冲区
        outdata[:, 0] = chunk[:frames]

        # 更新播放位置
        self._current_frame += frames

        # 检查是否播放完毕（只在第一次检测到时触发）
        if self._current_frame >= len(self._audio_data) and self._is_playing:
            # 播放完成
            self._is_playing = False
            self._is_paused = False
            self._stop_update_thread.set()

            logger.info("✓ 音频播放完成")

            # 调用完成回调
            if self._completion_callback:
                async def _on_complete():
                    try:
                        await self._completion_callback()
                    except Exception as e:
                        logger.error(f"完成回调执行失败: {e}")

                # 检查页面会话是否有效
                if hasattr(self.page, 'session') and self.page.session:
                    try:
                        self.page.run_task(_on_complete)
                    except RuntimeError:
                        # 页面会话可能已被销毁，忽略错误
                        logger.debug("页面会话已销毁，忽略完成回调")

    def _on_playback_finished(self):
        """播放完成回调"""
        if self._is_playing:
            self._is_playing = False
            self._is_paused = False
            self._stop_update_thread.set()

            logger.info("✓ 音频播放完成")

            # 调用完成回调
            if self._completion_callback:
                async def _on_complete():
                    await self._completion_callback()

                # 检查页面会话是否有效
                if hasattr(self.page, 'session') and self.page.session:
                    try:
                        self.page.run_task(_on_complete)
                    except RuntimeError:
                        # 页面会话可能已被销毁，忽略错误
                        pass

    # ==================== 音频数据处理 ====================

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
            # 同时更新采样率
            if len(audio_data) > 1:
                self.sample_rate = audio_data[1]

        # 检查是否是列表
        if isinstance(audio_data, list):
            logger.warning("接收到列表，自动转换为 numpy array")
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

    def _get_project_root(self) -> str:
        """获取项目根目录"""
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        return str(project_root)

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

        try:
            normalized_data = self._normalize_audio_data(audio_data)
            return len(normalized_data) / self.sample_rate
        except Exception as e:
            logger.warning(f"计算音频时长失败: {e}")
            return 0.0

    # ==================== 状态查询 ====================

    def is_playing(self) -> bool:
        """检查是否正在播放"""
        return self._is_playing and not self._is_paused

    def is_paused(self) -> bool:
        """检查是否已暂停"""
        return self._is_paused

    def get_state(self) -> AudioState:
        """获取当前音频状态"""
        if not self._is_playing:
            return AudioState.STOPPED
        elif self._is_paused:
            return AudioState.PAUSED
        else:
            return AudioState.PLAYING

    async def cleanup(self):
        """清理资源"""
        try:
            await self.stop()
            self._audio_data = None
            self._progress_callback = None
            self._completion_callback = None
        except Exception as e:
            logger.error(f"✗ 清理资源失败: {str(e)}")
