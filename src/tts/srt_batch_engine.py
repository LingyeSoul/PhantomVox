"""
SRT批量推理引擎

支持三种TTS模式的顺序非流式批量推理
"""

import asyncio
import logging
import torch  # ⭐ 导入torch用于显存监控
from typing import Optional, Callable
from dataclasses import dataclass

from .srt_parser import SRTParser, SRTEntry, ScheduledEntry
from .srt_config_models import (
    CustomVoiceConfig,
    VoiceDesignConfig,
    VoiceCloneConfig,
    SRTConfig,
)
from .timeline_scheduler import TimelineScheduler
from .audio_assembler import AudioAssembler
from .base_engine import BaseTTSEngine

logger = logging.getLogger(__name__)

# 常量定义
GENERATION_TIMEOUT_SECONDS = 300.0  # 5分钟超时
GPU_CLEANUP_INTERVAL = 10  # 每10条清理一次显存


@dataclass
class SRTBatchResult:
    """批量推理结果"""

    success: bool
    output_path: str
    total_entries: int
    generated_count: int
    failed_count: int
    total_duration: float
    adjustment_summary: dict
    error_message: str = ""


class SRTBatchEngine:
    """
    SRT批量推理引擎（支持三模式）

    核心特性：
    - 顺序非流式生成（保证质量）
    - 支持 CustomVoice/VoiceDesign/VoiceClone 三种模式
    - 自动时间轴调整和音频拼装
    - 进度报告和错误恢复
    """

    MODE_CUSTOM_VOICE = "custom_voice"
    MODE_VOICE_DESIGN = "voice_design"
    MODE_VOICE_CLONE = "voice_clone"

    def __init__(self, tts_engine: BaseTTSEngine):
        self.tts_engine = tts_engine
        self.parser = SRTParser()
        self.scheduler = TimelineScheduler()
        self.assembler = AudioAssembler()
        self.mode = self.MODE_VOICE_CLONE
        self.config: SRTConfig = VoiceCloneConfig()

    def set_mode(self, mode: str, config: SRTConfig):
        """设置模式和配置"""
        if mode not in (
            self.MODE_CUSTOM_VOICE,
            self.MODE_VOICE_DESIGN,
            self.MODE_VOICE_CLONE,
        ):
            raise ValueError(f"不支持的模式: {mode}")
        self.mode = mode
        self.config = config

    async def process_srt(
        self,
        srt_file_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        auto_adjust: bool = True,
    ) -> SRTBatchResult:
        """
        处理SRT文件

        Args:
            srt_file_path: SRT文件路径
            output_path: 输出音频路径
            progress_callback: 进度回调(current, total, current_text)
            auto_adjust: 是否自动调整时间轴

        Returns:
            SRTBatchResult: 处理结果
        """
        try:
            entries = self.parser.parse_file(srt_file_path)
            if not entries:
                return SRTBatchResult(
                    success=False,
                    output_path="",
                    total_entries=0,
                    generated_count=0,
                    failed_count=0,
                    total_duration=0.0,
                    adjustment_summary={},
                    error_message="SRT文件为空或解析失败",
                )

            stats = self.parser.get_statistics(entries)
            logger.info(
                f"SRT统计: {stats['count']}条字幕, 总时长{stats['total_duration']:.2f}s"
            )

            generated_audios = []
            audio_durations = []
            failed_indices = []

            total = len(entries)
            semaphore = asyncio.Semaphore(
                3
            )  # Limit concurrent tasks to avoid GPU memory overflow
            completed_count = 0
            results = [None] * total  # Pre-allocate results list

            async def generate_with_semaphore(entry: SRTEntry, index: int):
                async with semaphore:
                    try:
                        # ⭐ 添加5分钟超时保护，防止永久卡住
                        audio, sr = await asyncio.wait_for(
                            self._generate_single(entry.text),
                            timeout=GENERATION_TIMEOUT_SECONDS,  # 5分钟
                        )
                        duration = len(audio) / sr
                        return (index, audio, sr, duration, None)
                    except asyncio.TimeoutError as e:
                        logger.error(
                            f"⚠️ 字幕 {entry.index} 生成超时（5分钟），跳过此条"
                        )
                        return (
                            index,
                            None,
                            24000,
                            0.0,
                            e,
                        )
                    except Exception as e:
                        logger.error(f"✗ 字幕 {entry.index} 生成失败: {e}")
                        return (index, None, 24000, 0.0, e)

            tasks = [
                generate_with_semaphore(entry, i) for i, entry in enumerate(entries)
            ]

            # Use as_completed to update progress in real-time as tasks finish
            for coro in asyncio.as_completed(tasks):
                index, audio, sr, duration, error = await coro
                results[index] = (audio, sr, duration, error)

                if error:
                    failed_indices.append(index)

                # Real-time progress callback after each task completes
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total, entries[index].text[:50])
                logger.info(
                    f"完成字幕 {completed_count}/{total}: {entries[index].text[:30]}... ({duration:.2f}s)"
                )

                # ⭐ 每处理10条清理一次显存，防止累积
                if completed_count % GPU_CLEANUP_INTERVAL == 0:
                    try:
                        import gc

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gpu_mem = torch.cuda.memory_allocated() / 1024**3
                            logger.info(
                                f"🧹 已清理GPU缓存，当前显存: {gpu_mem:.2f}GB (处理了 {completed_count} 条)"
                            )
                    except Exception as cleanup_err:
                        logger.warning(f"清理显存时出错: {cleanup_err}")

            # Process results in order
            for index, (audio, sr, duration, error) in enumerate(results):
                generated_audios.append((audio, sr))
                audio_durations.append(duration)

            successful_count = total - len(failed_indices)
            if successful_count == 0:
                return SRTBatchResult(
                    success=False,
                    output_path="",
                    total_entries=total,
                    generated_count=0,
                    failed_count=len(failed_indices),
                    total_duration=0.0,
                    adjustment_summary={},
                    error_message="所有字幕生成失败",
                )

            if auto_adjust:
                scheduled = self.scheduler.schedule(entries, audio_durations)
            else:
                scheduled = [
                    ScheduledEntry(
                        entry=e,
                        actual_start=e.start_time,
                        actual_end=e.start_time + audio_durations[i],
                        audio_data=generated_audios[i][0],
                        sample_rate=generated_audios[i][1],
                    )
                    for i, e in enumerate(entries)
                ]

            for i, s in enumerate(scheduled):
                s.audio_data = generated_audios[i][0]
                s.sample_rate = generated_audios[i][1]

            self.assembler.sample_rate = (
                scheduled[0].sample_rate if scheduled else 24000
            )
            final_path = self.assembler.assemble(scheduled, output_path)

            total_duration = self.scheduler.get_total_duration(scheduled)
            adjustment_summary = self.scheduler.get_adjustment_summary()

            return SRTBatchResult(
                success=True,
                output_path=final_path,
                total_entries=total,
                generated_count=successful_count,
                failed_count=len(failed_indices),
                total_duration=total_duration,
                adjustment_summary=adjustment_summary,
            )

        except Exception as e:
            logger.exception("SRT批量处理失败")
            return SRTBatchResult(
                success=False,
                output_path="",
                total_entries=0,
                generated_count=0,
                failed_count=0,
                total_duration=0.0,
                adjustment_summary={},
                error_message=str(e),
            )

    async def _generate_single(self, text: str) -> tuple:
        """根据当前模式生成单条音频"""
        if self.mode == self.MODE_CUSTOM_VOICE:
            return await self._generate_custom_voice(text)
        elif self.mode == self.MODE_VOICE_DESIGN:
            return await self._generate_voice_design(text)
        else:
            return await self._generate_voice_clone(text)

    async def _generate_custom_voice(self, text: str) -> tuple:
        """Custom Voice模式生成"""
        config = self.config
        if not isinstance(config, CustomVoiceConfig):
            raise TypeError("当前模式需要CustomVoiceConfig")

        return await self.tts_engine.custom_voice_synthesize_async(
            text=text,
            speaker=config.speaker,
            language=config.language,
            instruct=config.instruct,
            speed_factor=config.speed_factor,
            pitch_factor=config.pitch_factor,
        )

    async def _generate_voice_design(self, text: str) -> tuple:
        """Voice Design模式生成"""
        config = self.config
        if not isinstance(config, VoiceDesignConfig):
            raise TypeError("当前模式需要VoiceDesignConfig")

        return await self.tts_engine.voice_design_synthesize_async(
            text=text,
            design_prompt=config.design_prompt,
            language=config.language,
            speed_factor=config.speed_factor,
            pitch_factor=config.pitch_factor,
        )

    async def _generate_voice_clone(self, text: str) -> tuple:
        """Voice Clone模式生成"""
        config = self.config
        if not isinstance(config, VoiceCloneConfig):
            raise TypeError("当前模式需要VoiceCloneConfig")

        valid, error = config.validate()
        if not valid:
            raise ValueError(error)

        kwargs = config.to_generation_kwargs()
        return await self.tts_engine.voice_clone_synthesize_async(text=text, **kwargs)
