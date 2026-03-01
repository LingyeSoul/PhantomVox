"""
时间轴调度器模块

处理字幕时间轴与音频实际时长的冲突检测和调整
"""

import logging
from typing import List, Tuple
from dataclasses import dataclass

from api.constants import SAMPLE_RATE

from .srt_parser import SRTEntry, ScheduledEntry

logger = logging.getLogger(__name__)


@dataclass
class TimelineAdjustment:
    """时间轴调整记录"""

    entry_index: int
    original_start: float
    original_end: float
    adjusted_start: float
    adjusted_end: float
    reason: str


class TimelineScheduler:
    """
    时间轴调度器

    核心算法：检测音频时长与字幕时间的冲突并自动延后
    """

    def __init__(self, gap_padding: float = 1.0, sample_rate: int = SAMPLE_RATE):
        self.gap_padding = gap_padding
        self.sample_rate = sample_rate
        self.adjustments: List[TimelineAdjustment] = []

    def schedule(
        self, entries: List[SRTEntry], audio_durations: List[float]
    ) -> List[ScheduledEntry]:
        """
        调度算法

        场景1: 音频正常（在字幕时间范围内）
        [字幕A: 00:00-00:05] 音频3s → 无冲突

        场景2: 音频超出下一条开始时间
        [字幕A: 00:00-00:05] 音频7s → 冲突!
                              ↓
        [字幕B: 00:05-00:10] 需要延后到 00:08 (00:00+7s+1s间隔)

        场景3: 连锁反应
        [字幕A: 00:00-00:05] 音频7s
        [字幕B: 00:05-00:10] 音频6s → 延后到00:08
        [字幕C: 00:10-00:15] 音频5s → 需要再延后

        Args:
            entries: SRT字幕条目列表
            audio_durations: 对应每条字幕的音频时长列表

        Returns:
            List[ScheduledEntry]: 调度后的条目列表
        """
        if len(entries) != len(audio_durations):
            raise ValueError(
                f"条目数量({len(entries)})与音频时长数量({len(audio_durations)})不匹配"
            )

        self.adjustments = []
        scheduled = []
        last_end_time = 0.0

        for i, (entry, audio_duration) in enumerate(zip(entries, audio_durations)):
            if i == 0:
                actual_start = entry.start_time
            else:
                min_start = last_end_time + self.gap_padding
                actual_start = max(entry.start_time, min_start)

            actual_end = actual_start + audio_duration

            if actual_start > entry.start_time:
                adjustment = TimelineAdjustment(
                    entry_index=entry.index,
                    original_start=entry.start_time,
                    original_end=entry.end_time,
                    adjusted_start=actual_start,
                    adjusted_end=actual_end,
                    reason=f"前一条音频结束于{last_end_time:.2f}s，延后以保证{self.gap_padding}s间隔",
                )
                self.adjustments.append(adjustment)
                logger.info(
                    f"字幕 {entry.index}: 时间调整后 "
                    f"{entry.start_time:.2f}s-{entry.end_time:.2f}s → "
                    f"{actual_start:.2f}s-{actual_end:.2f}s"
                )

            scheduled.append(
                ScheduledEntry(
                    entry=entry,
                    actual_start=actual_start,
                    actual_end=actual_end,
                    audio_data=None,
                    sample_rate=self.sample_rate,
                )
            )

            last_end_time = actual_end

        return scheduled

    def calculate_gaps(
        self, scheduled: List[ScheduledEntry]
    ) -> List[Tuple[float, float]]:
        """
        计算需要填充静音的间隙

        Args:
            scheduled: 调度后的条目列表

        Returns:
            List[Tuple[float, float]]: [(开始时间, 持续时长), ...]
        """
        gaps = []

        for i in range(len(scheduled) - 1):
            current = scheduled[i]
            next_entry = scheduled[i + 1]

            gap_start = current.actual_end
            gap_duration = next_entry.actual_start - current.actual_end

            if gap_duration > 0.01:
                gaps.append((gap_start, gap_duration))

        return gaps

    def get_total_duration(self, scheduled: List[ScheduledEntry]) -> float:
        """获取调整后总时长"""
        if not scheduled:
            return 0.0
        return scheduled[-1].actual_end

    def get_adjustment_summary(self) -> dict:
        """获取调整摘要"""
        if not self.adjustments:
            return {
                "adjusted_count": 0,
                "total_delay": 0.0,
                "max_delay": 0.0,
            }

        delays = [adj.adjusted_start - adj.original_start for adj in self.adjustments]

        return {
            "adjusted_count": len(self.adjustments),
            "total_delay": sum(delays),
            "max_delay": max(delays),
            "adjustments": [
                {
                    "index": adj.entry_index,
                    "delay": adj.adjusted_start - adj.original_start,
                    "reason": adj.reason,
                }
                for adj in self.adjustments
            ],
        }
