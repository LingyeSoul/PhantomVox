"""
SRT字幕解析器模块

提供SRT格式字幕文件的解析功能
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from pathlib import Path
from html.parser import HTMLParser
from api.constants import SAMPLE_RATE

logger = logging.getLogger(__name__)

MAX_SRT_FILE_SIZE = 10 * 1024 * 1024


class HTMLStripper(HTMLParser):
    """HTML标签移除器，使用HTMLParser避免XSS漏洞"""

    def __init__(self):
        super().__init__()
        self.reset()
        self.result = []
        self._skip_content = False

    def handle_starttag(self, tag: str, attrs: list):
        if tag.lower() in ("script", "style", "noscript"):
            self._skip_content = True

    def handle_endtag(self, tag: str):
        if tag.lower() in ("script", "style", "noscript"):
            self._skip_content = False

    def handle_data(self, data: str):
        if not self._skip_content:
            self.result.append(data)

    def get_data(self) -> str:
        return "".join(self.result)


def strip_html_tags(text: str) -> str:
    """
    安全地移除HTML标签

    使用HTMLParser而不是正则表达式，避免XSS漏洞

    Args:
        text: 包含HTML的文本

    Returns:
        str: 移除标签后的文本
    """
    stripper = HTMLStripper()
    stripper.feed(text)
    return stripper.get_data()


@dataclass
class SRTEntry:
    """SRT字幕条目"""

    index: int
    start_time: float  # seconds
    end_time: float  # seconds
    text: str

    @property
    def duration(self) -> float:
        """字幕持续时间"""
        return self.end_time - self.start_time


@dataclass
class ScheduledEntry:
    """调度后的字幕条目（包含生成结果）"""

    entry: SRTEntry
    actual_start: float  # 调整后的开始时间
    actual_end: float  # 调整后的结束时间
    audio_data: Optional[Any] = None  # numpy array
    sample_rate: int = SAMPLE_RATE

    @property
    def audio_duration(self) -> float:
        """实际音频时长"""
        if self.audio_data is not None:
            return len(self.audio_data) / self.sample_rate  # type: ignore
        return 0.0


class SRTParser:
    """SRT字幕解析器"""

    # SRT时间戳格式: 00:00:00,000 --> 00:00:00,000
    TIME_PATTERN = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
        r"\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    def parse(self, srt_content: str) -> List[SRTEntry]:
        """
        解析SRT文本内容

        Args:
            srt_content: SRT文件内容字符串

        Returns:
            List[SRTEntry]: 字幕条目列表

        Raises:
            ValueError: 解析失败时抛出
        """
        entries = []
        blocks = self._split_blocks(srt_content)

        for block in blocks:
            if not block.strip():
                continue

            entry = self._parse_block(block)
            if entry:
                entries.append(entry)

        logger.info(f"SRT解析完成: {len(entries)} 条字幕")
        return entries

    def parse_file(self, file_path: str) -> List[SRTEntry]:
        """
        从文件解析SRT

        Args:
            file_path: SRT文件路径

        Returns:
            List[SRTEntry]: 字幕条目列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件过大
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SRT文件不存在: {file_path}")

        file_size = path.stat().st_size
        if file_size > MAX_SRT_FILE_SIZE:
            raise ValueError(
                f"SRT文件过大: {file_size / 1024 / 1024:.2f} MB "
                f"(最大允许 {MAX_SRT_FILE_SIZE / 1024 / 1024:.0f} MB)"
            )

        logger.info(f"正在解析SRT文件: {file_path} ({file_size / 1024:.2f} KB)")

        # 尝试多种编码
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
        content = None

        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                logger.debug(f"使用编码 {encoding} 读取文件")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"无法解码SRT文件，尝试了以下编码: {encodings}")

        return self.parse(content)

    def _split_blocks(self, content: str) -> List[str]:
        """将SRT内容分割成块"""
        # 标准化换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # 按空行分割
        blocks = re.split(r"\n\s*\n", content)
        return [block.strip() for block in blocks if block.strip()]

    def _parse_block(self, block: str) -> Optional[SRTEntry]:
        """解析单个字幕块"""
        lines = block.split("\n")
        if not lines:
            return None

        # 第一行应该是序号
        try:
            index = int(lines[0].strip())
        except ValueError:
            # 可能是时间戳在第一行
            index = 0
            time_line_idx = 0
        else:
            time_line_idx = 1

        if time_line_idx >= len(lines):
            return None

        # 解析时间戳
        time_line = lines[time_line_idx].strip()
        time_match = self.TIME_PATTERN.match(time_line)

        if not time_match:
            logger.warning(f"无法解析时间戳行: {time_line}")
            return None

        # 提取时间
        sh, sm, ss, sms, eh, em, es, ems = map(int, time_match.groups())
        start_time = sh * 3600 + sm * 60 + ss + sms / 1000.0
        end_time = eh * 3600 + em * 60 + es + ems / 1000.0

        # 提取文本（剩余行）
        text_lines = lines[time_line_idx + 1 :]
        text = "\n".join(text_lines).strip()

        text = strip_html_tags(text)

        # 规范化空白
        text = " ".join(text.split())

        if not text:
            logger.warning(f"字幕 {index} 文本为空，跳过")
            return None

        return SRTEntry(
            index=index, start_time=start_time, end_time=end_time, text=text
        )

    def validate_timeline(self, entries: List[SRTEntry]) -> Tuple[bool, List[str]]:
        """
        验证时间轴是否合理

        Args:
            entries: 字幕条目列表

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []

        if not entries:
            errors.append("字幕列表为空")
            return False, errors

        # 检查时间是否非递减
        for i in range(len(entries) - 1):
            current = entries[i]
            next_entry = entries[i + 1]

            if current.start_time > current.end_time:
                errors.append(
                    f"字幕 {current.index}: 开始时间({current.start_time:.3f}s) "
                    f"晚于结束时间({current.end_time:.3f}s)"
                )

            if current.end_time > next_entry.start_time:
                # 重叠警告（不是错误，因为可以处理）
                logger.warning(
                    f"字幕 {current.index} 和 {next_entry.index} 时间重叠: "
                    f"{current.end_time:.3f}s > {next_entry.start_time:.3f}s"
                )

        # 检查最后一条
        last = entries[-1]
        if last.start_time > last.end_time:
            errors.append(f"字幕 {last.index}: 开始时间晚于结束时间")

        return len(errors) == 0, errors

    def get_statistics(self, entries: List[SRTEntry]) -> dict:
        """
        获取字幕统计信息

        Args:
            entries: 字幕条目列表

        Returns:
            dict: 统计信息
        """
        if not entries:
            return {
                "count": 0,
                "total_duration": 0.0,
                "avg_text_length": 0.0,
                "max_text_length": 0,
                "min_text_length": 0,
            }

        text_lengths = [len(e.text) for e in entries]
        total_duration = entries[-1].end_time if entries else 0.0

        return {
            "count": len(entries),
            "total_duration": total_duration,
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "max_text_length": max(text_lengths),
            "min_text_length": min(text_lengths),
        }
