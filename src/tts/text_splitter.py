"""
文本分割工具

提供多行文本和句子分割功能，用于批量语音生成
"""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)


def split_by_newline(text: str, min_length: int = 1) -> List[str]:
    """
    按换行符分割文本，过滤空行

    Args:
        text: 输入文本
        min_length: 最小行长度（过滤短行）

    Returns:
        List[str]: 分割后的文本列表
    """
    if not text:
        return []

    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= min_length:
            result.append(stripped)

    logger.debug(f"按行分割: 输入 {len(lines)} 行, 输出 {len(result)} 个有效文本")
    return result


def split_by_sentences(text: str, language: str = "chinese") -> List[str]:
    """
    按句子分割文本

    支持中英日韩等语言的句子分割，根据语言选择合适的句末标点。

    Args:
        text: 输入文本
        language: 语言类型 (chinese, english, japanese, korean, auto)

    Returns:
        List[str]: 分割后的句子列表
    """
    if not text:
        return []

    # 预处理：移除多余空白
    text = text.strip()
    if not text:
        return []

    # 语言特定的句末标点
    lang_lower = language.lower()

    if lang_lower in ["chinese", "zh", "中文"]:
        # 中文：。！？；…+
        sentence_endings = r'[。！？；]+|…{2,}'
    elif lang_lower in ["japanese", "ja", "日语"]:
        # 日语：。！？；
        sentence_endings = r'[。！？；]+'
    elif lang_lower in ["korean", "ko", "韩语"]:
        # 韩语：。！？；
        sentence_endings = r'[。！？；]+'
    elif lang_lower in ["english", "en", "英语"]:
        # 英语：. ! ? (注意避免缩写如 Mr. Dr. 等)
        # 简单处理：使用更严格的匹配
        sentence_endings = r'[.!?]+'
    else:
        # 自动模式或默认：使用通用标点
        sentence_endings = r'[。！？；.!?]+'

    # 分割并保留分隔符
    parts = re.split(f'({sentence_endings})', text)

    sentences = []
    current = ""

    for part in parts:
        current += part
        # 检查是否到达句末
        if re.match(sentence_endings, part):
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""

    # 添加最后一个不完整的句子（如果没有以标点结尾）
    if current.strip():
        sentences.append(current.strip())

    logger.debug(f"按句分割 ({language}): 输出 {len(sentences)} 个句子")
    return sentences


def estimate_batch_size(texts: List[str], max_chars_per_batch: int = 500) -> int:
    """
    估算合适的批处理大小

    根据平均文本长度估算每批处理的文本数量。

    Args:
        texts: 文本列表
        max_chars_per_batch: 每批最大字符数

    Returns:
        int: 建议的批处理大小
    """
    if not texts:
        return 1

    avg_length = sum(len(t) for t in texts) / len(texts)
    if avg_length == 0:
        return 1

    return max(1, int(max_chars_per_batch / avg_length))


def smart_split(text: str, mode: str = "multiline", language: str = "chinese") -> List[str]:
    """
    智能分割文本

    根据模式自动选择分割方式。

    Args:
        text: 输入文本
        mode: 分割模式 ("multiline" 或 "sentence")
        language: 语言类型（用于句子分割）

    Returns:
        List[str]: 分割后的文本列表
    """
    if mode == "multiline":
        return split_by_newline(text)
    elif mode == "sentence":
        return split_by_sentences(text, language)
    else:
        logger.warning(f"未知的分割模式: {mode}, 使用默认按行分割")
        return split_by_newline(text)
