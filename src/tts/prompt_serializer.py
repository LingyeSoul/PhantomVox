"""
Prompt 特征序列化工具

负责 VoiceClonePromptItem 的保存和加载
"""

import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Union

# 导入 VoiceClonePromptItem 数据类
try:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
except ImportError:
    # 如果导入失败，定义一个简单的 dataclass（向后兼容）
    from dataclasses import dataclass
    @dataclass
    class VoiceClonePromptItem:
        ref_code: Optional[torch.Tensor]
        ref_spk_embedding: torch.Tensor
        x_vector_only_mode: bool
        icl_mode: bool
        ref_text: Optional[str] = None

logger = logging.getLogger(__name__)

# 版本常量
FEATURES_VERSION = "1.0"


def save_prompt_features(
    prompt_item: Any,
    file_path: str,
    metadata: Optional[dict] = None
) -> bool:
    """
    保存 VoiceClonePromptItem 到文件

    Args:
        prompt_item: 要保存的 prompt 特征对象（可以是单个对象或列表）
        file_path: 目标文件路径（.pt）
        metadata: 额外的元数据（可选）

    Returns:
        bool: 是否成功
    """
    try:
        # 如果是列表，取第一个元素
        if isinstance(prompt_item, list):
            if len(prompt_item) == 0:
                logger.error("Prompt 特征列表为空")
                return False
            prompt_item = prompt_item[0]

        # 确保 tensor 在 CPU 上（避免 GPU 内存泄漏）
        ref_code = prompt_item.ref_code
        ref_spk_embedding = prompt_item.ref_spk_embedding

        if ref_code is not None and hasattr(ref_code, 'device'):
            ref_code = ref_code.cpu()
        if hasattr(ref_spk_embedding, 'device'):
            ref_spk_embedding = ref_spk_embedding.cpu()

        # 准备保存数据
        data = {
            "ref_code": ref_code,
            "ref_spk_embedding": ref_spk_embedding,
            "x_vector_only_mode": prompt_item.x_vector_only_mode,
            "icl_mode": prompt_item.icl_mode,
            "ref_text": prompt_item.ref_text,
            "version": FEATURES_VERSION,
            "created_at": datetime.now().isoformat()
        }

        # 添加额外元数据
        if metadata:
            data["metadata"] = metadata

        # 保存到文件
        torch.save(data, file_path)

        logger.info(f"✓ Prompt 特征已保存: {Path(file_path).name}")
        return True

    except Exception as e:
        logger.error(f"✗ 保存 Prompt 特征失败: {e}")
        return False


def load_prompt_features(
    file_path: str,
    device=None
) -> Optional[Union[VoiceClonePromptItem, List[VoiceClonePromptItem]]]:
    """
    从文件加载 VoiceClonePromptItem（返回对象格式）

    Args:
        file_path: 特征文件路径（.pt）
        device: 目标设备（如 'cuda:0', 'cpu'），None 表示保持在 CPU

    Returns:
        VoiceClonePromptItem 对象，或 None
        如果保存时是列表，则返回单个对象（取第一个）
    """
    try:
        # 加载数据
        data = torch.load(file_path, weights_only=False)

        # 版本检查
        version = data.get("version", "0.0")
        if version != FEATURES_VERSION:
            logger.warning(
                f"特征版本不匹配: {version} (当前: {FEATURES_VERSION})"
            )

        # 获取 ref_code 和 ref_spk_embedding
        ref_code = data.get("ref_code")
        ref_spk_embedding = data["ref_spk_embedding"]
        x_vector_only_mode = data["x_vector_only_mode"]
        icl_mode = data["icl_mode"]
        ref_text = data.get("ref_text")

        # 如果指定了设备，将张量移动到目标设备
        if device is not None:
            if ref_code is not None and hasattr(ref_code, 'to'):
                ref_code = ref_code.to(device)
            if hasattr(ref_spk_embedding, 'to'):
                ref_spk_embedding = ref_spk_embedding.to(device)

        # 创建 VoiceClonePromptItem 对象
        prompt_item = VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
            ref_text=ref_text
        )

        logger.info(f"✓ Prompt 特征已加载: {Path(file_path).name}")
        logger.debug(f"  - ref_code: {'[tensor]' if ref_code is not None else 'None'}")
        if ref_spk_embedding is not None:
            logger.debug(f"  - ref_spk_embedding: [tensor] shape={ref_spk_embedding.shape}")
        logger.debug(f"  - x_vector_only_mode: {x_vector_only_mode}")
        logger.debug(f"  - icl_mode: {icl_mode}")
        logger.debug(f"  - ref_text: {ref_text}")

        return prompt_item

    except Exception as e:
        logger.error(f"✗ 加载 Prompt 特征失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def validate_prompt_features(prompt_item: dict) -> bool:
    """
    验证 prompt 特征的完整性

    Args:
        prompt_item: 要验证的特征字典

    Returns:
        bool: 是否有效
    """
    try:
        # 检查必需字段
        if prompt_item.get("ref_spk_embedding") is None:
            logger.error("ref_spk_embedding 为空")
            return False

        # 检查 tensor 形状
        if not isinstance(prompt_item["ref_spk_embedding"], torch.Tensor):
            logger.error("ref_spk_embedding 不是 Tensor")
            return False

        # 检查是否在 CPU 上（避免 GPU 内存泄漏）
        if hasattr(prompt_item["ref_spk_embedding"], 'device'):
            if prompt_item["ref_spk_embedding"].device.type != "cpu":
                logger.warning("ref_spk_embedding 在 GPU 上，移动到 CPU")
                prompt_item["ref_spk_embedding"] = prompt_item["ref_spk_embedding"].cpu()

        if prompt_item.get("ref_code") is not None and hasattr(prompt_item["ref_code"], 'device'):
            if prompt_item["ref_code"].device.type != "cpu":
                prompt_item["ref_code"] = prompt_item["ref_code"].cpu()

        return True

    except Exception as e:
        logger.error(f"验证 Prompt 特征失败: {e}")
        return False
