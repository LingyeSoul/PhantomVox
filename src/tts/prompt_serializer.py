"""
Prompt 特征序列化工具

负责 VoiceClonePromptItem 的保存和加载

安全说明:
- 使用 safetensors 格式存储，防止任意代码执行
- 自动迁移旧 .pt 文件到 safetensors 格式
- 仅支持加载 tensors 和基本类型（str, int, float, bool, None）
"""

import logging
import json
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
        ref_code: Any
        ref_spk_embedding: Any
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
    保存 VoiceClonePromptItem 到 safetensors 文件

    Args:
        prompt_item: 要保存的 prompt 特征对象（可以是单个对象或列表）
        file_path: 目标文件路径（.safetensors，兼容 .pt 会自动转换）
        metadata: 额外的元数据（可选）

    Returns:
        bool: 是否成功
    """
    import torch
    from safetensors.torch import save_file

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

        # 准备 tensors (只包含 Tensor 类型，确保连续性)
        if not ref_spk_embedding.is_contiguous():
            ref_spk_embedding = ref_spk_embedding.contiguous()

        tensors = {
            "ref_spk_embedding": ref_spk_embedding,
        }
        if ref_code is not None:
            if not ref_code.is_contiguous():
                ref_code = ref_code.contiguous()
            tensors["ref_code"] = ref_code

        # 准备 metadata (非 Tensor 类型，safetensors 只支持字符串值)
        meta = {
            "x_vector_only_mode": str(prompt_item.x_vector_only_mode),
            "icl_mode": str(prompt_item.icl_mode),
            "ref_text": prompt_item.ref_text or "",
            "version": FEATURES_VERSION,
            "created_at": datetime.now().isoformat()
        }

        # 添加额外元数据
        if metadata:
            meta["metadata"] = json.dumps(metadata, ensure_ascii=False)

        # 如果路径是 .pt，自动改为 .safetensors
        if file_path.endswith('.pt'):
            file_path = file_path[:-3] + '.safetensors'

        # 保存到 safetensors 文件
        save_file(tensors, file_path, metadata=meta)

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
    从文件加载 VoiceClonePromptItem

    支持 safetensors 和旧 .pt 格式（自动迁移）

    Args:
        file_path: 特征文件路径（.safetensors 或 .pt）
        device: 目标设备（如 'cuda:0', 'cpu'），None 表示保持在 CPU

    Returns:
        VoiceClonePromptItem 对象，或 None
    """
    from safetensors import safe_open

    try:
        # 自动检测文件格式
        if file_path.endswith('.safetensors'):
            return _load_from_safetensors(file_path, device)
        elif file_path.endswith('.pt'):
            # 自动迁移 .pt 到 .safetensors
            safe_path = file_path[:-3] + '.safetensors'
            if not Path(safe_path).exists():
                logger.info(f"检测到旧格式文件，正在自动迁移: {file_path}")
                if not _migrate_pt_to_safetensors(file_path, safe_path):
                    logger.error("自动迁移失败，请重新生成特征文件")
                    return None
            # 从迁移后的 safetensors 文件加载
            return _load_from_safetensors(safe_path, device)
        else:
            # 无扩展名时，优先查找 safetensors
            if Path(file_path + '.safetensors').exists():
                return _load_from_safetensors(file_path + '.safetensors', device)
            elif Path(file_path + '.pt').exists():
                # 自动迁移
                pt_path = file_path + '.pt'
                safe_path = file_path + '.safetensors'
                if not Path(safe_path).exists():
                    logger.info(f"检测到旧格式文件，正在自动迁移: {pt_path}")
                    if not _migrate_pt_to_safetensors(pt_path, safe_path):
                        logger.error("自动迁移失败，请重新生成特征文件")
                        return None
                return _load_from_safetensors(safe_path, device)
            else:
                logger.error(f"找不到特征文件: {file_path}")
                return None
    except Exception as e:
        logger.error(f"✗ 加载 Prompt 特征失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def _load_from_safetensors(
    file_path: str,
    device=None
) -> Optional[VoiceClonePromptItem]:
    """
    从 safetensors 文件加载（内部函数）

    Args:
        file_path: safetensors 文件路径
        device: 目标设备

    Returns:
        VoiceClonePromptItem 对象，或 None
    """
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # 读取 metadata
            meta = f.metadata() or {}

            # 读取 tensors
            ref_spk_embedding = f.get_tensor("ref_spk_embedding")
            ref_code = f.get_tensor("ref_code") if "ref_code" in f.keys() else None

            # 移动到目标设备
            if device is not None:
                if ref_code is not None:
                    ref_code = ref_code.to(device)
                ref_spk_embedding = ref_spk_embedding.to(device)

        # 解析 metadata
        x_vector_only_mode = meta.get("x_vector_only_mode", "False").lower() == "true"
        icl_mode = meta.get("icl_mode", "False").lower() == "true"
        ref_text = meta.get("ref_text") or None

        # 版本检查
        version = meta.get("version", "0.0")
        if version != FEATURES_VERSION:
            logger.warning(
                f"特征版本不匹配: {version} (当前: {FEATURES_VERSION})"
            )

        # 构建 VoiceClonePromptItem
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
        logger.error(f"加载 safetensors 文件失败: {e}")
        return None


def _migrate_pt_to_safetensors(pt_path: str, safe_path: str) -> bool:
    """
    将旧 .pt 文件迁移到 safetensors 格式（内部函数）

    Args:
        pt_path: 源 .pt 文件路径
        safe_path: 目标 .safetensors 文件路径

    Returns:
        bool: 是否成功
    """
    import torch

    try:
        # 加载旧文件（使用 weights_only=True 安全加载）
        data = torch.load(pt_path, weights_only=True)

        # 提取 tensors（确保连续性，safetensors 要求）
        ref_spk_embedding = data["ref_spk_embedding"]
        if not ref_spk_embedding.is_contiguous():
            ref_spk_embedding = ref_spk_embedding.contiguous()

        tensors = {"ref_spk_embedding": ref_spk_embedding}

        if data.get("ref_code") is not None:
            ref_code = data["ref_code"]
            if not ref_code.is_contiguous():
                ref_code = ref_code.contiguous()
            tensors["ref_code"] = ref_code

        # 提取 metadata (safetensors metadata 只支持字符串值)
        meta = {
            "x_vector_only_mode": str(data.get("x_vector_only_mode", False)),
            "icl_mode": str(data.get("icl_mode", False)),
            "ref_text": data.get("ref_text") or "",
            "version": data.get("version", FEATURES_VERSION),
            "created_at": data.get("created_at", datetime.now().isoformat())
        }
        if data.get("metadata"):
            meta["metadata"] = json.dumps(data["metadata"], ensure_ascii=False)

        # 保存为 safetensors
        save_file(tensors, safe_path, metadata=meta)

        # 删除旧的 .pt 文件
        Path(pt_path).unlink()
        logger.info(f"✓ 迁移成功并删除旧文件: {pt_path} -> {safe_path}")
        return True

    except Exception as e:
        logger.error(f"迁移失败: {e}")
        return False


def validate_prompt_features(prompt_item: dict) -> bool:
    """
    验证 prompt 特征的完整性

    Args:
        prompt_item: 要验证的特征字典

    Returns:
        bool: 是否有效
    """
    import torch

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
