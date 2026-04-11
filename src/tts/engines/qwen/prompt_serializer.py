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

try:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class VoiceClonePromptItem:
        ref_code: Any
        ref_spk_embedding: Any
        x_vector_only_mode: bool
        icl_mode: bool
        ref_text: Optional[str] = None


logger = logging.getLogger(__name__)

FEATURES_VERSION = "1.0"


def save_prompt_features(
    prompt_item: Any, file_path: str, metadata: Optional[dict] = None
) -> bool:
    try:
        if isinstance(prompt_item, list):
            if len(prompt_item) == 0:
                logger.error("Prompt 特征列表为空")
                return False
            prompt_item = prompt_item[0]

        ref_code = prompt_item.ref_code
        ref_spk_embedding = prompt_item.ref_spk_embedding

        if ref_code is not None and hasattr(ref_code, "device"):
            ref_code = ref_code.cpu()
        if hasattr(ref_spk_embedding, "device"):
            ref_spk_embedding = ref_spk_embedding.cpu()

        if not ref_spk_embedding.is_contiguous():
            ref_spk_embedding = ref_spk_embedding.contiguous()

        tensors = {
            "ref_spk_embedding": ref_spk_embedding,
        }
        if ref_code is not None:
            if not ref_code.is_contiguous():
                ref_code = ref_code.contiguous()
            tensors["ref_code"] = ref_code

        meta = {
            "x_vector_only_mode": str(prompt_item.x_vector_only_mode),
            "icl_mode": str(prompt_item.icl_mode),
            "ref_text": prompt_item.ref_text or "",
            "version": FEATURES_VERSION,
            "created_at": datetime.now().isoformat(),
        }

        if metadata:
            meta["metadata"] = json.dumps(metadata, ensure_ascii=False)

        if file_path.endswith(".pt"):
            file_path = file_path[:-3] + ".safetensors"

        import torch
        from safetensors.torch import save_file

        save_file(tensors, file_path, metadata=meta)

        logger.info(f"✓ Prompt 特征已保存: {Path(file_path).name}")
        return True

    except Exception as e:
        logger.error(f"✗ 保存 Prompt 特征失败: {e}")
        return False


def load_prompt_features(
    file_path: str, device=None
) -> Optional[Union[VoiceClonePromptItem, List[VoiceClonePromptItem]]]:
    try:
        if file_path.endswith(".safetensors"):
            return _load_from_safetensors(file_path, device)
        elif file_path.endswith(".pt"):
            safe_path = file_path[:-3] + ".safetensors"
            if not Path(safe_path).exists():
                logger.info(f"检测到旧格式文件，正在自动迁移: {file_path}")
                if not _migrate_pt_to_safetensors(file_path, safe_path):
                    logger.error("自动迁移失败，请重新生成特征文件")
                    return None
            return _load_from_safetensors(safe_path, device)
        else:
            if Path(file_path + ".safetensors").exists():
                return _load_from_safetensors(file_path + ".safetensors", device)
            elif Path(file_path + ".pt").exists():
                pt_path = file_path + ".pt"
                safe_path = file_path + ".safetensors"
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
    file_path: str, device=None
) -> Optional[VoiceClonePromptItem]:
    from safetensors import safe_open

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}

            ref_spk_embedding = f.get_tensor("ref_spk_embedding")
            ref_code = f.get_tensor("ref_code") if "ref_code" in f.keys() else None

            if device is not None:
                if ref_code is not None:
                    ref_code = ref_code.to(device)
                ref_spk_embedding = ref_spk_embedding.to(device)

        x_vector_only_mode = meta.get("x_vector_only_mode", "False").lower() == "true"
        icl_mode = meta.get("icl_mode", "False").lower() == "true"
        ref_text = meta.get("ref_text") or None

        version = meta.get("version", "0.0")
        if version != FEATURES_VERSION:
            logger.warning(f"特征版本不匹配: {version} (当前: {FEATURES_VERSION})")

        prompt_item = VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
            ref_text=ref_text,
        )

        logger.info(f"✓ Prompt 特征已加载: {Path(file_path).name}")
        logger.debug(f"  - ref_code: {'[tensor]' if ref_code is not None else 'None'}")
        if ref_spk_embedding is not None:
            logger.debug(
                f"  - ref_spk_embedding: [tensor] shape={ref_spk_embedding.shape}"
            )
        logger.debug(f"  - x_vector_only_mode: {x_vector_only_mode}")
        logger.debug(f"  - icl_mode: {icl_mode}")
        logger.debug(f"  - ref_text: {ref_text}")

        return prompt_item

    except Exception as e:
        logger.error(f"加载 safetensors 文件失败: {e}")
        return None


def _migrate_pt_to_safetensors(pt_path: str, safe_path: str) -> bool:
    import torch

    try:
        data = torch.load(pt_path, weights_only=True)

        ref_spk_embedding = data["ref_spk_embedding"]
        if not ref_spk_embedding.is_contiguous():
            ref_spk_embedding = ref_spk_embedding.contiguous()

        tensors = {"ref_spk_embedding": ref_spk_embedding}

        if data.get("ref_code") is not None:
            ref_code = data["ref_code"]
            if not ref_code.is_contiguous():
                ref_code = ref_code.contiguous()
            tensors["ref_code"] = ref_code

        meta = {
            "x_vector_only_mode": str(data.get("x_vector_only_mode", False)),
            "icl_mode": str(data.get("icl_mode", False)),
            "ref_text": data.get("ref_text") or "",
            "version": data.get("version", FEATURES_VERSION),
            "created_at": data.get("created_at", datetime.now().isoformat()),
        }
        if data.get("metadata"):
            meta["metadata"] = json.dumps(data["metadata"], ensure_ascii=False)

        from safetensors.torch import save_file

        save_file(tensors, safe_path, metadata=meta)

        Path(pt_path).unlink()
        logger.info(f"✓ 迁移成功并删除旧文件: {pt_path} -> {safe_path}")
        return True

    except Exception as e:
        logger.error(f"迁移失败: {e}")
        return False


def validate_prompt_features(prompt_item: dict) -> bool:
    import torch

    try:
        if prompt_item.get("ref_spk_embedding") is None:
            logger.error("ref_spk_embedding 为空")
            return False

        if not isinstance(prompt_item["ref_spk_embedding"], torch.Tensor):
            logger.error("ref_spk_embedding 不是 Tensor")
            return False

        if hasattr(prompt_item["ref_spk_embedding"], "device"):
            if prompt_item["ref_spk_embedding"].device.type != "cpu":
                logger.warning("ref_spk_embedding 在 GPU 上，移动到 CPU")
                prompt_item["ref_spk_embedding"] = prompt_item[
                    "ref_spk_embedding"
                ].cpu()

        if prompt_item.get("ref_code") is not None and hasattr(
            prompt_item["ref_code"], "device"
        ):
            if prompt_item["ref_code"].device.type != "cpu":
                prompt_item["ref_code"] = prompt_item["ref_code"].cpu()

        return True

    except Exception as e:
        logger.error(f"验证 Prompt 特征失败: {e}")
        return False
