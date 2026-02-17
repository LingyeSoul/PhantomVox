"""
Prompt 管理模块

提供 Voice Clone Prompt 的创建、转换和管理功能
"""

import logging
from typing import List, Optional, Any

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

logger = logging.getLogger(__name__)


class PromptManager:
    """Voice Clone Prompt 管理器"""

    def __init__(self, model, device: str = "cuda:0"):
        """
        初始化 Prompt 管理器

        Args:
            model: Qwen3TTSModel 实例
            device: 运行设备
        """
        self.model = model
        self.device = device

    def convert_prompt_to_prompt_items(
        self, clone_prompt: Any
    ) -> List[VoiceClonePromptItem]:
        """
        将各种格式的 clone_prompt 转换为 VoiceClonePromptItem 对象列表

        Args:
            clone_prompt: 可以是以下格式之一:
                - VoiceClonePromptItem 对象
                - List[VoiceClonePromptItem]
                - dict (从 load_prompt_features 加载的格式)

        Returns:
            List[VoiceClonePromptItem]: 模型期望的对象列表
        """
        if isinstance(clone_prompt, list):
            if len(clone_prompt) == 0:
                from tts.exceptions import TTSInvalidParameterError

                raise TTSInvalidParameterError("clone_prompt 列表为空")
            if isinstance(clone_prompt[0], VoiceClonePromptItem):
                return clone_prompt
            else:
                from tts.exceptions import TTSInvalidParameterError

                raise TTSInvalidParameterError(
                    "clone_prompt 列表中的元素必须是 VoiceClonePromptItem 对象"
                )

        if isinstance(clone_prompt, VoiceClonePromptItem):
            return [clone_prompt]

        if isinstance(clone_prompt, dict):
            try:
                ref_code_list = clone_prompt.get("ref_code")
                ref_spk_embedding_list = clone_prompt["ref_spk_embedding"]
                x_vector_only_mode_list = clone_prompt["x_vector_only_mode"]
                icl_mode_list = clone_prompt["icl_mode"]
                ref_text_list = clone_prompt.get("ref_text")

                n = len(ref_spk_embedding_list)

                prompt_items = []
                for i in range(n):
                    ref_code = ref_code_list[i] if ref_code_list is not None else None
                    ref_spk_embedding = ref_spk_embedding_list[i]
                    x_vector_only_mode = x_vector_only_mode_list[i]
                    icl_mode = icl_mode_list[i]
                    ref_text = ref_text_list[i] if ref_text_list is not None else None

                    if hasattr(ref_spk_embedding, "to"):
                        ref_spk_embedding = ref_spk_embedding.to(self.device)
                    if ref_code is not None and hasattr(ref_code, "to"):
                        ref_code = ref_code.to(self.device)

                    prompt_items.append(
                        VoiceClonePromptItem(
                            ref_code=ref_code,
                            ref_spk_embedding=ref_spk_embedding,
                            x_vector_only_mode=x_vector_only_mode,
                            icl_mode=icl_mode,
                            ref_text=ref_text,
                        )
                    )

                logger.info(
                    f"✓ 已转换 {len(prompt_items)} 个 VoiceClonePromptItem 对象"
                )
                return prompt_items

            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"转换 clone_prompt 字典失败: {e}")
                from tts.exceptions import TTSInvalidParameterError

                raise TTSInvalidParameterError(f"clone_prompt 字典格式无效: {e}")

        logger.error(f"不支持的 clone_prompt 类型: {type(clone_prompt)}")
        from tts.exceptions import TTSInvalidParameterError

        raise TTSInvalidParameterError(
            "clone_prompt 必须是 VoiceClonePromptItem、List[VoiceClonePromptItem] 或 dict"
        )

    def create_voice_clone_prompt(
        self, ref_audio: str, ref_text: str, x_vector_only: bool = False
    ):
        """
        创建可重用的声音克隆 prompt

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            x_vector_only: 是否仅使用 x_vector

        Returns:
            VoiceClonePromptItem 列表
        """
        if not self.model:
            from tts.exceptions import TTSModelNotLoadedError

            raise TTSModelNotLoadedError("模型未加载")

        try:
            logger.info("正在提取声音特征...")

            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only
            )

            logger.info("✓ 声音特征提取完成")
            return prompt_items

        except Exception as e:
            logger.error(f"✗ 特征提取失败: {str(e)}")
            raise

    def create_and_save_prompt_features(
        self, ref_audio: str, ref_text: str, save_path: str, x_vector_only: bool = False
    ) -> bool:
        """
        创建并保存 prompt 特征到文件

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            save_path: 保存路径（.pt 文件）
            x_vector_only: 是否仅使用 x_vector

        Returns:
            bool: 是否成功
        """
        try:
            prompt_item = self.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only=x_vector_only
            )

            from tts.prompt_serializer import save_prompt_features

            metadata = {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only": x_vector_only,
            }

            return save_prompt_features(prompt_item, save_path, metadata)

        except Exception as e:
            logger.error(f"创建并保存特征失败: {e}")
            return False
