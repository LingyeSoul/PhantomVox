"""
Qwen3-TTS 引擎封装

提供文本转语音的核心功能，支持三种模式：
1. Custom Voice - 使用预设说话人 + 情感指令
2. Voice Design - 通过自然语言描述设计声音
3. Voice Clone - 使用参考音频克隆声音
"""

from qwen_tts import Qwen3TTSModel
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class QwenEngine:
    """Qwen3-TTS 引擎封装"""

    # 模型类型常量
    MODEL_CUSTOM_VOICE = "CustomVoice"
    MODEL_VOICE_DESIGN = "VoiceDesign"
    MODEL_BASE = "Base"

    def __init__(self, model_path=None, model_type=None, device="cuda:0", dtype=None, attn_implementation=None):
        """
        初始化 Qwen TTS 引擎

        Args:
            model_path: 模型路径
            model_type: 模型类型 (CustomVoice/VoiceDesign/Base)
            device: 运行设备 ("cpu", "cuda", 或 "cuda:0")
            dtype: 数据类型（可选，传递给 qwen-tts）
            attn_implementation: 注意力实现（可选，传递给 qwen-tts）
        """
        self.model = None
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self._load_model()

    def _load_model(self):
        """加载 Qwen3-TTS 模型"""
        try:
            logger.info(f"正在加载 Qwen3-TTS 模型 ({self.model_type or '默认'})...")
            logger.info(f"设备: {self.device}")

            # 构建模型加载参数
            model_kwargs = {"device_map": self.device}

            # 如果指定了 dtype，添加到参数中
            if self.dtype:
                model_kwargs["dtype"] = self.dtype
                logger.info(f"数据类型: {self.dtype}")

            # 如果指定了注意力实现，添加到参数中
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation
                logger.info(f"注意力实现: {self.attn_implementation}")

            if self.model_path:
                # 使用自定义路径
                logger.info(f"加载模型: {self.model_path}")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            else:
                # 使用 HuggingFace 模型 ID
                model_id = f"Qwen/Qwen3-TTS-12Hz-1.7B-{self.model_type}" if self.model_type else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                logger.info(f"使用模型 ID: {model_id}")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    **model_kwargs
                )

            logger.info("✓ Qwen3-TTS 模型加载完成")

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {str(e)}")
            raise

    # ========== Custom Voice 模式 ==========

    def custom_voice_synthesize(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Custom Voice 模式生成语音

        Args:
            text: 输入文本
            speaker: 说话人 (Vivian/Serena/Uncle_Fu/Dylan/Eric/Ryan/Aiden/Ono_Anna/Sohee)
            language: 语言 (Chinese/English/Japanese/Korean/Auto)
            instruct: 情感指令
            **kwargs: 其他参数 (speed_factor, pitch_factor 等)

        Returns:
            (audio_data, sample_rate)
        """
        if not self.model:
            raise RuntimeError("模型未加载")

        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Custom Voice): {text[:50]}...")

            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                **kwargs
            )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    def get_supported_speakers(self) -> list:
        """获取支持的说话人列表"""
        if self.model:
            try:
                return self.model.get_supported_speakers()
            except:
                pass
        return ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

    def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        if self.model:
            try:
                return self.model.get_supported_languages()
            except:
                pass
        return ["Chinese", "English", "Japanese", "Korean", "Auto"]

    # ========== Voice Design 模式 ==========

    def voice_design_synthesize(
        self,
        text: str,
        design_prompt: str,
        language: str = "Chinese",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Voice Design 模式生成语音

        Args:
            text: 输入文本
            design_prompt: 声音设计描述
            language: 语言
            **kwargs: 其他参数

        Returns:
            (audio_data, sample_rate)
        """
        if not self.model:
            raise RuntimeError("模型未加载")

        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Voice Design): {text[:50]}...")

            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=design_prompt,
                **kwargs
            )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    # ========== Voice Clone 模式 ==========

    def voice_clone_synthesize(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        clone_prompt=None,
        x_vector_only: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        使用 Voice Clone 模式生成语音

        Args:
            text: 输入文本
            ref_audio: 参考音频路径
            ref_text: 参考文本
            clone_prompt: 已保存的 clone_prompt (可选)
            x_vector_only: 是否仅使用 x_vector (快速模式)
            **kwargs: 其他参数

        Returns:
            (audio_data, sample_rate)
        """
        if not self.model:
            raise RuntimeError("模型未加载")

        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        try:
            logger.info(f"正在生成语音 (Voice Clone): {text[:50]}...")

            if clone_prompt:
                # 使用已保存的 clone_prompt
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=clone_prompt,
                    **kwargs
                )
            else:
                # 使用新的参考音频
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                    **kwargs
                )

            logger.info("✓ 语音生成成功")
            return wavs[0], sr

        except Exception as e:
            logger.error(f"✗ 语音生成失败: {str(e)}")
            raise

    def create_voice_clone_prompt(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only: bool = False
    ):
        """
        创建可重用的声音克隆 prompt

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            x_vector_only: 是否仅使用 x_vector

        Returns:
            prompt_items (可传递给 generate_voice_clone 的 voice_clone_prompt 参数)
        """
        if not self.model:
            raise RuntimeError("模型未加载")

        try:
            logger.info("正在提取声音特征...")

            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )

            logger.info("✓ 声音特征提取完成")
            return prompt_items

        except Exception as e:
            logger.error(f"✗ 特征提取失败: {str(e)}")
            raise

    # ========== 兼容旧 API ==========

    def synthesize(self, text, voice="default", speed=1.0, pitch=1.0):
        """
        基础合成方法 (向后兼容)

        内部调用 custom_voice_synthesize

        Returns:
            audio_data (numpy array) - 只返回音频数据，采样率固定为 24000
        """
        # 将旧参数映射到新参数
        speaker_map = {
            "default": "Vivian",
            "female": "Serena",
            "male": "Uncle_Fu"
        }
        speaker = speaker_map.get(voice, "Vivian")

        audio_data, _ = self.custom_voice_synthesize(
            text=text,
            speaker=speaker,
            language="Chinese",
            speed_factor=speed,
            pitch_factor=pitch
        )
        return audio_data

    def get_available_voices(self):
        """获取可用的声音列表 (向后兼容)"""
        return ["default", "female", "male", "child", "elderly"]

    def clone_voice(self, audio_samples, text):
        """
        声音克隆（向后兼容，已废弃）

        建议使用 voice_clone_synthesize 方法
        """
        logger.warning("clone_voice 方法已废弃，建议使用 voice_clone_synthesize")
        # 尝试使用新方法
        return self.voice_clone_synthesize(
            text=text,
            ref_audio=audio_samples,
            ref_text="",  # 旧版本没有参考文本
            x_vector_only=True
        )
