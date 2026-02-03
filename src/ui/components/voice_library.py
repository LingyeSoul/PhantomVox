"""
声音库管理器

跨页面共享的声音预设和克隆库管理
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Voice Design 内置预设
VOICE_DESIGN_PRESETS = {
    "温柔女声": "体现温柔细腻的女性声音，音调柔和，语速舒缓，营造出温暖、亲切的听觉效果。",
    "活泼少女": "体现活泼可爱的少女声音，音调偏高且富有弹性，语速轻快，营造出年轻、活力四射的听觉效果。",
    "磁性大叔": "体现磁性深沉的男性声音，音调偏低且富有质感，语速稳重，营造出成熟、有魅力的听觉效果。",
    "正太少年": "体现清脆自然的少年声音，音调适中，语速流畅，营造出纯真、少年感的听觉效果。",
    "知性御姐": "体现成熟知性的女性声音，音调中低，语速从容，营造出优雅、有韵味的听觉效果。",
    "沉稳长者": "体现稳重沧桑的长者声音，音调低沉，语速缓慢有力，营造出阅历丰富、值得信赖的听觉效果。",
}

# Custom Voice 说话人列表
CUSTOM_VOICE_SPEAKERS = [
    "Vivian",      # 明亮略带个性的年轻女性
    "Serena",      # 温柔温柔的年轻女性
    "Uncle_Fu",    # 沉稳的中年男性
    "Dylan",       # 清晰自然的北京男性
    "Eric",        # 活泼的成都男性
    "Ryan",        # 动感节奏强的英语男性
    "Aiden",       # 阳光美国男性
    "Ono_Anna",    # 活泼日本女性
    "Sohee"        # 温暖韩国女性
]

# 支持的语言列表
SUPPORTED_LANGUAGES = {
    "Chinese": "中文",
    "English": "英语",
    "Japanese": "日语",
    "Korean": "韩语",
    "Auto": "自动检测"
}


class VoiceLibrary:
    """跨页面共享的声音库管理器"""

    def __init__(self, config_manager: ConfigManager):
        """
        初始化声音库管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager

        # 加载各类数据
        self.design_presets = self._load_design_presets()
        self.clone_library = self._load_clone_library()
        self.recent_instructs = self._load_recent_instructs()

        logger.info("声音库管理器初始化完成")

    # ========== Voice Design 预设管理 ==========

    def _load_design_presets(self) -> Dict[str, str]:
        """加载声音设计预设"""
        user_presets = self.config_manager.get(
            "voice_design.design_presets",
            {}
        )
        # 合并内置预设和用户自定义预设
        return {**VOICE_DESIGN_PRESETS, **user_presets}

    def get_design_preset(self, name: str) -> Optional[str]:
        """获取声音设计预设"""
        return self.design_presets.get(name)

    def get_all_design_presets(self) -> Dict[str, str]:
        """获取所有声音设计预设"""
        return self.design_presets

    def add_design_preset(self, name: str, description: str) -> bool:
        """
        添加新的声音设计预设

        Args:
            name: 预设名称
            description: 预设描述

        Returns:
            是否成功添加
        """
        try:
            # 获取用户自定义预设
            user_presets = self.config_manager.get(
                "voice_design.design_presets",
                {}
            )

            # 添加新预设
            user_presets[name] = description

            # 保存到配置
            self.config_manager.set("voice_design.design_presets", user_presets)

            # 更新内存中的预设
            self.design_presets = {**VOICE_DESIGN_PRESETS, **user_presets}

            logger.info(f"添加声音设计预设: {name}")
            return True

        except Exception as e:
            logger.error(f"添加声音设计预设失败: {str(e)}")
            return False

    def remove_design_preset(self, name: str) -> bool:
        """
        删除声音设计预设（只能删除用户自定义的预设）

        Args:
            name: 预设名称

        Returns:
            是否成功删除
        """
        # 不能删除内置预设
        if name in VOICE_DESIGN_PRESETS:
            logger.warning(f"不能删除内置预设: {name}")
            return False

        try:
            user_presets = self.config_manager.get(
                "voice_design.design_presets",
                {}
            )

            if name in user_presets:
                del user_presets[name]
                self.config_manager.set("voice_design.design_presets", user_presets)

                # 更新内存中的预设
                self.design_presets = {**VOICE_DESIGN_PRESETS, **user_presets}

                logger.info(f"删除声音设计预设: {name}")
                return True

            return False

        except Exception as e:
            logger.error(f"删除声音设计预设失败: {str(e)}")
            return False

    def save_design_history(self, name: str, description: str) -> bool:
        """
        保存声音设计到历史记录

        Args:
            name: 设计名称
            description: 设计描述

        Returns:
            是否成功保存
        """
        try:
            history = self.config_manager.get(
                "voice_design.design_history",
                []
            )

            # 添加到历史记录前面
            history.insert(0, {
                "name": name,
                "description": description,
                "timestamp": datetime.now().isoformat()
            })

            # 限制历史记录数量（最多50条）
            history = history[:50]

            self.config_manager.set("voice_design.design_history", history)
            logger.info(f"保存声音设计历史: {name}")
            return True

        except Exception as e:
            logger.error(f"保存声音设计历史失败: {str(e)}")
            return False

    def get_design_history(self, limit: int = 10) -> List[dict]:
        """
        获取声音设计历史记录

        Args:
            limit: 返回记录数量限制

        Returns:
            历史记录列表
        """
        history = self.config_manager.get(
            "voice_design.design_history",
            []
        )
        return history[:limit]

    # ========== Voice Clone 克隆库管理 ==========

    def _load_clone_library(self) -> List[dict]:
        """加载克隆声音库"""
        return self.config_manager.get(
            "voice_clone.clone_library",
            []
        )

    def get_all_clones(self) -> List[dict]:
        """获取所有克隆声音"""
        return self.clone_library

    def get_clone(self, clone_id: str) -> Optional[dict]:
        """根据ID获取克隆声音"""
        for clone in self.clone_library:
            if clone["id"] == clone_id:
                return clone
        return None

    def add_clone(
        self,
        name: str,
        ref_audio: str,
        ref_text: str
    ) -> Optional[str]:
        """
        添加新的克隆声音

        Args:
            name: 克隆名称
            ref_audio: 参考音频路径
            ref_text: 参考文本

        Returns:
            克隆ID，失败返回None
        """
        try:
            clone_id = f"clone_{int(time.time())}"

            clone = {
                "id": clone_id,
                "name": name,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "created_at": datetime.now().isoformat()
            }

            self.clone_library.append(clone)
            self.config_manager.set("voice_clone.clone_library", self.clone_library)

            logger.info(f"添加克隆声音: {name} ({clone_id})")
            return clone_id

        except Exception as e:
            logger.error(f"添加克隆声音失败: {str(e)}")
            return None

    def remove_clone(self, clone_id: str) -> bool:
        """
        删除克隆声音

        Args:
            clone_id: 克隆ID

        Returns:
            是否成功删除
        """
        try:
            original_length = len(self.clone_library)
            self.clone_library = [
                c for c in self.clone_library if c["id"] != clone_id
            ]

            if len(self.clone_library) < original_length:
                self.config_manager.set(
                    "voice_clone.clone_library",
                    self.clone_library
                )
                logger.info(f"删除克隆声音: {clone_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"删除克隆声音失败: {str(e)}")
            return False

    # ========== Custom Voice 情感指令管理 ==========

    def _load_recent_instructs(self) -> List[str]:
        """加载最近使用的情感指令"""
        return self.config_manager.get(
            "custom_voice.recent_instructs",
            []
        )

    def add_recent_instruct(self, instruct: str) -> bool:
        """
        添加最近使用的情感指令

        Args:
            instruct: 情感指令文本

        Returns:
            是否成功保存
        """
        try:
            if not instruct or not instruct.strip():
                return False

            recent = self.config_manager.get(
                "custom_voice.recent_instructs",
                []
            )

            # 去重并添加到前面
            instruct = instruct.strip()
            if instruct in recent:
                recent.remove(instruct)
            recent.insert(0, instruct)

            # 限制数量（最多20条）
            recent = recent[:20]

            self.config_manager.set("custom_voice.recent_instructs", recent)
            self.recent_instructs = recent

            logger.debug(f"添加最近指令: {instruct[:30]}...")
            return True

        except Exception as e:
            logger.error(f"保存最近指令失败: {str(e)}")
            return False

    def get_recent_instructs(self, limit: int = 10) -> List[str]:
        """
        获取最近使用的情感指令

        Args:
            limit: 返回数量限制

        Returns:
            情感指令列表
        """
        return self.recent_instructs[:limit]

    # ========== 辅助方法 ==========

    @staticmethod
    def get_custom_voice_speakers() -> List[str]:
        """获取 Custom Voice 说话人列表"""
        return CUSTOM_VOICE_SPEAKERS.copy()

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """获取支持的语言列表"""
        return SUPPORTED_LANGUAGES.copy()
