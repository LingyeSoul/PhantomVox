"""
声音库管理器

跨页面共享的声音预设和克隆库管理
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config.config_manager import ConfigManager
from tts.vocal_data_manager import VocalDataManager
from tts.prompt_serializer import save_prompt_features

logger = logging.getLogger(__name__)

# 收藏相关常量
MAX_FAVORITES = 50  # 收藏最大数量
MAX_DESIGN_HISTORY = 50  # 设计历史最大数量

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


class FavoriteManager:
    """通用收藏管理器 - 消除重复代码"""

    def __init__(self, vocal_manager, content_key: str, save_method, load_method,
                 item_type_name: str = "item", max_items: int = MAX_FAVORITES):
        """
        初始化收藏管理器

        Args:
            vocal_manager: VocalDataManager 实例
            content_key: 内容字段名 ("instruct" 或 "description")
            save_method: 保存方法
            load_method: 加载方法
            item_type_name: 项目类型名称（用于日志）
            max_items: 最大收藏数量
        """
        self.vocal_manager = vocal_manager
        self.content_key = content_key
        self.save_method = save_method
        self.items = load_method()
        self.item_type_name = item_type_name
        self.max_items = max_items

    def add(self, name: str, content: str) -> bool:
        """
        添加收藏

        Args:
            name: 名称
            content: 内容

        Returns:
            是否成功保存
        """
        try:
            if not content or not content.strip():
                return False

            content = content.strip()

            # 去重：检查是否已存在相同内容
            for item in self.items:
                if item[self.content_key] == content:
                    logger.debug(f"该{self.item_type_name}已收藏")
                    return False

            # 添加到列表前面
            self.items.insert(0, {
                "name": name,
                self.content_key: content
            })

            # 限制数量
            self.items = self.items[:self.max_items]

            # 保存
            self.save_method(self.items)
            logger.info(f"添加收藏{self.item_type_name}: {name}")
            return True

        except Exception as e:
            logger.error(f"保存收藏{self.item_type_name}失败: {str(e)}")
            return False

    def update(self, old_content: str, new_name: str, new_content: str) -> bool:
        """
        更新收藏

        Args:
            old_content: 旧的内容（用于查找）
            new_name: 新的名称
            new_content: 新的内容

        Returns:
            是否成功更新
        """
        try:
            # 查找并更新
            for i, item in enumerate(self.items):
                if item[self.content_key] == old_content:
                    self.items[i] = {
                        "name": new_name,
                        self.content_key: new_content
                    }
                    # 保存
                    self.save_method(self.items)
                    logger.info(f"更新收藏{self.item_type_name}: {new_name}")
                    return True
            return False

        except Exception as e:
            logger.error(f"更新收藏{self.item_type_name}失败: {str(e)}")
            return False

    def remove(self, content: str) -> bool:
        """
        移除收藏

        Args:
            content: 内容

        Returns:
            是否成功移除
        """
        try:
            original_length = len(self.items)
            self.items = [
                item for item in self.items
                if item[self.content_key] != content
            ]

            if len(self.items) < original_length:
                self.save_method(self.items)
                logger.info(f"移除收藏{self.item_type_name}")
                return True
            return False

        except Exception as e:
            logger.error(f"移除收藏{self.item_type_name}失败: {str(e)}")
            return False

    def get_all(self, limit: int = None) -> List[dict]:
        """
        获取所有收藏

        Args:
            limit: 返回数量限制（None表示返回全部）

        Returns:
            收藏列表
        """
        if limit is None:
            return self.items.copy()
        return self.items[:limit]

    def is_favorite(self, content: str) -> bool:
        """
        检查是否已收藏

        Args:
            content: 内容

        Returns:
            是否已收藏
        """
        return any(item[self.content_key] == content for item in self.items)


class VoiceLibrary:
    """跨页面共享的声音库管理器"""

    def __init__(self, config_manager: ConfigManager):
        """
        初始化声音库管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager

        # 新增：Vocal 数据管理器
        project_root = Path(__file__).parent.parent.parent.parent
        self.vocal_manager = VocalDataManager(str(project_root / "vocal"))

        # 加载各类数据
        self.design_presets = self._load_design_presets()
        self.clone_library = self._load_clone_library()

        # 使用 FavoriteManager 管理收藏（消除重复代码）
        self.instruct_manager = FavoriteManager(
            self.vocal_manager,
            content_key="instruct",
            save_method=self.vocal_manager.save_favorite_instructs,
            load_method=self.vocal_manager.load_favorite_instructs,
            item_type_name="情感指令",
            max_items=MAX_FAVORITES
        )
        self.design_manager = FavoriteManager(
            self.vocal_manager,
            content_key="description",
            save_method=self.vocal_manager.save_favorite_designs,
            load_method=self.vocal_manager.load_favorite_designs,
            item_type_name="设计描述",
            max_items=MAX_FAVORITES
        )

        # 保持向后兼容的属性访问
        self.favorite_instructs = self.instruct_manager.items
        self.favorite_designs = self.design_manager.items

        # 设计历史（仅内存存储，不持久化）
        self.design_history = []

        # 清空 config.json 中的旧历史数据
        old_history = self.config_manager.get("voice_design.design_history", [])
        if old_history:
            self.config_manager.set("voice_design.design_history", [])
            logger.info(f"✓ 已清空 config.json 中的旧设计历史数据（{len(old_history)} 条）")

        logger.info("声音库管理器初始化完成")

    # ========== Voice Design 预设管理 ==========

    def _load_design_presets(self) -> Dict[str, str]:
        """加载声音设计预设（从 vocal/presets 加载）"""
        user_presets = self.vocal_manager.load_design_presets()

        # 如果 vocal 中没有数据，尝试从 config.json 迁移
        if not user_presets:
            legacy_presets = self.config_manager.get("voice_design.design_presets", {})
            if legacy_presets:
                logger.info(f"检测到旧格式设计预设数据，开始迁移 {len(legacy_presets)} 条...")
                self.vocal_manager.save_design_presets(legacy_presets)
                # 迁移完成后，清空 config.json 中的旧数据
                self.config_manager.set("voice_design.design_presets", {})
                logger.info(f"✓ 已清空 config.json 中的旧设计预设数据")
                user_presets = legacy_presets

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
            # 获取当前用户预设
            user_presets = self.vocal_manager.load_design_presets()

            # 添加新预设
            user_presets[name] = description

            # 保存到 vocal
            self.vocal_manager.save_design_presets(user_presets)

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
            user_presets = self.vocal_manager.load_design_presets()

            if name in user_presets:
                del user_presets[name]
                # 保存到 vocal
                self.vocal_manager.save_design_presets(user_presets)

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
        保存声音设计到历史记录（仅内存存储）

        Args:
            name: 设计名称
            description: 设计描述

        Returns:
            是否成功保存
        """
        try:
            design_id = f"design_{int(time.time())}"

            # 只保存到内存列表，不持久化
            self.design_history.insert(0, {
                "id": design_id,
                "name": name,
                "description": description,
                "timestamp": datetime.now().isoformat()
            })

            # 限制数量（最多N条）
            self.design_history = self.design_history[:MAX_DESIGN_HISTORY]

            logger.debug(f"保存声音设计到内存历史: {name}")
            return True

        except Exception as e:
            logger.error(f"保存声音设计历史失败: {str(e)}")
            return False

    def get_design_history(self, limit: int = 10) -> List[dict]:
        """
        获取声音设计历史记录（从内存获取）

        Args:
            limit: 返回记录数量限制

        Returns:
            历史记录列表
        """
        return self.design_history[:limit]

    # ========== Voice Clone 克隆库管理 ==========

    def _load_clone_library(self) -> List[dict]:
        """
        加载克隆声音库（支持新旧格式自动迁移）
        """
        # 1. 尝试从 vocal/clones 加载
        index_file = Path(self.vocal_manager.clones_dir) / ".index.json"

        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)

                clones = []
                for clone_id, meta in index.items():
                    full_data = self.vocal_manager.load_clone_data(clone_id)
                    if full_data:
                        clones.append({
                            "id": clone_id,
                            "name": meta["name"],
                            "ref_audio": full_data["ref_audio"],
                            "ref_text": full_data["ref_text"],
                            "created_at": meta["created_at"],
                            "has_prompt_features": meta.get("has_prompt_features", False)
                        })

                logger.info(f"从 vocal 加载了 {len(clones)} 个克隆")
                return clones
            except Exception as e:
                logger.error(f"从 vocal 加载克隆失败: {e}")

        # 2. 从 config.json 迁移（向后兼容）
        legacy_clones = self.config_manager.get("voice_clone.clone_library", [])

        if legacy_clones:
            logger.info("检测到旧格式克隆数据，开始迁移...")
            self._migrate_legacy_clones(legacy_clones)

            # 重新加载
            return self._load_clone_library()

        return []

    def _migrate_legacy_clones(self, legacy_clones: List[dict]):
        """
        迁移旧格式的克隆数据到 vocal 文件夹

        Args:
            legacy_clones: 旧格式的克隆列表
        """
        migrated_count = 0
        failed_count = 0

        for clone in legacy_clones:
            clone_id = clone["id"]
            name = clone["name"]
            ref_audio = clone["ref_audio"]
            ref_text = clone["ref_text"]

            # 检查原始音频是否存在
            if not os.path.exists(ref_audio):
                logger.warning(f"跳过丢失音频的克隆: {name}")
                failed_count += 1
                continue

            # 保存到 vocal（不包含特征，延迟计算）
            success = self.vocal_manager.save_clone_data(
                clone_id=clone_id,
                name=name,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                prompt_features=None,  # 迁移时不计算特征
                original_ref_audio=ref_audio
            )

            if success:
                logger.info(f"✓ 迁移克隆: {name}")
                migrated_count += 1
            else:
                logger.error(f"✗ 迁移克隆失败: {name}")
                failed_count += 1

        # 迁移完成后，清空 config.json 中的旧数据
        if migrated_count > 0:
            self.config_manager.set("voice_clone.clone_library", [])
            logger.info(f"✓ 已清空 config.json 中的旧克隆数据")

        logger.info(f"克隆数据迁移完成: 成功 {migrated_count} 个, 失败 {failed_count} 个")

    def get_all_clones(self) -> List[dict]:
        """获取所有克隆声音"""
        return self.clone_library

    def get_clone(self, clone_id: str) -> Optional[dict]:
        """
        根据ID获取克隆声音（增强版）

        返回完整数据，包括预计算的特征
        """
        # 先从内存查找
        for clone in self.clone_library:
            if clone["id"] == clone_id:
                # 如果有特征标记，尝试加载完整数据
                if clone.get("has_prompt_features"):
                    full_data = self.vocal_manager.load_clone_data(clone_id)
                    if full_data and full_data["prompt_features"]:
                        # 合并完整数据
                        return {**clone, "prompt_features": full_data["prompt_features"]}
                return clone

        # 如果内存中没有，尝试直接从文件系统加载
        full_data = self.vocal_manager.load_clone_data(clone_id)
        if full_data:
            return {
                "id": clone_id,
                "name": full_data["metadata"]["name"],
                "ref_audio": full_data["ref_audio"],
                "ref_text": full_data["ref_text"],
                "created_at": full_data["metadata"]["created_at"],
                "prompt_features": full_data.get("prompt_features")
            }

        return None

    def add_clone(
        self,
        name: str,
        ref_audio: str,
        ref_text: str,
        prompt_features=None,
        x_vector_only: bool = False
    ) -> Optional[str]:
        """
        添加新的克隆声音（增强版）

        Args:
            name: 克隆名称
            ref_audio: 参考音频路径
            ref_text: 参考文本
            prompt_features: 预计算的特征（可选，但推荐）
            x_vector_only: 是否仅使用 x_vector

        Returns:
            克隆ID，失败返回None
        """
        try:
            clone_id = f"clone_{int(time.time())}"

            # 保存到 vocal 文件系统
            success = self.vocal_manager.save_clone_data(
                clone_id=clone_id,
                name=name,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                prompt_features=prompt_features,
                x_vector_only=x_vector_only
            )

            if not success:
                return None

            # 更新内存中的列表
            clone = {
                "id": clone_id,
                "name": name,
                "ref_audio": str(self.vocal_manager.clones_dir / clone_id / "ref_audio.wav"),
                "ref_text": ref_text,
                "created_at": datetime.now().isoformat(),
                "has_prompt_features": prompt_features is not None
            }

            self.clone_library.append(clone)

            # 保存索引到 config.json（用于快速初始化）
            self.config_manager.set("voice_clone.clone_library", self.clone_library)

            logger.info(f"添加克隆声音: {name} ({clone_id})")
            return clone_id

        except Exception as e:
            logger.error(f"添加克隆声音失败: {str(e)}")
            return None

    def remove_clone(self, clone_id: str) -> bool:
        """
        删除克隆声音（重构版）

        Args:
            clone_id: 克隆ID

        Returns:
            是否成功删除
        """
        try:
            # 从文件系统删除
            success = self.vocal_manager.delete_clone_data(clone_id)

            if success:
                # 从内存列表移除
                original_length = len(self.clone_library)
                self.clone_library = [
                    c for c in self.clone_library if c["id"] != clone_id
                ]

                if len(self.clone_library) < original_length:
                    # 更新 config.json 索引
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

    def update_clone_features(self, clone_id: str, prompt_features) -> bool:
        """
        更新克隆的特征数据

        用于首次使用时计算并保存特征

        Args:
            clone_id: 克隆ID
            prompt_features: Prompt 特征对象

        Returns:
            bool: 是否成功
        """
        try:
            clone_data = self.vocal_manager.load_clone_data(clone_id)

            if not clone_data:
                return False

            # 保存特征
            clone_dir = Path(self.vocal_manager.clones_dir) / clone_id
            save_prompt_features(
                prompt_features,
                str(clone_dir / "prompt_features.pt")
            )

            # 更新元数据
            metadata = clone_data["metadata"]
            metadata["has_prompt_features"] = True

            with open(clone_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 更新索引
            self.vocal_manager._update_clone_index(clone_id, metadata)

            # 更新内存中的标记
            for clone in self.clone_library:
                if clone["id"] == clone_id:
                    clone["has_prompt_features"] = True
                    break

            logger.info(f"✓ 更新克隆特征: {clone_id}")
            return True

        except Exception as e:
            logger.error(f"更新克隆特征失败: {e}")
            return False

    # ========== Custom Voice 情感指令收藏管理 ==========

    def _load_favorite_instructs(self) -> List[dict]:
        """
        加载收藏的情感指令（委托给 FavoriteManager）

        Returns:
            List[dict]: 指令列表 [{"name": str, "instruct": str}, ...]
        """
        return self.instruct_manager.get_all()

    def add_favorite_instruct(self, name: str, instruct: str) -> bool:
        """
        添加收藏的情感指令（委托给 FavoriteManager）

        Args:
            name: 指令名称
            instruct: 情感指令文本

        Returns:
            是否成功保存
        """
        return self.instruct_manager.add(name, instruct)

    def update_favorite_instruct(self, old_instruct: str, new_name: str, new_instruct: str) -> bool:
        """
        更新收藏的情感指令（委托给 FavoriteManager）

        Args:
            old_instruct: 旧的指令内容（用于查找）
            new_name: 新的名称
            new_instruct: 新的指令内容

        Returns:
            是否成功更新
        """
        return self.instruct_manager.update(old_instruct, new_name, new_instruct)

    def remove_favorite_instruct(self, instruct: str) -> bool:
        """
        移除收藏的情感指令（委托给 FavoriteManager）

        Args:
            instruct: 情感指令文本

        Returns:
            是否成功移除
        """
        result = self.instruct_manager.remove(instruct)
        # 更新向后兼容的属性
        self.favorite_instructs = self.instruct_manager.items
        return result

    def get_favorite_instructs(self, limit: int = None) -> List[dict]:
        """
        获取收藏的情感指令（委托给 FavoriteManager）

        Args:
            limit: 返回数量限制（None表示返回全部）

        Returns:
            指令列表 [{"name": str, "instruct": str}, ...]
        """
        return self.instruct_manager.get_all(limit)

    def is_favorite_instruct(self, instruct: str) -> bool:
        """
        检查情感指令是否已收藏（委托给 FavoriteManager）

        Args:
            instruct: 情感指令文本

        Returns:
            是否已收藏
        """
        return self.instruct_manager.is_favorite(instruct)

    # ========== Voice Design 设计描述收藏管理 ==========

    def _load_favorite_designs(self) -> List[dict]:
        """
        加载收藏的设计描述（委托给 FavoriteManager）

        Returns:
            List[dict]: 设计列表 [{"name": str, "description": str}, ...]
        """
        return self.design_manager.get_all()

    def add_favorite_design(self, name: str, description: str) -> bool:
        """
        添加收藏的设计描述（委托给 FavoriteManager）

        Args:
            name: 设计名称
            description: 设计描述

        Returns:
            是否成功保存
        """
        return self.design_manager.add(name, description)

    def update_favorite_design(self, old_description: str, new_name: str, new_description: str) -> bool:
        """
        更新收藏的设计描述（委托给 FavoriteManager）

        Args:
            old_description: 旧的描述（用于查找）
            new_name: 新的名称
            new_description: 新的描述

        Returns:
            是否成功更新
        """
        return self.design_manager.update(old_description, new_name, new_description)

    def remove_favorite_design(self, description: str) -> bool:
        """
        移除收藏的设计描述（委托给 FavoriteManager）

        Args:
            description: 设计描述

        Returns:
            是否成功移除
        """
        result = self.design_manager.remove(description)
        # 更新向后兼容的属性
        self.favorite_designs = self.design_manager.items
        return result

    def get_favorite_designs(self, limit: int = None) -> List[dict]:
        """
        获取收藏的设计描述

        Args:
            limit: 返回数量限制（None表示返回全部）

        Returns:
            设计列表 [{"name": str, "description": str}, ...]
        """
        if limit is None:
            return self.favorite_designs.copy()
        return self.favorite_designs[:limit]

    # ========== 辅助方法 ==========

    @staticmethod
    def get_custom_voice_speakers() -> List[str]:
        """获取 Custom Voice 说话人列表"""
        return CUSTOM_VOICE_SPEAKERS.copy()

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """获取支持的语言列表"""
        return SUPPORTED_LANGUAGES.copy()
