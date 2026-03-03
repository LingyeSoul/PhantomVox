"""
Vocal 数据文件系统管理器

负责管理 vocal/ 文件夹中的所有语音数据
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_NAME_MAX_LENGTH = 15  # 默认名称最大长度（字符）
DEFAULT_DESCRIPTION_MAX_LENGTH = 10  # 默认描述最大长度（字符）
MAX_FAVORITES = 50  # 收藏最大数量
MAX_AUDIO_FILE_SIZE = 100 * 1024 * 1024  # 音频文件最大大小：100 MB（防止资源耗尽攻击）
MAX_NAME_LENGTH = 100  # 收藏名称最大长度（字符）
MAX_CONTENT_LENGTH = 5000  # 收藏内容最大长度（字符）


class VocalDataManager:
    """语音数据文件系统管理器"""

    def __init__(self, vocal_root: str):
        """
        初始化 Vocal 数据管理器

        Args:
            vocal_root: vocal 文件夹的根路径
        """
        self.vocal_root = Path(vocal_root)
        self.clones_dir = self.vocal_root / "clones"
        self.designs_dir = self.vocal_root / "designs"
        self.presets_dir = self.vocal_root / "presets"
        self.instructs_dir = self.vocal_root / "instructs"

        # 初始化目录结构
        self._init_directories()

    def _init_directories(self):
        """初始化目录结构"""
        for dir_path in [self.clones_dir, self.designs_dir, self.presets_dir, self.instructs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # ========== 克隆管理 ==========

    def save_clone_data(
        self,
        clone_id: str,
        name: str,
        ref_audio_path: str,
        ref_text: str,
        prompt_features: Optional[Any] = None,
        x_vector_only: bool = False,
        original_ref_audio: Optional[str] = None
    ) -> bool:
        """
        保存完整的克隆数据

        Args:
            clone_id: 克隆ID
            name: 克隆名称
            ref_audio_path: 参考音频路径
            ref_text: 参考文本
            prompt_features: 预计算的特征（可选）
            x_vector_only: 是否仅使用 x_vector
            original_ref_audio: 原始参考音频路径（用于记录）

        Returns:
            bool: 是否成功
        """
        clone_dir = self.clones_dir / clone_id
        clone_dir.mkdir(exist_ok=True)

        try:
            # 1. 保存元数据
            metadata = {
                "id": clone_id,
                "name": name,
                "created_at": datetime.now().isoformat(),
                "x_vector_only": x_vector_only,
                "has_prompt_features": prompt_features is not None,
                "features_version": "1.0" if prompt_features else None,
                "original_ref_audio": original_ref_audio or ref_audio_path
            }

            with open(clone_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 2. 复制/转换参考音频
            if os.path.exists(ref_audio_path):
                target_audio = clone_dir / "ref_audio.wav"
                if not ref_audio_path.lower().endswith('.wav'):
                    self._convert_to_wav(ref_audio_path, target_audio)
                else:
                    shutil.copy2(ref_audio_path, target_audio)
            else:
                logger.warning(f"参考音频文件不存在: {ref_audio_path}")
                return False

            # 3. 保存参考文本
            with open(clone_dir / "ref_text.txt", "w", encoding="utf-8") as f:
                f.write(ref_text)

            # 4. 保存特征数据（如果有）
            if prompt_features:
                from tts.prompt_serializer import save_prompt_features
                save_prompt_features(
                    prompt_features,
                    str(clone_dir / "prompt_features.pt")
                )

            # 5. 更新索引
            self._update_clone_index(clone_id, metadata)

            logger.info(f"✓ 克隆数据已保存: {name} ({clone_id})")
            return True

        except Exception as e:
            logger.error(f"✗ 保存克隆数据失败: {e}")
            # 清理失败的目录
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            return False

    def load_clone_data(self, clone_id: str) -> Optional[dict]:
        """
        加载克隆数据

        Args:
            clone_id: 克隆ID

        Returns:
            dict: 包含 metadata, ref_audio, ref_text, prompt_features 的字典
        """
        clone_dir = self.clones_dir / clone_id

        if not clone_dir.exists():
            return None

        try:
            # 加载元数据
            with open(clone_dir / "metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 加载参考文本
            with open(clone_dir / "ref_text.txt", "r", encoding="utf-8") as f:
                ref_text = f.read().strip()

            # 检查音频文件
            ref_audio = str(clone_dir / "ref_audio.wav")
            if not os.path.exists(ref_audio):
                logger.warning(f"克隆 {clone_id} 的音频文件丢失")
                return None

            # 尝试加载特征数据（可选，支持 .safetensors 和 .pt 格式）
            prompt_features = None
            # 使用无扩展名路径，让 load_prompt_features 自动检测格式
            features_base = clone_dir / "prompt_features"
            if features_base.with_suffix('.safetensors').exists() or features_base.with_suffix('.pt').exists():
                try:
                    from tts.prompt_serializer import load_prompt_features
                    prompt_features = load_prompt_features(str(features_base))
                except Exception as e:
                    logger.warning(f"加载特征数据失败: {e}")

            return {
                "metadata": metadata,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "prompt_features": prompt_features
            }

        except Exception as e:
            logger.error(f"加载克隆数据失败: {e}")
            return None

    def delete_clone_data(self, clone_id: str) -> bool:
        """
        删除克隆数据

        Args:
            clone_id: 克隆ID

        Returns:
            bool: 是否成功
        """
        clone_dir = self.clones_dir / clone_id

        if clone_dir.exists():
            try:
                shutil.rmtree(clone_dir)
                self._remove_from_clone_index(clone_id)
                logger.info(f"✓ 克隆数据已删除: {clone_id}")
                return True
            except Exception as e:
                logger.error(f"删除克隆数据失败: {e}")
                return False
        return False

    def _convert_to_wav(self, source_path: str, target_path: Path):
        """
        安全地转换音频为 WAV 格式（带路径验证和音频属性验证）

        Args:
            source_path: 源音频路径
            target_path: 目标 WAV 路径

        Raises:
            FileNotFoundError: 源文件不存在
            ValueError: 音频验证失败（路径、大小、时长等）
            RuntimeError: 转换失败
        """
        # 导入安全工具函数
        try:
            from .audio_utils import (
                validate_audio_path,
                validate_file_exists_and_readable,
                validate_file_size,
                validate_audio_extension,
                convert_to_wav as utils_convert_to_wav,
                QWEN_TTS_SAMPLE_RATE
            )
        except ImportError:
            logger.warning("audio_utils 模块不可用，使用旧方法")
            # 回退到旧方法（带基本验证）
            self._convert_to_wav_legacy(source_path, target_path)
            return

        try:
            # 路径规范化和验证（不限制目录位置，允许任意文件路径）
            # 桌面应用应允许用户从任何位置选择音频文件
            safe_source_path = validate_audio_path(source_path, allowed_directories=None)

            # 文件存在性和可读性验证
            validate_file_exists_and_readable(safe_source_path)

            # 文件大小验证
            validate_file_size(safe_source_path)

            # 文件扩展名验证
            validate_audio_extension(safe_source_path)

            logger.info(f"开始转换音频: {safe_source_path}")

            # 使用安全的转换函数（包含时长、采样率验证）
            utils_convert_to_wav(
                source_path=safe_source_path,
                target_path=str(target_path),
                sample_rate=QWEN_TTS_SAMPLE_RATE,
                allowed_directories=None  # 允许任意源路径
            )

            logger.info(f"✓ 音频转换成功: {target_path}")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error(f"音频转换失败: {e}")
            raise
        except Exception as e:
            logger.error(f"音频转换失败（未预期的错误）: {e}")
            # 记录详细错误到日志，返回通用错误给用户
            raise RuntimeError("音频转换失败，请检查文件格式是否正确")

    def _convert_to_wav_legacy(self, source_path: str, target_path: Path):
        """
        旧版音频转换方法（仅当 audio_utils 不可用时使用）

        Args:
            source_path: 源音频路径
            target_path: 目标 WAV 路径
        """
        # 基本安全验证
        abs_source_path = os.path.abspath(source_path)
        normalized_path = os.path.normpath(abs_source_path)

        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"音频文件不存在: {source_path}")

        file_size = os.path.getsize(normalized_path)
        if file_size == 0:
            raise ValueError(f"音频文件为空: {source_path}")
        if file_size > MAX_AUDIO_FILE_SIZE:
            raise ValueError(
                f"音频文件过大: {file_size / 1024 / 1024:.2f} MB "
                f"(最大允许 {MAX_AUDIO_FILE_SIZE / 1024 / 1024:.0f} MB)"
            )

        logger.info(f"开始转换音频（旧方法）: {normalized_path}")

        # 方法1：使用 pydub (ffmpeg)
        try:
            from pydub import AudioSegment

            logger.debug("使用 pydub (ffmpeg) 转换音频")

            # 加载音频
            audio = AudioSegment.from_file(normalized_path)

            # 基本验证
            duration_seconds = len(audio) / 1000.0
            if duration_seconds > 300:  # 5分钟
                raise ValueError(f"音频时长过长: {duration_seconds / 60:.2f} 分钟")

            # 设置输出参数：24kHz, 单声道
            audio = audio.set_frame_rate(24000)
            audio = audio.set_channels(1)

            # 使用临时文件确保原子性写入
            temp_path = str(target_path) + '.tmp'
            audio.export(temp_path, format="wav")

            # 原子性重命名
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)

            logger.info(f"✓ 音频转换成功 (pydub): {target_path}")
            return

        except ImportError:
            logger.debug("pydub 未安装，尝试其他方法")
        except Exception as e:
            logger.warning(f"pydub 转换失败: {e}，尝试其他方法")
            temp_path = str(target_path) + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        # 方法2：使用 librosa（备选）
        try:
            import librosa
            import soundfile as sf

            logger.debug("使用 librosa 转换音频")
            audio, sr = librosa.load(normalized_path, sr=24000, mono=True)

            sf.write(target_path, audio, sr)
            logger.info(f"✓ 音频转换成功 (librosa): {target_path}")
            return

        except ImportError:
            logger.debug("librosa 未安装")
        except Exception as e:
            logger.error(f"librosa 转换失败: {e}")

        # 如果所有方法都失败，抛出异常
        raise RuntimeError(
            f"无法转换音频文件: {source_path}\n"
            f"请确保：\n"
            f"  1. 已安装 pydub（pip install pydub）\n"
            f"  2. 音频文件格式正确（支持 mp3, wav, m4a, ogg 等）\n"
            f"  3. 音频文件未损坏"
        )

    def _update_clone_index(self, clone_id: str, metadata: dict):
        """
        更新克隆索引

        Args:
            clone_id: 克隆ID
            metadata: 元数据字典
        """
        index_file = self.clones_dir / ".index.json"

        index = {}
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

        index[clone_id] = {
            "name": metadata["name"],
            "created_at": metadata["created_at"],
            "has_prompt_features": metadata["has_prompt_features"]
        }

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def _remove_from_clone_index(self, clone_id: str):
        """
        从索引中移除克隆

        Args:
            clone_id: 克隆ID
        """
        index_file = self.clones_dir / ".index.json"

        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            if clone_id in index:
                del index[clone_id]

                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)

    def get_clone_index(self) -> Dict[str, dict]:
        """
        获取克隆索引

        Returns:
            dict: 克隆索引字典
        """
        index_file = self.clones_dir / ".index.json"

        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ========== 设计历史管理 ==========

    # ========== 设计预设管理 ==========

    def save_design_presets(self, presets: Dict[str, str]) -> bool:
        """
        保存设计预设

        Args:
            presets: 预设字典 {name: description}

        Returns:
            bool: 是否成功
        """
        try:
            with open(self.presets_dir / "design_presets.json", "w", encoding="utf-8") as f:
                json.dump(presets, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存设计预设失败: {e}")
            return False

    def load_design_presets(self) -> Dict[str, str]:
        """
        加载设计预设（自动迁移旧数据）

        Returns:
            dict: 预设字典
        """
        # 尝试从 vocal/presets 加载
        file_path = self.presets_dir / "design_presets.json"

        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载设计预设失败: {e}")

        return {}

    # ========== 情感指令收藏管理 ==========

    def save_favorite_instructs(self, instructs: List[dict]) -> bool:
        """
        保存收藏的情感指令

        Args:
            instructs: 指令列表 [{"name": str, "instruct": str}, ...]

        Returns:
            bool: 是否成功
        """
        try:
            with open(self.instructs_dir / "favorite_instructs.json", "w", encoding="utf-8") as f:
                json.dump(instructs, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存收藏情感指令失败: {e}")
            return False

    def load_favorite_instructs(self) -> List[dict]:
        """
        加载收藏的情感指令（自动迁移旧数据）

        Returns:
            list: 指令列表 [{"name": str, "instruct": str}, ...]
        """
        file_path = self.instructs_dir / "favorite_instructs.json"

        # 如果新文件存在，直接加载
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 如果是旧格式（字符串列表），需要迁移
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                    return self._migrate_favorite_instructs(data)
                return data
            except Exception as e:
                logger.error(f"加载收藏情感指令失败: {e}")

        # 检查是否有旧文件需要迁移
        old_file_path = self.instructs_dir / "recent_instructs.json"
        if old_file_path.exists():
            try:
                logger.info("检测到旧的情感指令文件，开始迁移...")
                with open(old_file_path, "r", encoding="utf-8") as f:
                    old_instructs = json.load(f)

                # 迁移到新格式
                new_data = self._migrate_favorite_instructs(old_instructs)

                # 保存到新文件
                if self.save_favorite_instructs(new_data):
                    logger.info(f"✓ 已迁移 {len(old_instructs)} 条情感指令到新格式")
                    # 备份旧文件（重命名为 .bak）
                    backup_path = self.instructs_dir / "recent_instructs.json.bak"
                    old_file_path.rename(backup_path)
                    logger.info(f"✓ 旧文件已备份为: {backup_path.name}")
                    return new_data
            except Exception as e:
                logger.error(f"迁移情感指令失败: {e}")

        return []

    def _migrate_favorite_instructs(self, old_instructs: List[str]) -> List[dict]:
        """
        将旧格式（字符串列表）迁移到新格式（字典列表）

        Args:
            old_instructs: 旧格式的指令列表

        Returns:
            新格式的指令列表
        """
        new_data = []
        for instruct in old_instructs:
            # 使用前 N 个字符作为默认名称
            name = instruct[:DEFAULT_NAME_MAX_LENGTH] + "..." if len(instruct) > DEFAULT_NAME_MAX_LENGTH else instruct
            new_data.append({
                "name": name,
                "instruct": instruct
            })
        return new_data

    # ========== 设计描述收藏管理 ==========

    def save_favorite_designs(self, designs: List[dict]) -> bool:
        """
        保存收藏的设计描述

        Args:
            designs: 设计列表 [{"name": str, "description": str}, ...]

        Returns:
            bool: 是否成功
        """
        try:
            with open(self.designs_dir / "favorite_designs.json", "w", encoding="utf-8") as f:
                json.dump(designs, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存收藏设计失败: {e}")
            return False

    def load_favorite_designs(self) -> List[dict]:
        """
        加载收藏的设计描述

        Returns:
            list: 设计列表 [{"name": str, "description": str}, ...]
        """
        file_path = self.designs_dir / "favorite_designs.json"

        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载收藏设计失败: {e}")

        return []
