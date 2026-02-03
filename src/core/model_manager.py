"""
PhantomVox 模型管理器

负责 TTS 模型的下载、安装、验证和更新
使用 ModelScope 进行模型下载
"""

import os
import asyncio
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Callable, List
import json
import shutil

logger = logging.getLogger(__name__)


class ModelInfo:
    """模型信息"""

    def __init__(
        self,
        model_id: str,
        name: str,
        size: str,
        repo_id: str,
        description: str = "",
        required: bool = False,
        dependencies: Optional[List[str]] = None
    ):
        self.model_id = model_id
        self.name = name
        self.size = size
        self.repo_id = repo_id  # ModelScope 仓库 ID
        self.description = description
        self.required = required  # 是否必需
        self.dependencies = dependencies or []  # 依赖的其他模型


class ModelManager:
    """模型管理器 - 基于 ModelScope"""

    # Qwen3-TTS 模型列表
    AVAILABLE_MODELS = {
        # ========== 分词器（必需） ==========
        "tokenizer-12hz": ModelInfo(
            model_id="tokenizer-12hz",
            name="Qwen3-TTS 分词器 (12Hz)",
            size="~50MB",
            repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
            description="12Hz 采样率分词器，文本编码必需",
            required=True
        ),

        # ========== 1.7B 系列模型 ==========
        "1.7b-customvoice": ModelInfo(
            model_id="1.7b-customvoice",
            name="Qwen3-TTS 1.7B 自定义声音",
            size="~3.4GB",
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            description="1.7B 参数模型，支持自定义声音克隆"
        ),

        "1.7b-voicedesign": ModelInfo(
            model_id="1.7b-voicedesign",
            name="Qwen3-TTS 1.7B 声音设计",
            size="~3.4GB",
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            description="1.7B 参数模型，高级声音设计功能"
        ),

        "1.7b-base": ModelInfo(
            model_id="1.7b-base",
            name="Qwen3-TTS 1.7B 基础版",
            size="~3.4GB",
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            description="1.7B 参数基础模型，推荐使用",
            dependencies=["tokenizer-12hz"]
        ),

        # ========== 0.6B 系列模型 ==========
        "0.6b-customvoice": ModelInfo(
            model_id="0.6b-customvoice",
            name="Qwen3-TTS 0.6B 自定义声音",
            size="~1.2GB",
            repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            description="0.6B 参数模型，轻量级自定义声音"
        ),

        "0.6b-base": ModelInfo(
            model_id="0.6b-base",
            name="Qwen3-TTS 0.6B 基础版",
            size="~1.2GB",
            repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            description="0.6B 参数基础模型，适合低配置设备",
            dependencies=["tokenizer-12hz"]
        ),
    }

    def __init__(self, models_dir: str = "./models", config_manager=None):
        """
        初始化模型管理器

        Args:
            models_dir: 模型存储目录
            config_manager: 配置管理器实例
        """
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.config_manager = config_manager
        self._download_status: Dict[str, str] = {}
        self._download_progress: Dict[str, float] = {}

    def get_installed_models(self) -> list:
        """获取已安装的模型列表"""
        installed = []

        for model_id in self.AVAILABLE_MODELS.keys():
            model_path = self.models_dir / model_id
            if model_path.exists() and self._is_model_complete(model_path):
                installed.append(model_id)

        return installed

    def _is_model_complete(self, model_path: Path) -> bool:
        """检查模型是否完整"""
        # 检查是否有文件（ModelScope 下载后会有多个文件）
        if not model_path.is_dir():
            return False

        # 检查是否有配置文件
        config_files = list(model_path.glob("*.json")) + list(model_path.glob("config.json"))
        if not config_files:
            return False

        # 检查是否有模型权重文件
        model_files = (
            list(model_path.glob("*.safetensors")) +
            list(model_path.glob("*.bin")) +
            list(model_path.glob("*.pth"))
        )

        return len(model_files) > 0

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.AVAILABLE_MODELS.get(model_id)

    def list_available_models(self) -> Dict[str, ModelInfo]:
        """列出所有可用模型"""
        return self.AVAILABLE_MODELS.copy()

    def _check_modelscope(self) -> bool:
        """检查 ModelScope 是否已安装"""
        try:
            # 使用 Python import 检查，比 subprocess 更可靠
            import modelscope
            # 尝试访问版本号以验证包是否完整
            _ = modelscope.__version__
            return True
        except ImportError:
            return False
        except AttributeError:
            # 包已安装但可能版本过旧或损坏
            return False

    async def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> bool:
        """
        下载模型 - 使用 ModelScope Python API

        Args:
            model_id: 模型 ID
            progress_callback: 进度回调函数 (model_id, progress, status)

        Returns:
            bool: 是否下载成功
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            logger.error(f"未找到模型: {model_id}")
            return False

        # 检查 ModelScope 是否已安装
        if not self._check_modelscope():
            error_msg = "ModelScope 未安装。请运行: pip install modelscope"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(model_id, 0, error_msg)
            return False

        model_path = self.models_dir / model_id
        model_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"开始下载模型: {model_info.name}")
            self._download_status[model_id] = "downloading"
            self._download_progress[model_id] = 0.0

            if progress_callback:
                progress_callback(model_id, 0.0, f"开始下载 {model_info.name}...")

            # 使用 ModelScope Python API 下载
            from modelscope.hub.snapshot_download import snapshot_download

            # 在线程池中执行下载，避免阻塞事件循环
            loop = asyncio.get_event_loop()

            def download_in_thread():
                return snapshot_download(
                    model_info.repo_id,
                    cache_dir=str(self.models_dir),
                    local_dir=str(model_path),
                    revision='master'
                    # 注意：ModelScope 的 snapshot_download 不支持自定义进度回调
                    # 它会在内部自动显示进度到终端
                )

            # 在后台线程中执行下载
            downloaded_path = await loop.run_in_executor(None, download_in_thread)

            if downloaded_path:
                logger.info(f"模型下载成功: {model_info.name}")
                self._download_status[model_id] = "completed"
                self._download_progress[model_id] = 100.0

                if progress_callback:
                    progress_callback(model_id, 100.0, f"✓ {model_info.name} 下载完成")

                return True
            else:
                raise Exception("下载返回空路径")

        except Exception as e:
            logger.error(f"模型下载失败: {str(e)}")
            self._download_status[model_id] = "failed"
            if progress_callback:
                progress_callback(model_id, self._download_progress.get(model_id, 0), f"失败: {str(e)}")
            return False

    async def download_model_with_dependencies(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> bool:
        """
        下载模型及其依赖

        Args:
            model_id: 模型 ID
            progress_callback: 进度回调函数

        Returns:
            bool: 是否全部下载成功
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False

        all_success = True

        # 先下载依赖
        for dep_id in model_info.dependencies:
            dep_info = self.get_model_info(dep_id)
            if not dep_info:
                continue

            # 检查依赖是否已安装
            dep_path = self.models_dir / dep_id
            if dep_path.exists() and self._is_model_complete(dep_path):
                logger.info(f"依赖 {dep_info.name} 已安装，跳过")
                continue

            if progress_callback:
                progress_callback(model_id, 0, f"下载依赖: {dep_info.name}")

            success = await self.download_model(dep_id, progress_callback)
            if not success:
                logger.error(f"依赖 {dep_info.name} 下载失败")
                all_success = False

        # 下载主模型
        if all_success:
            success = await self.download_model(model_id, progress_callback)
            if not success:
                all_success = False

        return all_success

    async def delete_model(self, model_id: str) -> bool:
        """
        删除模型

        Args:
            model_id: 模型 ID

        Returns:
            bool: 是否删除成功
        """
        model_path = self.models_dir / model_id

        try:
            if model_path.exists():
                # 删除模型目录
                shutil.rmtree(model_path)
                logger.info(f"模型已删除: {model_id}")
                return True
            else:
                logger.warning(f"模型不存在: {model_id}")
                return False
        except Exception as e:
            logger.error(f"删除模型失败: {str(e)}")
            return False

    def get_download_progress(self, model_id: str) -> tuple[float, str]:
        """
        获取下载进度

        Args:
            model_id: 模型 ID

        Returns:
            tuple: (进度百分比, 状态描述)
        """
        progress = self._download_progress.get(model_id, 0.0)
        status = self._download_status.get(model_id, "not_started")
        return progress, status

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """获取模型路径"""
        model_path = self.models_dir / model_id
        if model_path.exists() and self._is_model_complete(model_path):
            return model_path
        return None

    def calculate_model_size(self, model_id: str) -> str:
        """计算模型实际大小"""
        model_path = self.models_dir / model_id
        if not model_path.exists():
            return "未知"

        total_size = sum(
            f.stat().st_size
            for f in model_path.rglob('*')
            if f.is_file()
        )

        size_mb = total_size / 1024 / 1024
        if size_mb < 1024:
            return f"{size_mb:.1f}MB"
        else:
            return f"{size_mb / 1024:.2f}GB"

    def get_recommended_model(self) -> str:
        """获取推荐模型 ID"""
        # 根据配置获取推荐的模型
        if self.config_manager:
            device = self.config_manager.get("model.device", "cuda:0")

            # CPU 设备推荐 0.6B 模型
            if device == "cpu":
                return "0.6b-base"

        # 默认推荐 1.7B 模型
        return "1.7b-base"

    def check_model_usable(self, model_id: str) -> tuple[bool, str]:
        """
        检查模型是否可用

        Args:
            model_id: 模型 ID

        Returns:
            tuple: (是否可用, 原因描述)
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False, "模型不存在"

        # 检查依赖
        for dep_id in model_info.dependencies:
            dep_path = self.models_dir / dep_id
            if not dep_path.exists() or not self._is_model_complete(dep_path):
                dep_info = self.get_model_info(dep_id)
                return False, f"缺少依赖: {dep_info.name if dep_info else dep_id}"

        # 检查主模型
        model_path = self.models_dir / model_id
        if not model_path.exists():
            return False, "模型未安装"

        if not self._is_model_complete(model_path):
            return False, "模型文件不完整"

        return True, "模型可用"

    def list_usable_models(self) -> list:
        """列出所有可用的模型（排除分词器）"""
        usable = []
        for model_id in self.AVAILABLE_MODELS.keys():
            # 排除分词器模型
            if "tokenizer" in model_id.lower():
                continue
            is_usable, _ = self.check_model_usable(model_id)
            if is_usable:
                usable.append(model_id)
        return usable

    def list_usable_models_by_type(self, model_type: str) -> list:
        """
        列出指定类型的可用模型

        Args:
            model_type: 模型类型 ("customvoice", "voicedesign", "base")

        Returns:
            list: 可用的模型 ID 列表
        """
        usable = []
        for model_id in self.AVAILABLE_MODELS.keys():
            # 排除分词器模型
            if "tokenizer" in model_id.lower():
                continue

            # 检查模型类型是否匹配
            if model_type.lower() not in model_id.lower():
                continue

            is_usable, _ = self.check_model_usable(model_id)
            if is_usable:
                usable.append(model_id)
        return usable
