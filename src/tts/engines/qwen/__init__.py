"""
Qwen3-TTS 引擎实现

基于 Qwen3-TTS 的 TTS 引擎，支持 Custom Voice / Voice Design / Voice Clone 三种模式
"""

from tts.engine_registry import ModelDefinition
from tts.engines.qwen.engine import QwenEngine

QWEN_MODEL_DEFINITIONS = [
    ModelDefinition(
        model_id="tokenizer-12hz",
        name="Qwen3-TTS 分词器 (12Hz)",
        engine_id="qwen",
        size="~50MB",
        repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        description="12Hz 采样率分词器",
    ),
    ModelDefinition(
        model_id="1.7b-customvoice",
        name="Qwen3-TTS 1.7B 自定义声音",
        engine_id="qwen",
        size="~3.4GB",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        description="1.7B 参数模型，支持自定义声音克隆",
        dependencies=["tokenizer-12hz"],
    ),
    ModelDefinition(
        model_id="1.7b-voicedesign",
        name="Qwen3-TTS 1.7B 声音设计",
        engine_id="qwen",
        size="~3.4GB",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        description="1.7B 参数模型，高级声音设计功能",
        dependencies=["tokenizer-12hz"],
    ),
    ModelDefinition(
        model_id="1.7b-base",
        name="Qwen3-TTS 1.7B 基础版",
        engine_id="qwen",
        size="~3.4GB",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        description="1.7B 参数基础模型",
        dependencies=["tokenizer-12hz"],
    ),
    ModelDefinition(
        model_id="0.6b-customvoice",
        name="Qwen3-TTS 0.6B 自定义声音",
        engine_id="qwen",
        size="~1.2GB",
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        description="0.6B 轻量级自定义声音",
        dependencies=["tokenizer-12hz"],
    ),
    ModelDefinition(
        model_id="0.6b-base",
        name="Qwen3-TTS 0.6B 基础版",
        engine_id="qwen",
        size="~1.2GB",
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        description="0.6B 基础模型",
        dependencies=["tokenizer-12hz"],
    ),
]

try:
    from tts.engine_registry import EngineRegistry

    EngineRegistry.instance().register("qwen", QwenEngine, QWEN_MODEL_DEFINITIONS)
except Exception as _e:
    import logging

    logging.getLogger(__name__).warning(f"Qwen engine registration failed: {_e}")
