# PhantomVox

> 基于 Qwen3-TTS 的本地文本转语音应用

PhantomVox 是一个使用 Flet 框架构建的跨平台桌面应用，提供高质量的文本转语音功能。

## 特性

- 🎙️ **高质量 TTS** - 基于 Qwen3-TTS 模型，提供自然的语音合成
- 🎨 **现代化界面** - Material Design 3 风格的用户界面
- ⚙️ **参数调整** - 支持语速、音调、声音类型等参数调节
- 💾 **音频导出** - 支持将生成的语音保存为 WAV 文件
- 🔊 **实时播放** - 内置音频播放器，可直接播放生成的语音
- 📝 **日志显示** - 实时显示操作日志和调试信息

## 技术栈

- **UI 框架**: [Flet](https://flet.dev/) 0.80.5 - 基于 Flutter 的 Python UI 框架
- **TTS 引擎**: [qwen-tts](https://github.com/QwenLM/Qwen3-TTS) - Qwen3-TTS 文本转语音
- **模型下载**: [modelscope](https://modelscope.cn/) - 模型下载和管理
- **音频处理**: soundfile, sounddevice, numpy

## 系统要求

- Python 3.8+ (推荐 Python 3.12)
- Windows / macOS / Linux
- 至少 4GB RAM
- 约 2GB 磁盘空间（用于模型文件）

## 快速开始

### 1. 环境配置

运行环境配置脚本：

```batch
setup_env.bat
```

该脚本会自动：
- 创建 Python 虚拟环境
- 安装 PyTorch (CPU 版本)
- 安装所有依赖包
- 安装 qwen-tts 和 modelscope

### 2. 激活虚拟环境

```batch
venv\Scripts\activate.bat
```

### 3. 运行程序

```batch
python src\main.py
```

## 手动安装

如果你想手动配置环境：

```batch
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate.bat

# 激活虚拟环境 (Linux/macOS)
source venv/bin/activate

# 升级 pip
python -m pip install --upgrade pip

# 安装 PyTorch (CPU 版本)
pip install torch>=2.0.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# 安装依赖
pip install -r requirements.txt

# 安装 modelscope 和 qwen-tts
pip install -U modelscope
pip install qwen-tts
```

## 使用说明

1. **输入文本** - 在文本输入框中输入要转换为语音的文本
2. **选择声音** - 从下拉菜单中选择声音类型（默认、女声、男声）
3. **调整参数** - 使用滑块调整语速和音调
4. **生成语音** - 点击"播放"按钮生成并播放语音
5. **保存音频** - 点击"保存音频"按钮将生成的语音保存为 WAV 文件

## 注意事项

### 首次运行

- 首次运行时，qwen-tts 会自动下载模型文件（约 1.7GB）
- 模型会缓存到 `~/.cache/huggingface/hub/` 目录
- 请确保网络连接正常，下载过程可能需要较长时间

### 性能优化

- 使用 CPU 版本的 PyTorch，适合本地运行
- 如果有 NVIDIA 显卡，可以安装 GPU 版本的 PyTorch 以提高性能

### 模型缓存

模型文件下载后会被缓存，无需重复下载。如果需要清除缓存：

```batch
# Windows
del /s /q %USERPROFILE%\.cache\huggingface\hub

# Linux/macOS
rm -rf ~/.cache/huggingface/hub
```

## 项目结构

```
PhantomVox/
├── src/
│   ├── ui/
│   │   └── main_ui.py          # 主 UI 控制器
│   ├── tts/
│   │   ├── qwen_engine.py      # TTS 引擎封装
│   │   └── audio_manager.py    # 音频管理器
│   ├── core/
│   │   ├── terminal.py         # 异步终端组件
│   │   └── event.py            # UI 事件系统
│   ├── config/
│   │   └── config_manager.py   # 配置管理器
│   ├── utils/
│   │   └── logger.py           # 日志工具
│   └── main.py                 # 应用入口
├── requirements.txt            # 依赖列表
├── setup_env.bat               # 环境配置脚本
└── README.md                   # 项目说明
```

## 参考资料

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Flet 官方文档](https://flet.dev/docs/)

## 许可证

GNU General Public License v3.0

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- [Qwen Team](https://qwen.ai/) - 提供 Qwen3-TTS 模型
- [Flet Team](https://flet.dev/) - 提供优秀的 Python UI 框架
- [SillyTavernLauncher](https://github.com/yourusername/SillyTavernLauncher) - 提供 UI 框架参考
