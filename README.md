# PhantomVox

> 基于 Qwen3-TTS 的本地文本转语音应用

PhantomVox 是一个使用 Flet 框架构建的跨平台桌面应用，提供高质量的文本转语音功能。本项目有整合包集成了完整的 Python 环境，开箱即用。

## 特性

- 🎙️ **高质量 TTS** - 基于 Qwen3-TTS 模型，提供自然的语音合成
- 🎨 **现代化界面** - Material Design 3 风格的用户界面
- ⚙️ **参数调整** - 支持语速、音调、声音类型等参数调节
- 💾 **音频导出** - 支持将生成的语音保存为 WAV 文件
- 🔊 **实时播放** - 内置音频播放器，可直接播放生成的语音
- 🌐 **HTTP API** - 提供 OpenAI 兼容的 TTS API 接口
- 🚀 **GPU 加速** - 支持 CUDA 加速，提供更快的推理速度
- 📊 **流式输出** - 支持流式音频生成
- 📝 **日志显示** - 实时显示操作日志和调试信息

## 技术栈

- **UI 框架**: [Flet](https://flet.dev/) 0.80.5 - 基于 Flutter 的 Python UI 框架
- **TTS 引擎**: [qwen-tts](https://github.com/QwenLM/Qwen3-TTS) - Qwen3-TTS 文本转语音（本地修改版）
- **深度学习框架**: PyTorch 2.9.1 + CUDA 12.8
- **模型下载**: [modelscope](https://modelscope.cn/) - 模型下载和管理
- **音频处理**: soundfile, sounddevice, numpy
- **API 服务**: FastAPI + Uvicorn

## 系统要求

### 硬件要求

- **GPU**: NVIDIA 显卡（推荐，需支持 CUDA 12.8）
  - 显存建议 5GB 以上

### 软件要求

- **操作系统**: Windows 10/11 (64-bit)
- **显卡驱动**: NVIDIA 驱动版本 >= 527.41（支持 CUDA 12.8）

## 快速开始

### 1. 环境配置

运行环境配置脚本之前，请确保已安装 Python 3.12
带_embed后缀的是给整合包使用的脚本

安装SOX：https://sourceforge.net/projects/sox/ 自行下载安装到系统

运行环境配置脚本：

```batch
setup_env.bat
```

该脚本会自动完成以下操作：

1. **升级 pip** - 更新到最新版本
2. **安装 PyTorch** - 安装 PyTorch 2.9.1 + CUDA 12.8 版本
3. **安装依赖** - 安装所有必需的 Python 包
4. **安装本地 qwen-tts** - 安装项目集成的 qwen_tts-0.0.6
5. **安装 flash-attention** - 安装性能优化组件

安装过程可能需要 10-20 分钟，具体时间取决于网络速度。

### 2. 运行程序

环境配置完成后，直接运行启动脚本：

```batch
start.bat
```

### 3. 命令行环境

如需使用命令行工具（如 pip、python 等）：

```batch
cmd.bat
```

在命令行环境中，你可以：

```batch
# 查看版本
python --version

# 查看 Python 包
pip list

# 安装新包
pip install [package]

# 运行程序
python src\main.py
```

## 使用说明

### GUI 模式

1. **输入文本** - 在文本输入框中输入要转换为语音的文本
2. **选择声音** - 从下拉菜单中选择声音类型
3. **调整参数** - 使用滑块调整语速和音调
4. **生成语音** - 点击"播放"按钮生成并播放语音
5. **保存音频** - 点击"保存音频"按钮将生成的语音保存为 WAV 文件

### API 模式

项目同时提供 HTTP API 服务，支持 OpenAI 兼容的 TTS 接口。

**启动 API 服务器**：

```batch
python src\api\main.py
```

**API 端点**：

- `POST /v1/audio/speech` - OpenAI 兼容的 TTS 接口
- `GET /health` - 健康检查
- `GET /status` - 服务状态

详细 API 文档请访问：`http://localhost:8000/docs`

## 首次运行

### 模型下载

- 首次运行时，需要自行在模型管理页面下载模型
- 模型会缓存到用户目录下的 `models` 目录
- 请确保网络连接正常，下载过程可能需要较长时间


## 性能优化

### GPU 加速

本项目已预配置 CUDA 12.8 支持，如果你的系统有 NVIDIA 显卡：

1. 确保安装了最新的 NVIDIA 驱动（版本 >= 527.41）
2. 运行程序时会自动使用 GPU 加速
3. 你可以在日志中看到 "CUDA available" 的提示

### Flash Attention

Flash Attention 是一个性能优化组件，可以显著提高推理速度并减少内存占用。

- 本项目已集成预编译的 Flash Attention wheel
- 自动配置，无需手动设置

### CPU 模式

如果你没有 NVIDIA 显卡，程序会自动使用 CPU 模式运行。请注意：

- CPU 模式下生成速度较慢
- 建议降低音频质量和采样率以提高性能

## 项目结构

```
PhantomVox/
├── python/                      # 嵌入式 Python 环境
│   ├── python.exe               # Python 解释器
│   ├── Scripts/                 # pip 等工具
│   └── Lib/                     # Python 标准库
│
├── env/                         # 外部工具
│   └── sox.exe                  # 音频处理工具
│
├── src/                         # 源代码
│   ├── main.py                  # 应用入口
│   ├── version.py               # 版本信息
│   │
│   ├── ui/                      # 用户界面
│   │   └── main_ui.py           # 主 UI 控制器
│   │
│   ├── tts/                     # TTS 引擎
│   │   ├── qwen_engine.py       # Qwen TTS 引擎封装
│   │   └── audio_manager.py     # 音频管理器
│   │
│   ├── api/                     # HTTP API
│   │   ├── routes/              # API 路由
│   │   │   ├── openai.py        # OpenAI 兼容接口
│   │   │   ├── tts.py           # TTS 端点
│   │   │   └── tts_stream.py    # 流式 TTS
│   │   └── models.py            # API 数据模型
│   │
│   ├── core/                    # 核心组件
│   │   ├── model_manager.py     # 模型管理器
│   │   ├── tts_server.py        # TTS 服务器
│   │   ├── terminal.py          # 异步终端组件
│   │   ├── event.py             # UI 事件系统
│   │   └── network.py           # 网络工具
│   │
│   ├── config/                  # 配置管理
│   │   └── config_manager.py    # 配置管理器
│   │
│   └── utils/                   # 工具函数
│       └── logger.py            # 日志工具
│
├── qwen_tts-0.0.6-py3-none-any.whl  # 本地 qwen-tts 包
├── requirements.txt             # 依赖列表
├── setup_env.bat                # 环境配置脚本
├── start.bat                    # 启动脚本
├── cmd.bat                      # 命令行环境脚本
└── README.md                    # 项目说明
```

## 本地化说明

本项目使用本地修改版的 `qwen-tts` (v0.0.6)，而非官方版本。
源自 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 项目，已合并这两个仓库的修改：
[https://github.com/rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
[https://github.com/dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)

### 主要改进

- 优化了音频生成流程
- 改进了流式输出支持
- 增强了稳定性

## 日志位置

程序运行日志会显示在 UI 的日志面板中，有助于诊断问题。

## 许可证

GNU General Public License v3.0

## 致谢

- [Qwen Team](https://qwen.ai/) - 提供 Qwen3-TTS 模型
- [rekuenkdr 提供修改版 qwen-tts](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
- [dffdeeq 提供修改版 qwen-tts](https://github.com/dffdeeq/Qwen3-TTS-streaming)

