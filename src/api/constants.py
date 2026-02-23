"""API 常量定义"""

VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Serena",
    "fable": "Uncle_Fu",
    "onyx": "Dylan",
    "nova": "Eric",
    "shimmer": "Ono_Anna",
}

ALLOWED_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Ryan",
    "Aiden",
    "Sohee",
]

DEFAULT_SPEAKER = "Vivian"

SUPPORTED_AUDIO_FORMATS = ["wav", "pcm"]

PLANNED_AUDIO_FORMATS = ["mp3", "opus", "aac", "flac"]

# Timeout settings (in seconds)
DEFAULT_TTS_TIMEOUT = 300  # 5 minutes for normal synthesis
STREAMING_TIMEOUT = 600  # 10 minutes for streaming synthesis
MAX_TEXT_LENGTH = 10000  # Maximum text length for synthesis


# ==================== 音频常量 ====================

# 采样率 (Qwen-TTS 标准)
SAMPLE_RATE = 24000

# 10ms @ 24kHz = 240 samples (用于流式音频重叠)
OVERLAP_SAMPLES = 240

# 流式生成参数
DEFAULT_EMIT_EVERY_FRAMES = 8
DEFAULT_DECODE_WINDOW_FRAMES = 80
FIRST_CHUNK_EMIT_EVERY = 5
FIRST_CHUNK_DECODE_WINDOW = 48
FIRST_CHUNK_FRAMES = 48

# PCM 量化
PCM_INT16_MAX = 32767

# 音频格式
AUDIO_FORMAT_PCM = "pcm"
AUDIO_FORMAT_WAV = "wav"