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
