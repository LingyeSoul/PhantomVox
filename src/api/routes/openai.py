"""
OpenAI 兼容路由

提供与 OpenAI TTS API 兼容的接口
"""

from fastapi import APIRouter, HTTPException, Depends, Header, status
from fastapi.responses import StreamingResponse
from scipy.io import wavfile
import numpy as np
import io
import logging
from typing import Optional

from api.models import OpenAITTSRequest
from api.dependencies import get_tts_engine, log_message
from api.constants import VOICE_MAPPING, ALLOWED_SPEAKERS, DEFAULT_SPEAKER

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.post("/audio/speech")
async def openai_tts(
    request: OpenAITTSRequest,
    authorization: Optional[str] = Header(None),
    engine=Depends(get_tts_engine),
):
    """
    OpenAI 兼容的 TTS 端点

    此端点兼容 OpenAI TTS API 格式，可以使用标准 OpenAI SDK 调用。

    **映射规则**：
    - model "tts-1" 或 "tts-1-hd" -> 使用 Custom Voice 模式
    - voice 参数映射到内部说话人：
      - alloy -> Vivian
      - echo -> Serena
      - fable -> Uncle_Fu
      - onyx -> Dylan
      - nova -> Eric
      - shimmer -> Ono_Anna
    - speed 映射到 speed_factor
    - response_format 目前支持 wav（其他格式会自动转为 wav）

    **使用 curl 调用示例**：
    ```bash
    curl http://localhost:8848/v1/audio/speech \\
      -H "Authorization: Bearer sk-dummy" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "tts-1",
        "input": "你好，世界！",
        "voice": "alloy"
      }' \\
      --output speech.mp3
    ```

    **使用 Python OpenAI SDK 调用示例**：
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8848/v1",
        api_key="sk-dummy"
    )

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="你好，世界！"
    )

    response.stream_to_file("speech.mp3")
    ```
    """
    try:
        # 可选：验证 API 密钥格式
        # if authorization and not authorization.startswith("Bearer sk-"):
        #     raise HTTPException(
        #         status_code=status.HTTP_401_UNAUTHORIZED,
        #         detail="Invalid authorization header format"
        #     )

        # 映射 OpenAI voice 到内部说话人
        speaker = VOICE_MAPPING.get(request.voice, "Vivian")

        # 记录请求
        log_message(
            f"OpenAI TTS Request: model={request.model}, voice={request.voice} -> {speaker}, "
            f"text='{request.input[:50]}...'",
            "info",
        )

        # 调用内部 TTS 引擎
        audio_data, sample_rate = await engine.custom_voice_synthesize_async(
            text=request.input,
            speaker=speaker,
            language="Chinese",  # OpenAI API 默认根据文本自动检测
        )

        # 根据格式返回音频
        # 支持 wav 和 pcm 格式
        if request.response_format == "pcm":
            # 原始 PCM 数据，无文件头
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            media_type = "audio/l16;rate=24000;channels=1"
        else:
            # WAV 格式（默认）
            with io.BytesIO() as audio_buffer:
                wavfile.write(audio_buffer, sample_rate, audio_data)
                audio_bytes = audio_buffer.getvalue()

            # 确定 media_type
            media_type_map = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
            }
            media_type = media_type_map.get(request.response_format, "audio/wav")

        # 记录成功
        log_message(
            f"OpenAI TTS Success: {len(audio_data)} samples, {sample_rate}Hz, format={request.response_format}",
            "info",
        )

        # 返回音频流
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        log_message(f"OpenAI TTS Error: {str(e)}", "error")
        logger.exception("Unexpected error in OpenAI TTS synthesis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
