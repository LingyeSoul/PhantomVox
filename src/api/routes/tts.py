"""
TTS 合成路由

支持三种模式：Custom Voice, Voice Design, Voice Clone
"""

from fastapi import APIRouter, HTTPException, Depends, status
from scipy.io import wavfile
import numpy as np
import base64
import io
import logging
from typing import Optional

from api.models import TTSRequest, TTSResponse
from api.dependencies import get_tts_engine, get_voice_library, log_message
from api.routes.status import get_stats

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    engine=Depends(get_tts_engine),
    voice_library=Depends(get_voice_library),
    stats=Depends(get_stats)
):
    """
    文本转语音合成

    支持三种模式：
    - **custom_voice**: 使用预设说话人 + 情感指令
    - **voice_design**: 通过自然语言描述设计声音
    - **voice_clone**: 使用保存的克隆音色

    **Custom Voice 模式示例**：
    ```json
    {
        "text": "你好，世界！",
        "mode": "custom_voice",
        "speaker": "Vivian",
        "language": "Chinese",
        "instruct": "温柔的声音",
        "speed_factor": 1.0,
        "pitch_factor": 1.0
    }
    ```

    **Voice Design 模式示例**：
    ```json
    {
        "text": "你好，世界！",
        "mode": "voice_design",
        "design_prompt": "温柔细腻的女性声音，音调柔和，语速舒缓"
    }
    ```

    **Voice Clone 模式示例**：
    ```json
    {
        "text": "你好，世界！",
        "mode": "voice_clone",
        "clone_id": "clone_1234567890"
    }
    ```
    或者使用 clone_name：
    ```json
    {
        "text": "你好，世界！",
        "mode": "voice_clone",
        "clone_name": "我的克隆音色"
    }
    ```
    """
    try:
        # 记录请求
        log_message(
            f"TTS Request: mode={request.mode}, text='{request.text[:50]}...'",
            'info'
        )

        # 根据模式调用相应的合成方法
        if request.mode == "custom_voice":
            audio_data, sample_rate = await engine.custom_voice_synthesize_async(
                text=request.text,
                speaker=request.speaker,
                language=request.language,
                instruct=request.instruct,
                speed_factor=request.speed_factor,
                pitch_factor=request.pitch_factor
            )

        elif request.mode == "voice_design":
            audio_data, sample_rate = await engine.voice_design_synthesize_async(
                text=request.text,
                design_prompt=request.design_prompt,
                language=request.language,
                speed_factor=request.speed_factor,
                pitch_factor=request.pitch_factor
            )

        elif request.mode == "voice_clone":
            # 从 VoiceLibrary 查找克隆音色
            if voice_library is None:
                stats.record_request(success=False)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Voice library not available"
                )

            clone = None

            # 优先按名称查找
            if request.clone_name:
                for c in voice_library.get_all_clones():
                    if c["name"] == request.clone_name:
                        clone = c
                        break

                # 如果按名称找到多个或未找到，且提供了 clone_id，则使用 clone_id
                if (clone is None or
                    len([c for c in voice_library.get_all_clones()
                         if c["name"] == request.clone_name]) > 1):
                    if request.clone_id:
                        clone = voice_library.get_clone(request.clone_id)

            elif request.clone_id:
                clone = voice_library.get_clone(request.clone_id)

            if not clone:
                stats.record_request(success=False)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"克隆音色未找到：clone_id={request.clone_id}, clone_name={request.clone_name}"
                )

            # 使用克隆音色进行合成
            audio_data, sample_rate = await engine.voice_clone_synthesize_async(
                text=request.text,
                ref_audio=clone["ref_audio"],
                ref_text=clone["ref_text"]
            )

        else:
            stats.record_request(success=False)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid mode: {request.mode}"
            )

        # 转换音频为 WAV 格式
        audio_buffer = io.BytesIO()
        wavfile.write(audio_buffer, sample_rate, audio_data)
        audio_bytes = audio_buffer.getvalue()

        # 编码为 base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # 记录成功
        stats.record_request(success=True)
        log_message(
            f"TTS Success: {len(audio_data)} samples, {sample_rate}Hz",
            'info'
        )

        return TTSResponse(
            audio=audio_base64,
            format="wav",
            sample_rate=sample_rate,
            duration=len(audio_data) / sample_rate
        )

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        stats.record_request(success=False)
        log_message(f"TTS Error: {str(e)}", 'error')
        logger.exception("Unexpected error in TTS synthesis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
