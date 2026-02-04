"""
流式 TTS 合成路由

提供真正的流式音频返回的 TTS 端点（边生成边解码边输出）
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
import logging
import struct
import numpy as np
from typing import Optional, AsyncGenerator

from api.models import TTSRequest, OpenAITTSRequest
from api.dependencies import get_tts_engine, get_voice_library, log_message
from api.routes.status import get_stats

router = APIRouter()
logger = logging.getLogger(__name__)


# OpenAI voice 到内部说话人的映射
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Serena",
    "fable": "Uncle_Fu",
    "onyx": "Dylan",
    "nova": "Eric",
    "shimmer": "Ono_Anna",
}


# ============================================
# 辅助函数
# ============================================

def create_wav_header(sample_rate: int, num_samples: int) -> bytes:
    """创建 WAV 文件头

    注意：WAV 格式限制文件大小为 4GB - 8 字节
    如果 num_samples 太大，会自动限制在安全范围内
    """
    byte_rate = sample_rate * 2  # 16-bit mono

    # WAV 格式限制：最大文件大小 4GB - 8 字节 (0xFFFFFFF8)
    # 减去头部 36 字节，最大数据大小 = 0xFFFFFFF8 - 36 = 0xFFFFFFD2
    # 最大样本数 = 0xFFFFFFD2 / 2 = 0x7FFFFFE9
    max_safe_samples = 0x7FFFFFE9

    # 如果传入的样本数超过安全值，限制在最大值
    if num_samples > max_safe_samples:
        num_samples = max_safe_samples

    data_size = num_samples * 2

    # RIFF header
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')

    # fmt chunk (chunk ID + size + format data)
    header += struct.pack('<4sIHHIIHH',
                         b'fmt ',      # chunk ID
                         16,          # chunk size (for PCM)
                         1,           # audio format (1 = PCM)
                         1,           # num channels
                         sample_rate, # sample rate
                         byte_rate,   # byte rate
                         2,           # block align
                         16)          # bits per sample

    # data chunk
    header += struct.pack('<4sI', b'data', data_size)

    return header


async def stream_result_to_wav(
    result_generator: AsyncGenerator[dict, None],
    initial_header: bool = True
) -> AsyncGenerator[bytes, None]:
    """
    将流式生成结果转换为 WAV 音频块

    Args:
        result_generator: 流式生成结果生成器
        initial_header: 是否在第一个块前添加 WAV 头

    Yields:
        bytes: WAV 音频数据块
    """
    first_chunk = True
    header_sent = False
    total_samples = 0

    async for result in result_generator:
        result_type = result.get('type')

        if result_type == 'audio_chunk':
            audio = result['audio']
            sample_rate = result['sample_rate']

            # 转换为 bytes (16-bit PCM)
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # 第一个块：发送 WAV 头（注意：我们不知道总样本数，使用最大值）
            if initial_header and not header_sent:
                # 使用 WAV 格式允许的最大样本数（约 2GB 音频数据，超过 24 小时）
                # WAV 格式限制：文件大小 < 4GB
                max_samples = 0x7FFFFFE9
                wav_header = create_wav_header(sample_rate, max_samples)
                yield wav_header
                header_sent = True

            total_samples += len(audio)
            yield audio_bytes

        elif result_type == 'done':
            logger.info(f"流式生成完成，共 {total_samples} 样本")
            break

        elif result_type == 'error':
            error_msg = result.get('error', '未知错误')
            logger.error(f"流式生成错误: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS generation failed: {error_msg}"
            )


@router.post("/tts/streaming")
async def synthesize_speech_streaming(
    request: TTSRequest,
    engine=Depends(get_tts_engine),
    voice_library=Depends(get_voice_library),
    stats=Depends(get_stats)
):
    """
    真正的流式文本转语音合成（边生成边解码边输出）

    这是新的流式端点，使用 monkey patch 拦截底层生成过程，
    实现 Codec Codes 分块生成和实时解码，大幅降低首字延迟。

    **Custom Voice 模式示例**：
    ```json
    {
        "text": "你好，世界！",
        "mode": "custom_voice",
        "speaker": "Vivian",
        "language": "Chinese"
    }
    ```

    **特点**：
    - 低延迟：生成 32 个 tokens 后立即解码并输出
    - 内存友好：不需要等待完整音频生成
    - 实时反馈：通过进度信息了解生成状态

    **使用 curl 测试**：
    ```bash
    curl -N -X POST http://localhost:8848/tts/streaming \\
      -H "Content-Type: application/json" \\
      -d '{"text":"你好，世界！","mode":"custom_voice","speaker":"Vivian"}' \\
      --output speech.wav
    ```

    **使用 Python 测试**：
    ```python
    import requests

    response = requests.post(
        "http://localhost:8848/tts/streaming",
        json={"text": "你好", "mode": "custom_voice", "speaker": "Vivian"},
        stream=True
    )

    with open("speech.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
    ```
    """
    try:
        # 记录请求
        log_message(
            f"[REAL-STREAMING] TTS Request: mode={request.mode}, text='{request.text[:50]}...'",
            'info'
        )

        # 检查引擎是否支持流式输出
        if not engine.enable_streaming:
            stats.record_request(success=False)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Streaming is not enabled. Please restart the server with enable_streaming=True"
            )

        async def audio_stream_generator():
            """异步音频流生成器"""
            try:
                # 根据模式调用相应的流式合成方法
                if request.mode == "custom_voice":
                    result_gen = engine.custom_voice_synthesize_streaming_async(
                        text=request.text,
                        speaker=request.speaker,
                        language=request.language,
                        instruct=request.instruct,
                        speed_factor=request.speed_factor,
                        pitch_factor=request.pitch_factor
                    )

                elif request.mode == "voice_design":
                    result_gen = engine.voice_design_synthesize_streaming_async(
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

                    # 使用克隆音色进行流式合成
                    result_gen = engine.voice_clone_synthesize_streaming_async(
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

                # 转换为 WAV 流并 yield
                async for wav_chunk in stream_result_to_wav(result_gen):
                    yield wav_chunk

            except HTTPException:
                raise
            except Exception as e:
                log_message(f"Streaming TTS Error: {str(e)}", 'error')
                logger.exception("Unexpected error in streaming TTS synthesis")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {str(e)}"
                )

        # 记录成功
        stats.record_request(success=True)

        # 返回流式响应
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # 禁用nginx缓冲
                "X-Content-Type-Options": "nosniff"
            }
        )

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        stats.record_request(success=False)
        log_message(f"Streaming TTS Error: {str(e)}", 'error')
        logger.exception("Unexpected error in streaming TTS endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/v1/audio/speech/streaming")
async def openai_tts_streaming(
    request: OpenAITTSRequest,
    authorization: Optional[str] = None,
    engine=Depends(get_tts_engine)
):
    """
    OpenAI 兼容的真正流式 TTS 端点

    此端点兼容 OpenAI TTS API 格式，返回真正的流式音频（边生成边输出）。
    可以使用标准 OpenAI SDK 调用。

    **使用 curl 调用示例**：
    ```bash
    curl -N http://localhost:8848/v1/audio/speech/streaming \\
      -H "Authorization: Bearer sk-dummy" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "tts-1",
        "input": "你好，世界！",
        "voice": "aloy"
      }' \\
      --output speech.wav
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
        voice="aloy",
        input="你好，世界！"
    )

    # 流式保存到文件
    with open("speech.wav", "wb") as f:
        for chunk in response.iter_bytes(chunk_size=4096):
            f.write(chunk)
    ```
    """
    try:
        # 映射 OpenAI voice 到内部说话人
        speaker = VOICE_MAPPING.get(request.voice, "Vivian")

        # 记录请求
        log_message(
            f"[REAL-STREAMING] OpenAI TTS Request: model={request.model}, voice={request.voice} -> {speaker}, "
            f"text='{request.input[:50]}...'",
            'info'
        )

        # 检查引擎是否支持流式输出
        if not engine.enable_streaming:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Streaming is not enabled. Please restart the server with enable_streaming=True"
            )

        async def audio_stream_generator():
            """异步音频流生成器"""
            try:
                # 调用真正的流式合成方法
                result_gen = engine.custom_voice_synthesize_streaming_async(
                    text=request.input,
                    speaker=speaker,
                    language="Chinese",  # OpenAI API 默认根据文本自动检测
                    speed_factor=request.speed
                )

                # 转换为 WAV 流
                async for wav_chunk in stream_result_to_wav(result_gen):
                    yield wav_chunk

            except Exception as e:
                log_message(f"OpenAI Streaming TTS Error: {str(e)}", 'error')
                logger.exception("Unexpected error in OpenAI streaming TTS synthesis")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {str(e)}"
                )

        # 记录成功
        log_message(
            f"[REAL-STREAMING] OpenAI TTS Success: Starting stream",
            'info'
        )

        # 返回音频流
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="speech.wav"',
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log_message(f"OpenAI Streaming TTS Error: {str(e)}", 'error')
        logger.exception("Unexpected error in OpenAI streaming TTS endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

