"""
流式 TTS 合成路由

提供真正的流式音频返回的 TTS 端点（边生成边解码边输出）
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
import logging
import struct
import numpy as np
from typing import Optional, AsyncGenerator, Tuple

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
    result_generator: AsyncGenerator[Tuple[np.ndarray, int], None],
    initial_header: bool = True,
    response_format: str = "wav"
) -> AsyncGenerator[bytes, None]:
    """
    将流式生成结果转换为音频块

    Args:
        result_generator: 流式生成结果生成器 (audio, sample_rate) 元组
        initial_header: 是否在第一个块前添加 WAV 头（仅当 response_format="wav" 时）
        response_format: 音频格式 ("wav", "pcm", "mp3", "opus")

    Yields:
        bytes: 音频数据块
    """
    header_sent = False
    total_samples = 0

    async for audio, sample_rate in result_generator:
        # 转换为 bytes (16-bit PCM)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # WAV 格式：第一个块发送 WAV 头
        if response_format == "wav" and initial_header and not header_sent:
            # 使用 WAV 格式允许的最大样本数（约 2GB 音频数据，超过 24 小时）
            # WAV 格式限制：文件大小 < 4GB
            max_samples = 0x7FFFFFE9
            wav_header = create_wav_header(sample_rate, max_samples)
            yield wav_header
            header_sent = True

        # PCM/RAW 格式：直接发送音频数据，无需文件头
        # 这样客户端可以立即开始播放，无需等待文件头
        total_samples += len(audio)
        yield audio_bytes

    logger.info(f"流式生成完成，共 {total_samples} 样本")


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
                        instruct=request.instruct
                    )

                elif request.mode == "voice_design":
                    result_gen = engine.voice_design_synthesize_streaming_async(
                        text=request.text,
                        design_prompt=request.design_prompt,
                        language=request.language
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
                    # 优先使用预计算的特征（如果存在）
                    if "prompt_features" in clone and clone["prompt_features"]:
                        # 使用预计算特征（快速）
                        result_gen = engine.voice_clone_synthesize_streaming_async(
                            text=request.text,
                            voice_clone_prompt=clone["prompt_features"]
                        )
                    else:
                        # 降级：重新计算特征
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
        # 添加完整的禁缓冲响应头以确保客户端实时接收数据
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
                "Transfer-Encoding": "chunked",  # 明确标识分块传输
                "Connection": "keep-alive",  # 保持连接
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

    **支持的格式**：
    - wav: 标准WAV格式（带文件头）
    - pcm: 原始PCM数据（无文件头，推荐用于流式播放）
    - mp3: MP3格式（未来支持）
    - opus: Opus格式（未来支持）

    **使用 curl 调用示例**：
    ```bash
    curl -N http://localhost:8848/v1/audio/speech/streaming \\
      -H "Authorization: Bearer sk-dummy" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "tts-1",
        "input": "你好，世界！",
        "voice": "aloy",
        "response_format": "pcm"
      }' \\
      --output speech.pcm
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
        input="你好，世界！",
        response_format="pcm"
    )

    # 流式保存到文件
    with open("speech.pcm", "wb") as f:
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
            f"format={request.response_format}, text='{request.input[:50]}...'",
            'info'
        )

        # 检查引擎是否支持流式输出
        if not engine.enable_streaming:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Streaming is not enabled. Please restart the server with enable_streaming=True"
            )

        # 确定媒体类型
        media_type_map = {
            "wav": "audio/wav",
            "pcm": "audio/raw",  # 原始PCM，推荐用于流式
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac"
        }
        media_type = media_type_map.get(request.response_format, "audio/wav")

        # 添加采样率到媒体类型（对于raw PCM）
        if request.response_format == "pcm":
            media_type = "audio/l16;rate=24000;channels=1"

        async def audio_stream_generator():
            """异步音频流生成器"""
            try:
                # 调用真正的流式合成方法
                result_gen = engine.custom_voice_synthesize_streaming_async(
                    text=request.input,
                    speaker=speaker,
                    language="Chinese"  # OpenAI API 默认根据文本自动检测
                )

                # 转换为音频流（根据格式决定是否添加WAV头）
                # 对于 PCM/RAW 格式，不添加文件头，客户端可以立即播放
                use_header = (request.response_format == "wav")
                async for wav_chunk in stream_result_to_wav(result_gen, initial_header=use_header, response_format=request.response_format):
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
            f"[REAL-STREAMING] OpenAI TTS Success: Starting stream (format={request.response_format})",
            'info'
        )

        # 返回音频流
        # 添加完整的禁缓冲响应头以确保客户端实时接收数据
        return StreamingResponse(
            audio_stream_generator(),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"',
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
                "Transfer-Encoding": "chunked",  # 明确标识分块传输
                "Connection": "keep-alive",  # 保持连接
                "X-Content-Type-Options": "nosniff"
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

