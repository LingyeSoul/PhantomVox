"""
流式 TTS 实时播放测试脚本

使用 sounddevice 边接收边播放音频的真正流式播放功能
"""

import requests
import time
import argparse
import sys
import threading
import queue
import numpy as np


class SoundDevicePlayer:
    """使用 sounddevice 进行实时播放"""

    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_playing = False
        self.audio_queue = queue.Queue(maxsize=100)  # 限制队列大小防止内存堆积
        self.buffer = np.array([], dtype=np.int16)  # 音频缓冲区
        self.stream = None
        self.lock = threading.Lock()

        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            raise ImportError("sounddevice 未安装。请运行: pip install sounddevice")

    def _audio_callback(self, outdata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f"Stream status: {status}")

        with self.lock:
            # 从队列获取新数据并添加到缓冲区
            while not self.audio_queue.empty():
                try:
                    audio_data = self.audio_queue.get_nowait()
                    new_samples = np.frombuffer(audio_data, dtype=np.int16)
                    self.buffer = np.concatenate([self.buffer, new_samples])
                except queue.Empty:
                    break

            # 从缓冲区取出所需帧数
            if len(self.buffer) >= frames:
                # 有足够的音频数据
                samples = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            else:
                # 数据不足，用剩余数据+静音填充
                samples = self.buffer
                silence = np.zeros(frames - len(samples), dtype=np.int16)
                samples = np.concatenate([samples, silence])
                self.buffer = np.array([], dtype=np.int16)

        # 转换 int16 到 float32 并归一化到 [-1, 1]
        audio_float = samples.astype(np.float32) / 32768.0
        outdata[:] = audio_float.reshape(-1, self.channels)

    def start(self):
        """启动播放流"""
        self.is_playing = True
        self.stream = self.sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=2048  # 每次回调请求 2048 帧
        )
        self.stream.start()

    def feed(self, audio_data):
        """喂入音频数据"""
        if self.is_playing:
            self.audio_queue.put(audio_data)

    def stop(self):
        """停止播放"""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # 清空队列和缓冲区
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        with self.lock:
            self.buffer = np.array([], dtype=np.int16)


def stream_and_play(
    url: str,
    text: str,
    api_type: str = "openai",
    mode: str = "custom_voice",
    voice: str = "alloy",
    speaker: str = "Vivian",
    sample_rate: int = 24000,
    pre_buffer_size: int = 48000,  # 预缓冲 2 秒 (24000 * 2)
    **kwargs
):
    """
    流式接收并播放音频

    Args:
        url: 服务器地址
        text: 要转换的文本
        api_type: API 类型 (openai/phantomvox)
        mode: TTS 模式
        voice: OpenAI voice
        speaker: 说话人
        sample_rate: 采样率
        pre_buffer_size: 预缓冲大小（字节）
        **kwargs: 其他参数
    """
    # 构建端点和请求
    if api_type == "openai":
        endpoint = f"{url}/v1/audio/speech/streaming"
        payload = {
            "model": kwargs.get("model", "tts-1"),
            "input": text,
            "voice": voice,
            "speed": kwargs.get("speed", 1.0)
        }
    else:
        endpoint = f"{url}/tts/streaming"
        payload = {
            "text": text,
            "mode": mode,
            "language": kwargs.get("language", "Chinese")
        }

        if mode == "custom_voice":
            payload["speaker"] = speaker
            if kwargs.get("instruct"):
                payload["instruct"] = kwargs["instruct"]
        elif mode == "voice_design":
            payload["design_prompt"] = kwargs.get("design_prompt", "")
        elif mode == "voice_clone":
            if kwargs.get("clone_id"):
                payload["clone_id"] = kwargs["clone_id"]
            if kwargs.get("clone_name"):
                payload["clone_name"] = kwargs["clone_name"]

    print(f"\n{'='*60}")
    print(f"流式 TTS 实时播放测试")
    print(f"{'='*60}")
    print(f"API: {api_type}")
    print(f"端点: {endpoint}")
    print(f"文本: {text}")
    if api_type == "phantomvox":
        print(f"模式: {mode}")
        if mode == "custom_voice":
            print(f"说话人: {speaker}")
    else:
        print(f"Voice: {voice}")
    print(f"预缓冲: {pre_buffer_size/sample_rate:.2f} 秒")
    print(f"{'='*60}\n")

    try:
        # 创建播放器
        player = SoundDevicePlayer(sample_rate=sample_rate)
        player.start()

        # 开始计时
        start_time = time.time()
        first_chunk_time = None
        first_play_time = None
        total_bytes = 0
        buffered_bytes = 0
        started_playing = False

        print("正在连接服务器...")

        with requests.post(endpoint, json=payload, stream=True, timeout=300) as response:
            if response.status_code != 200:
                print(f"错误: HTTP {response.status_code}")
                print(f"响应: {response.text}")
                return False

            print("✓ 连接成功，开始接收音频...")

            # 跳过 WAV 头（44字节）
            wav_header_skipped = False
            wav_header_remainder = b''  # 用于处理跨块的 WAV 头

            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue

                # 记录首块时间
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb = first_chunk_time - start_time
                    print(f"✓ 首块延迟 (TTFB): {ttfb*1000:.2f} ms")

                # 跳过 WAV 头（正确处理跨块情况）
                if not wav_header_skipped:
                    # 组合剩余数据和当前块
                    combined = wav_header_remainder + chunk
                    if len(combined) >= 44:
                        chunk = combined[44:]
                        wav_header_skipped = True
                        wav_header_remainder = b''
                    else:
                        # 还不够 44 字节，保存并等待下一个块
                        wav_header_remainder = combined
                        continue

                total_bytes += len(chunk)
                buffered_bytes += len(chunk)

                # 预缓冲
                if not started_playing:
                    if buffered_bytes >= pre_buffer_size:
                        first_play_time = time.time()
                        buffer_time = first_play_time - first_chunk_time
                        print(f"✓ 预缓冲完成: {buffer_time*1000:.2f} ms")
                        print(f"✓ 开始播放...")
                        started_playing = True

                # 喂入播放器
                player.feed(chunk)

                # 显示进度
                if started_playing and total_bytes % (sample_rate * 2) == 0:  # 每秒显示一次
                    elapsed = time.time() - first_play_time
                    played_seconds = total_bytes / 2 / sample_rate
                    print(f"  播放进度: {played_seconds:.1f}秒 (已接收 {total_bytes/1024:.1f} KB)")

            # 播放完成
            print(f"\n✓ 接收完成: {total_bytes/1024:.2f} KB")
            print("✓ 等待播放完成...")

            # 等待队列中的音频播放完成
            audio_duration = (total_bytes - 44) / 2 / sample_rate  # 音频时长（秒）
            time.sleep(audio_duration + 0.5)  # 额外加 0.5 秒缓冲

            player.stop()

            end_time = time.time()
            total_time = end_time - start_time

            if first_play_time:
                start_to_play = first_play_time - start_time
                print(f"\n{'='*60}")
                print(f"性能统计:")
                print(f"  请求到播放: {start_to_play*1000:.2f} ms")
                print(f"  总耗时: {total_time:.2f} 秒")
                print(f"  音频时长: {(total_bytes-44)/2/sample_rate:.2f} 秒")
                print(f"{'='*60}\n")

        return True

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n用户中断")
        player.stop()
        return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="流式 TTS 实时播放测试（使用 sounddevice）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 最简单的测试
  python test_streaming_play.py --text "你好，这是一个测试！"

  # 调整预缓冲大小
  python test_streaming_play.py --text "你好！" --pre-buffer 12000  # 0.5秒预缓冲
  python test_streaming_play.py --text "你好！" --pre-buffer 96000  # 4秒预缓冲

  # PhantomVox API
  python test_streaming_play.py --api phantomvox --speaker Vivian --text "你好！"

  # Voice Design 模式
  python test_streaming_play.py --api phantomvox --mode voice_design --design-prompt "温柔女声" --text "你好！"

  # 使用不同的 OpenAI voice
  python test_streaming_play.py --voice fable --text "今天天气真好！"
  python test_streaming_play.py --voice nova --text "这是测试语音！"
        """
    )

    parser.add_argument("--url", default="http://localhost:13650", help="服务器地址")
    parser.add_argument("--api", choices=["openai", "phantomvox"], default="openai",
                       help="API 类型（默认: openai）")
    parser.add_argument("--pre-buffer", type=int, default=48000,
                       help="预缓冲大小，字节（默认: 48000 = 2秒）")

    # OpenAI 参数
    parser.add_argument("--model", default="tts-1", help="OpenAI 模型")
    parser.add_argument("--voice", default="alloy",
                       choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                       help="OpenAI voice（默认: alloy）")
    parser.add_argument("--speed", type=float, default=1.0, help="语速（0.25-4.0）")

    # PhantomVox 参数
    parser.add_argument("--mode", choices=["custom_voice", "voice_design", "voice_clone"],
                       default="custom_voice", help="TTS 模式（默认: custom_voice）")
    parser.add_argument("--speaker", default="Vivian", help="说话人（默认: Vivian）")
    parser.add_argument("--language", default="Chinese", help="语言（默认: Chinese）")
    parser.add_argument("--instruct", default="", help="情感指令（custom_voice）")
    parser.add_argument("--design-prompt", default="", help="声音设计提示（voice_design）")
    parser.add_argument("--clone-id", default="", help="克隆音色 ID（voice_clone）")
    parser.add_argument("--clone-name", default="", help="克隆音色名称（voice_clone）")

    # 通用参数
    parser.add_argument("--text", default="今天的天气确实不错，适合户外活动呢！", help="要转换的文本")

    args = parser.parse_args()

    success = stream_and_play(
        url=args.url,
        text=args.text,
        api_type=args.api,
        mode=args.mode,
        voice=args.voice,
        speaker=args.speaker,
        model=args.model,
        speed=args.speed,
        language=args.language,
        instruct=args.instruct,
        design_prompt=args.design_prompt,
        clone_id=args.clone_id,
        clone_name=args.clone_name,
        pre_buffer_size=args.pre_buffer
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
