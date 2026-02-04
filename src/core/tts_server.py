"""
TTS HTTP 服务器模块

提供基于 HTTP 的 TTS 服务 API，支持局域网访问
"""

import json
import logging
import asyncio
import threading
import base64
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import io

from tts.qwen_engine import QwenEngine
from tts.audio_manager import AudioManager

logger = logging.getLogger(__name__)


class TTSRequest:
    """TTS 请求数据模型"""

    def __init__(self, data: dict):
        """初始化请求

        Args:
            data: 请求数据字典
        """
        self.text = data.get("text", "")
        self.mode = data.get("mode", "custom_voice")  # custom_voice, voice_design, voice_clone

        # Custom Voice 参数
        self.speaker = data.get("speaker", "Vivian")
        self.language = data.get("language", "Chinese")
        self.instruct = data.get("instruct", "")

        # Voice Design 参数
        self.design_prompt = data.get("design_prompt", "")

        # Voice Clone 参数
        self.ref_audio_base64 = data.get("ref_audio", "")
        self.ref_text = data.get("ref_text", "")
        self.x_vector_only = data.get("x_vector_only", False)

        # 其他参数
        self.speed_factor = data.get("speed_factor", 1.0)
        self.pitch_factor = data.get("pitch_factor", 1.0)

        # 输出格式
        self.output_format = data.get("output_format", "wav")  # wav, mp3, ogg


class TTSRequestHandler(BaseHTTPRequestHandler):
    """TTS HTTP 请求处理器"""

    # 类变量，用于存储共享资源
    server_instance: 'TTSServer' = None
    tts_engine_getter: Optional[Callable] = None
    log_callback: Optional[Callable] = None

    def _log(self, message: str, level: str = 'info'):
        """记录日志"""
        if self.log_callback:
            try:
                self.log_callback(message, level)
            except Exception:
                pass
        # 使用正确的日志级别方法
        level_upper = level.upper()
        if level_upper == 'INFO':
            logger.info(message)
        elif level_upper == 'ERROR':
            logger.error(message)
        elif level_upper == 'WARNING':
            logger.warning(message)
        elif level_upper == 'SUCCESS':
            logger.info(f"✓ {message}")
        else:
            logger.debug(message)

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        """设置响应头"""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json_response(self, data: dict, status_code: int = 200):
        """发送 JSON 响应"""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_error_response(self, message: str, status_code: int = 400):
        """发送错误响应"""
        self._send_json_response({
            "success": False,
            "error": message
        }, status_code)

    def _send_success_response(self, data: dict = None):
        """发送成功响应"""
        response = {"success": True}
        if data:
            response.update(data)
        self._send_json_response(response)

    def do_OPTIONS(self):
        """处理 OPTIONS 请求（CORS 预检）"""
        self._set_headers(200)

    def do_GET(self):
        """处理 GET 请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        client_ip = self.client_address[0]
        self._log(f"[GET] {path} from {client_ip}", 'info')

        if path == "/":
            self._handle_root()
        elif path == "/health":
            self._handle_health()
        elif path == "/status":
            self._handle_status()
        elif path == "/speakers":
            self._handle_speakers()
        elif path == "/languages":
            self._handle_languages()
        else:
            self._send_error_response("Not Found", 404)

    def do_POST(self):
        """处理 POST 请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        client_ip = self.client_address[0]
        self._log(f"[POST] {path} from {client_ip}", 'info')

        if path == "/tts":
            self._handle_tts()
        else:
            self._send_error_response("Not Found", 404)

    def _handle_root(self):
        """处理根路径请求"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PhantomVox TTS Service</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; border-radius: 4px; }
                .method { display: inline-block; background: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
                .path { font-family: monospace; font-weight: bold; color: #333; }
                .description { margin-top: 8px; color: #666; }
                code { background: #eee; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎙️ PhantomVox TTS Service</h1>
                <p>Welcome to PhantomVox TTS HTTP API Service!</p>

                <h2>API Endpoints:</h2>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/health</span>
                    <div class="description">Check if the service is running</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/status</span>
                    <div class="description">Get service status and statistics</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/speakers</span>
                    <div class="description">Get list of available speakers</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/languages</span>
                    <div class="description">Get list of supported languages</div>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/tts</span>
                    <div class="description">
                        Generate speech from text.<br>
                        Request body (JSON):<br>
                        <code>{ "text": "Hello world", "mode": "custom_voice", "speaker": "Vivian", "language": "Chinese" }</code>
                    </div>
                </div>

                <h2>Example cURL:</h2>
                <pre>curl -X POST http://localhost:PORT/tts \\
  -H "Content-Type: application/json" \\
  -d '{"text": "你好，世界！", "mode": "custom_voice", "speaker": "Vivian"}'</pre>
            </div>
        </body>
        </html>
        """
        self._set_headers(200, "text/html; charset=utf-8")
        self.wfile.write(html_content.encode('utf-8'))

    def _handle_health(self):
        """处理健康检查"""
        self._send_success_response({
            "status": "ok",
            "service": "PhantomVox TTS Service"
        })

    def _handle_status(self):
        """处理状态查询"""
        if self.server_instance:
            stats = self.server_instance.get_stats()
            self._send_success_response(stats)
        else:
            self._send_error_response("Server instance not available")

    def _handle_speakers(self):
        """处理说话人列表查询"""
        try:
            if self.tts_engine_getter:
                engine = self.tts_engine_getter()
                speakers = engine.get_supported_speakers()
                self._send_success_response({"speakers": speakers})
            else:
                self._send_error_response("TTS engine not available")
        except Exception as e:
            self._send_error_response(str(e), 500)

    def _handle_languages(self):
        """处理语言列表查询"""
        try:
            if self.tts_engine_getter:
                engine = self.tts_engine_getter()
                languages = engine.get_supported_languages()
                self._send_success_response({"languages": languages})
            else:
                self._send_error_response("TTS engine not available")
        except Exception as e:
            self._send_error_response(str(e), 500)

    def _handle_tts(self):
        """处理 TTS 合成请求"""
        try:
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error_response("Empty request body")
                return

            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            # 创建请求对象
            request = TTSRequest(data)

            # 验证必需参数
            if not request.text:
                self._send_error_response("Missing required parameter: text")
                return

            # 记录请求
            self._log(f"TTS Request: mode={request.mode}, text='{request.text[:50]}...'", 'info')

            # 获取 TTS 引擎
            if not self.tts_engine_getter:
                self._send_error_response("TTS engine not available", 500)
                return

            engine = self.tts_engine_getter()

            # 在线程池中执行 TTS 合成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                if request.mode == "custom_voice":
                    audio_data, sample_rate = loop.run_until_complete(
                        engine.custom_voice_synthesize_async(
                            text=request.text,
                            speaker=request.speaker,
                            language=request.language,
                            instruct=request.instruct,
                            speed_factor=request.speed_factor,
                            pitch_factor=request.pitch_factor
                        )
                    )
                elif request.mode == "voice_design":
                    if not request.design_prompt:
                        self._send_error_response("Missing required parameter: design_prompt")
                        return
                    audio_data, sample_rate = loop.run_until_complete(
                        engine.voice_design_synthesize_async(
                            text=request.text,
                            design_prompt=request.design_prompt,
                            language=request.language,
                            speed_factor=request.speed_factor,
                            pitch_factor=request.pitch_factor
                        )
                    )
                elif request.mode == "voice_clone":
                    if not request.ref_audio_base64:
                        self._send_error_response("Missing required parameter: ref_audio")
                        return
                    if not request.ref_text:
                        self._send_error_response("Missing required parameter: ref_text")
                        return

                    # 解码参考音频
                    ref_audio_data = base64.b64decode(request.ref_audio_base64)

                    # 保存临时音频文件
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(ref_audio_data)
                        tmp_path = tmp.name

                    try:
                        audio_data, sample_rate = loop.run_until_complete(
                            engine.voice_clone_synthesize_async(
                                text=request.text,
                                ref_audio=tmp_path,
                                ref_text=request.ref_text,
                                x_vector_only=request.x_vector_only
                            )
                        )
                    finally:
                        os.unlink(tmp_path)
                else:
                    self._send_error_response(f"Invalid mode: {request.mode}")
                    return

                # 转换音频为 WAV 格式
                import numpy as np
                from scipy.io import wavfile

                audio_buffer = io.BytesIO()
                wavfile.write(audio_buffer, sample_rate, audio_data)
                audio_bytes = audio_buffer.getvalue()

                # 返回音频数据（base64 编码）
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                # 记录成功
                if self.server_instance:
                    self.server_instance.record_request(success=True)

                self._send_success_response({
                    "audio": audio_base64,
                    "format": "wav",
                    "sample_rate": sample_rate,
                    "duration": len(audio_data) / sample_rate
                })

                self._log(f"TTS Success: {len(audio_data)} samples, {sample_rate}Hz", 'info')

            finally:
                loop.close()

        except json.JSONDecodeError:
            self._send_error_response("Invalid JSON in request body")
        except Exception as e:
            self._log(f"TTS Error: {str(e)}", 'error')
            if self.server_instance:
                self.server_instance.record_request(success=False)
            self._send_error_response(str(e), 500)

    def log_message(self, format, *args):
        """禁用默认的日志输出，使用自定义日志"""
        pass


class TTSServer:
    """TTS HTTP 服务器"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8848,
        tts_engine_getter: Optional[Callable] = None,
        log_callback: Optional[Callable] = None
    ):
        """初始化 TTS 服务器

        Args:
            host: 监听地址
            port: 监听端口
            tts_engine_getter: TTS 引擎获取函数
            log_callback: 日志回调函数
        """
        self.host = host
        self.port = port
        self.tts_engine_getter = tts_engine_getter
        self.log_callback = log_callback
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self._running = False

        # 请求统计
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._request_log = []  # 最近100条请求记录

        self._log(f"TTS Server initialized: {host}:{port}", 'info')

    def _log(self, message: str, level: str = 'info'):
        """记录日志"""
        if self.log_callback:
            try:
                self.log_callback(message, level)
            except Exception:
                pass
        # 使用正确的日志级别方法
        level_upper = level.upper()
        if level_upper == 'INFO':
            logger.info(message)
        elif level_upper == 'ERROR':
            logger.error(message)
        elif level_upper == 'WARNING':
            logger.warning(message)
        elif level_upper == 'SUCCESS':
            logger.info(f"✓ {message}")
        else:
            logger.debug(message)

    def start(self):
        """启动服务器（异步，不阻塞）"""
        if self._running:
            self._log("Server is already running", 'warning')
            return False

        # 在单独的线程中启动服务器
        self.server_thread = threading.Thread(
            target=self._start_server,
            daemon=True
        )
        self.server_thread.start()
        return True

    def _start_server(self):
        """在后台线程中启动服务器"""
        try:
            # 设置请求处理器的类变量
            TTSRequestHandler.server_instance = self
            TTSRequestHandler.tts_engine_getter = self.tts_engine_getter
            TTSRequestHandler.log_callback = self.log_callback

            # 创建服务器
            self._log(f"正在创建 HTTP 服务器，端口 {self.port}...", 'info')
            self.server = HTTPServer((self.host, self.port), TTSRequestHandler)
            self.server.allow_reuse_address = True

            # 标记为运行中（在 serve_forever 之前设置）
            self._running = True
            self._log(f"✓ TTS Server started on http://{self.host}:{self.port}", 'success')

            # 运行服务器（阻塞直到停止）
            self.server.serve_forever()

        except OSError as e:
            # 端口被占用等网络错误
            self._running = False
            self._log(f"✗ 端口 {self.port} 启动失败: {str(e)}", 'error')
        except Exception as e:
            self._running = False
            self._log(f"✗ 服务器启动异常: {type(e).__name__}: {str(e)}", 'error')

    def stop(self):
        """停止服务器"""
        if not self._running:
            self._log("Server is not running", 'warning')
            return False

        try:
            if self.server:
                self.server.shutdown()
                self.server.server_close()

            if self.server_thread:
                self.server_thread.join(timeout=5)

            self._running = False
            self._log("TTS Server stopped", 'info')
            return True

        except Exception as e:
            self._log(f"✗ Failed to stop server: {str(e)}", 'error')
            return False

    def is_running(self) -> bool:
        """检查服务器是否正在运行"""
        return self._running

    def get_url(self) -> str:
        """获取服务器 URL"""
        return f"http://{self.host}:{self.port}"

    def record_request(self, success: bool):
        """记录请求统计"""
        self._total_requests += 1
        if success:
            self._successful_requests += 1
            status = "✓"
        else:
            self._failed_requests += 1
            status = "✗"

        # 添加到请求日志
        self._request_log.append({
            "time": time.strftime("%H:%M:%S"),
            "status": status,
            "success": success
        })

        # 只保留最近100条
        if len(self._request_log) > 100:
            self._request_log.pop(0)

    def get_stats(self) -> dict:
        """获取服务器统计信息"""
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "recent_requests": self._request_log[-20:]  # 最近20条
        }

    def reset_stats(self):
        """重置统计信息"""
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._request_log = []
        self._log("Statistics reset", 'info')
