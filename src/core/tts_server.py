"""
FastAPI TTS 服务器实现

使用 FastAPI 框架重构的 TTS HTTP 服务器，提供与旧 TTSServer 类相同的接口
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from typing import Optional, Callable, Any
import logging
import threading
import asyncio

from api.dependencies import initialize_dependencies, cleanup_dependencies, log_message
from api.routes import health, status, metadata, tts, openai, tts_stream

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理

    使用现代 lifespan 上下文管理器处理启动/关闭事件
    """
    # 启动阶段
    logger.info("FastAPI TTS Server starting up...")

    # 启动任务引擎
    from core.task_engine import get_task_engine
    task_engine = get_task_engine()
    await task_engine.start()
    logger.info("Task engine started for API server")

    yield

    # 关闭阶段
    logger.info("FastAPI TTS Server shutting down...")

    # 停止任务引擎
    await task_engine.stop()
    logger.info("Task engine stopped")

    cleanup_dependencies()


def create_fastapi_app(
    tts_engine_getter: Callable,
    voice_library=None,
    log_callback: Optional[Callable] = None
) -> FastAPI:
    """
    创建并配置 FastAPI 应用

    Args:
        tts_engine_getter: TTS 引擎获取函数
        voice_library: VoiceLibrary 实例
        log_callback: 日志回调函数

    Returns:
        FastAPI: 配置好的 FastAPI 应用实例
    """
    # 初始化依赖
    initialize_dependencies(tts_engine_getter, voice_library, log_callback)

    # 创建 FastAPI 应用
    app = FastAPI(
        title="PhantomVox TTS Service",
        description="基于 Qwen3-TTS 的文本转语音 HTTP API 服务",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # 配置 CORS - 局域网环境默认允许所有来源
    cors_origins = ["*"]
    allow_credentials = False  # 通配符来源时必须禁用 credentials

    try:
        from config.config_manager import config_manager
        configured_origins = config_manager.get("security.cors_origins", None)
        if configured_origins:
            # 如果配置了具体来源，启用 credentials
            cors_origins = configured_origins
            allow_credentials = True
            logger.info(f"CORS configured with specific origins: {cors_origins}")
    except Exception as e:
        logger.warning(f"Failed to load CORS config: {e}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # 配置限流 - 从配置中读取限流设置
    try:
        from config.config_manager import config_manager
        rate_limit = config_manager.get("security.rate_limit_per_minute", 60)
        
        # 添加限流中间件
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        
        @app.exception_handler(limiter._default_rate_limit_exceeded_handler)
        async def rate_limit_handler(request, exc):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        
        logger.info(f"Rate limiting enabled: {rate_limit} requests per minute")
    except ImportError:
        logger.warning("slowapi not installed, rate limiting disabled")
    except Exception as e:
        logger.warning(f"Failed to configure rate limiting: {e}")

    # 注册路由
    app.include_router(health.router, tags=["Health"])
    app.include_router(status.router, tags=["Status"])
    app.include_router(metadata.router, tags=["Metadata"])
    app.include_router(tts.router, tags=["TTS"])
    app.include_router(openai.router, tags=["OpenAI"])
    app.include_router(tts_stream.router, tags=["TTS Stream"])

    # 根路径 HTML 页面
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """服务首页"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PhantomVox TTS Service - FastAPI</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; border-radius: 4px; }
                .method { display: inline-block; background: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
                .path { font-family: monospace; font-weight: bold; color: #333; }
                .description { margin-top: 8px; color: #666; }
                .api-docs { margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 4px; text-align: center; }
                .api-docs a { color: #1976d2; text-decoration: none; font-weight: bold; }
                .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 12px; margin-left: 8px; }
                .badge-openai { background: #10a37f; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎙️ PhantomVox TTS Service (FastAPI)</h1>
                <p>基于 FastAPI 的现代化 TTS HTTP API 服务</p>

                <div class="api-docs">
                    <p>📚 查看 <a href="/docs">Swagger API 文档</a> 或 <a href="/redoc">ReDoc 文档</a></p>
                </div>

                <h2>PhantomVox API 端点:</h2>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/health</span>
                    <div class="description">健康检查</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/status</span>
                    <div class="description">服务状态和统计信息</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/speakers</span>
                    <div class="description">获取可用说话人列表</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/languages</span>
                    <div class="description">获取支持的语言列表</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/clones</span>
                    <div class="description">获取保存的克隆音色列表</div>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span>
                    <span class="path">/design-presets</span>
                    <div class="description">获取语音设计预设列表</div>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/tts</span>
                    <div class="description">文本转语音合成（支持三种模式）</div>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/tts/streaming</span>
                    <div class="description">
                        <strong>真正的流式</strong>文本转语音合成（支持三种模式）
                        <br>边生成边解码边输出，首块延迟降低 50%+
                    </div>
                </div>

                <h2>OpenAI 兼容 API <span class="badge badge-openai">OpenAI</span>:</h2>

                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/v1/audio/speech</span>
                    <div class="description">
                        OpenAI TTS API 兼容端点
                        <br>可直接使用 OpenAI SDK 调用
                    </div>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span>
                    <span class="path">/v1/audio/speech/streaming</span>
                    <div class="description">
                        OpenAI TTS API 兼容<strong>真正的流式</strong>端点
                        <br>边生成边解码边输出，首块延迟降低 50%+
                    </div>
                </div>

                <h2>示例 cURL:</h2>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto;">
# PhantomVox API
curl -X POST http://localhost:PORT/tts \\
  -H "Content-Type: application/json" \\
  -d '{"text": "你好，世界！", "mode": "custom_voice", "speaker": "Vivian"}'

# PhantomVox 流式 API（真正的流式）
curl -N -X POST http://localhost:PORT/tts/streaming \\
  -H "Content-Type: application/json" \\
  -d '{"text": "你好，世界！", "mode": "custom_voice", "speaker": "Vivian"}' \\
  --output speech.wav

# OpenAI 兼容 API
curl http://localhost:PORT/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{"model": "tts-1", "input": "你好，世界！", "voice": "alloy"}' \\
  --output speech.mp3

# OpenAI 兼容流式 API（真正的流式）
curl -N http://localhost:PORT/v1/audio/speech/streaming \\
  -H "Content-Type: application/json" \\
  -d '{"model": "tts-1", "input": "你好，世界！", "voice": "alloy"}' \\
  --output speech.wav
                </pre>
            </div>
        </body>
        </html>
        """

    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理器"""
        logger.exception(f"Unhandled exception: {exc}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)}
        )

    return app


class FastAPITSServer:
    """
    FastAPI TTS 服务器

    提供与旧 TTSServer 类相同的接口，确保与 TTSServiceView 的兼容性
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8848,
        tts_engine_getter: Optional[Callable] = None,
        voice_library=None,
        log_callback: Optional[Callable] = None
    ):
        """初始化 FastAPI TTS 服务器

        Args:
            host: 监听地址
            port: 监听端口
            tts_engine_getter: TTS 引擎获取函数
            voice_library: VoiceLibrary 实例
            log_callback: 日志回调函数
        """
        self.host = host
        self.port = port
        self.tts_engine_getter = tts_engine_getter
        self.voice_library = voice_library
        self.log_callback = log_callback

        # 创建 FastAPI 应用
        self.app = create_fastapi_app(tts_engine_getter, voice_library, log_callback)

        # Uvicorn 服务器相关
        self._uvicorn_server: Optional[Any] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        self._log(f"FastAPI TTS Server initialized: {host}:{port}", 'info')

    def _log(self, message: str, level: str = 'info'):
        """记录日志"""
        log_message(message, level)

    def start(self):
        """启动服务器（异步，不阻塞）"""
        if self._running:
            self._log("Server is already running", 'warning')
            return False

        import uvicorn

        # 配置 Uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )

        # 创建服务器实例
        self._uvicorn_server = uvicorn.Server(config)

        # 在后台线程中运行
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self._server_thread.start()

        return True

    @property
    def server_thread(self):
        """公共属性：访问服务器线程（兼容旧接口）"""
        return self._server_thread

    def _run_server(self):
        """在后台线程中运行 Uvicorn 服务器"""
        try:
            self._running = True
            self._log(f"✓ FastAPI TTS Server started on http://{self.host}:{self.port}", 'success')

            # 运行服务器
            asyncio.run(self._uvicorn_server.serve())

        except Exception as e:
            self._running = False
            self._log(f"✗ Server error: {str(e)}", 'error')

    def stop(self):
        """停止服务器"""
        if not self._running:
            self._log("Server is not running", 'warning')
            return False

        try:
            if self._uvicorn_server:
                # 发送 shutdown 信号
                self._uvicorn_server.should_exit = True

            if self._server_thread:
                self._server_thread.join(timeout=5)

            self._running = False
            self._log("FastAPI TTS Server stopped", 'info')
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

    def get_stats(self) -> dict:
        """获取服务器统计信息"""
        from api.routes.status import _stats
        stats_data = _stats.get_stats()
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            **stats_data
        }

    def record_request(self, success: bool):
        """记录请求统计（兼容旧接口）"""
        from api.routes.status import _stats
        _stats.record_request(success)

    def reset_stats(self):
        """重置统计信息"""
        from api.routes.status import _stats
        _stats.reset()
        self._log("Statistics reset", 'info')
