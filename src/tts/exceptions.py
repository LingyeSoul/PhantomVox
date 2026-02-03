"""
TTS引擎自定义异常

提供TTS引擎相关的异常类定义
"""


class TTSError(Exception):
    """TTS异常基类"""
    pass


class TTSModelNotLoadedError(TTSError):
    """模型未加载异常"""
    pass


class TTSTimeoutError(TTSError):
    """合成超时异常"""
    pass


class TTSInvalidParameterError(TTSError):
    """参数无效异常"""
    pass


class TTSSynthesisError(TTSError):
    """合成失败异常"""
    pass
