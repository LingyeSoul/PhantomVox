"""
API 安全模块

提供 API 密钥验证和认证依赖
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import secrets

# Bearer token 安全方案 (auto_error=False 允许无认证请求通过)
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """
    验证 API 密钥

    从配置中读取 security.api_key，如果未配置则允许所有请求通过。
    如果配置了 API 密钥，则要求请求携带有效的 Bearer token。

    Args:
        credentials: HTTP Bearer 认证凭据

    Returns:
        bool: 验证通过返回 True

    Raises:
        HTTPException: 401 如果 API 密钥无效或缺失
    """
    from config.config_manager import config_manager

    # 从配置获取 API 密钥
    api_key = config_manager.get("security.api_key", "")

    # 如果未配置 API 密钥，允许所有请求通过
    if not api_key:
        return True

    # 检查是否提供了认证凭据
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证 API 密钥 (使用 secrets.compare_digest 防止时序攻击)
    if not secrets.compare_digest(credentials.credentials, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


# 可选的 API 密钥验证（不强制要求，但会验证如果提供）
async def optional_api_key_verification(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """
    可选的 API 密钥验证

    如果配置了 API 密钥且请求提供了认证头，则验证。
    如果没有提供认证头，请求仍被允许通过。

    这种模式适用于需要向后兼容的场景。
    """
    from config.config_manager import config_manager

    api_key = config_manager.get("security.api_key", "")

    # 未配置 API 密钥或未提供凭据，允许通过
    if not api_key or not credentials:
        return True

    # 提供了凭据但无效 (使用 secrets.compare_digest 防止时序攻击)
    if not secrets.compare_digest(credentials.credentials, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
