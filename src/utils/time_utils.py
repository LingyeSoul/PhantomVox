"""
时间格式化工具函数

提供时间相关的格式化功能
"""


def format_elapsed_time(elapsed_seconds: float) -> str:
    """
    格式化用时显示

    将秒数转换为易读的时间格式：
    - 60秒以上: "X分Y.Z秒"
    - 60秒以下: "X.YZ秒"

    Args:
        elapsed_seconds: 经过的秒数

    Returns:
        格式化的时间字符串
    """
    if elapsed_seconds >= 60:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        return f"{minutes}分{seconds:.1f}秒"
    else:
        return f"{elapsed_seconds:.2f}秒"
