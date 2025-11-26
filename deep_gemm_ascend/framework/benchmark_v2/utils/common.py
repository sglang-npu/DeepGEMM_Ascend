"""
通用工具函数
"""


def ceil_div(dividend: int, divisor: int) -> int:
    """
    向上取整除法

    Args:
        dividend: 被除数
        divisor: 除数（必须为正数）
    """
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (dividend + divisor - 1) // divisor


def round_up(value: int, alignment: int) -> int:
    """
    将数值向上对齐到指定倍数

    Args:
        value: 要对齐的数值
        alignment: 对齐倍数（必须为正数）
    """
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if value == 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment

