"""
Benchmark模块

提供向后兼容的导入接口
"""

# 导出主要类和函数，保持向后兼容
from .catlass_parameter import CatlassParameter
from .models import CatlassResult
from .distributed_benchmark_runner import GEMMBenchmarkRunner
from .file_io import load_shapes_from_excel, default_shape_group
from .padding_calculator import PaddingTag, PaddingCalculator, calc_padding_tags, estimate_bandwidth

__all__ = [
    'CatlassParameter',
    'CatlassResult',
    'GEMMBenchmarkRunner',
    'load_shapes_from_excel',
    'default_shape_group',
    'PaddingTag',
    'PaddingCalculator',
    'calc_padding_tags',
    'estimate_bandwidth',
]

