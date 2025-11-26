"""
数据模型模块

定义基准测试中使用的数据类
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CatlassResult:
    """Catlass 测试结果数据类"""
    idx: int
    M: int
    N: int
    K: int
    time: float
    diff: float
    kernel_func_name: str = ""
    parameters: Dict[str, int] = field(default_factory=lambda: {
        'mTile': 0,
        'nTile': 0,
        'kTile': 0
    })
    pipe_utilization: Dict[str, Any] = field(default_factory=dict)  # 存储PipeUtilization.xlsx中的数据

    @classmethod
    def from_dict(cls, data: dict) -> 'CatlassResult':
        """从字典创建 CatlassResult 对象"""
        return cls(
            idx=data['idx'],
            M=data['M'],
            N=data['N'],
            K=data['K'],
            time=data['time'],
            diff=data['diff'],
            kernel_func_name=data.get('kernel_func_name', ''),
            parameters=data.get('parameters', {}),
            pipe_utilization=data.get('pipe_utilization', {})
        )

