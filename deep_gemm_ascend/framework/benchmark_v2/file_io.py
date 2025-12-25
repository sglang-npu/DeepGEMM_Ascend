"""
文件I/O模块

提供文件读写功能，包括shape加载、checkpoint管理和结果写入
"""

import os
import json
import random
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict

import pandas as pd

from .models import CatlassResult

# CommonMatmulKernel的shape约束常量
COMMON_MATMUL_MAX_N = 50000
COMMON_MATMUL_MAX_K = 50000


def filter_common_matmul_shapes(shapes: List[List[int]], operator_name: Optional[str] = None) -> List[List[int]]:
    """
    根据算子类型筛选shape，如果是CommonMatmulKernel则筛选N<=5W, K<=5W
    
    Args:
        shapes: 原始shape列表
        operator_name: 算子名称（可选）
        
    Returns:
        筛选后的shape列表
    """
    if not operator_name or operator_name.lower() != 'commonmatmulkernel':
        return shapes
    
    filtered_shapes = []
    filtered_count = 0
    for shape in shapes:
        m, n, k = shape[0], shape[1], shape[2]
        if n > COMMON_MATMUL_MAX_N or k > COMMON_MATMUL_MAX_K:
            filtered_count += 1
            continue
        filtered_shapes.append(shape)
    
    if filtered_count:
        print(f"[filter_common_matmul_shapes] 已剔除 CommonMatmulKernel 不满足约束 (N>{COMMON_MATMUL_MAX_N} 或 K>{COMMON_MATMUL_MAX_K}) 的 shape 数量: {filtered_count}")
    
    return filtered_shapes


def generate_qwen3_shapes() -> List[List[int]]:
    """
    生成Qwen3相关的shape列表
    
    生成规则：
    - shape模板：[M, 4096, 4096], [M, 6144, 4096], [M, 24576, 4096], [M, 4096, 12288]
    - M从极小值、小值、中等值、大值、超大值各取一个
    - 共生成4个模板 × 5个M值 = 20个shape
    
    Returns:
        Qwen3 shape列表，共20个shape
    """
    # 定义5个M值级别
    m_values = {
        '极小值': 8,
        '小值': 30,
        '中等值': 800,
        '大值': 3000,
        '超大值': 10000
    }
    
    # 定义4个shape模板
    shape_templates = [
        [None, 4096, 4096],   # [M, 4096, 4096]
        [None, 6144, 4096],   # [M, 6144, 4096]
        [None, 24576, 4096],  # [M, 24576, 4096]
        [None, 4096, 12288],  # [M, 4096, 12288]
    ]
    
    qwen3_shapes = []
    for m_label, m_value in m_values.items():
        for template in shape_templates:
            shape = [m_value, template[1], template[2]]
            qwen3_shapes.append(shape)
    
    return qwen3_shapes


def prepare_shapes_with_qwen3(
    shapes: List[List[int]],
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    add_qwen3_shapes: bool = False
) -> List[List[int]]:
    """
    对shapes进行打乱、插入qwen3_shapes和切片处理
    
    Args:
        shapes: 原始shape列表
        start_idx: 切片起始位置（可选，从0开始，包含该位置）
        end_idx: 切片结束位置（可选，不包含该位置，None表示到末尾）
        add_qwen3_shapes: 是否添加qwen3_shapes到最前面，默认False
        
    Returns:
        处理后的shape列表（打乱、插入qwen3_shapes、切片后）
    """
    # 先随机打乱shape列表（使用固定随机种子，确保所有进程使用相同的打乱顺序）
    shuffled_shapes = shapes.copy()
    random.seed(42)  # 使用固定种子，确保所有进程得到相同的打乱顺序
    random.shuffle(shuffled_shapes)
    print(f"[prepare_shapes_with_qwen3] 已随机打乱shape列表（随机种子=42），打乱后数量: {len(shuffled_shapes)}")
    
    # 根据参数决定是否生成Qwen3 shapes并插入到最前面
    if add_qwen3_shapes:
        qwen3_shapes = generate_qwen3_shapes()
        shapes = qwen3_shapes + shuffled_shapes
        print(f"[prepare_shapes_with_qwen3] 已添加 {len(qwen3_shapes)} 个Qwen3 shape到最前面，总数量: {len(shapes)}")
    else:
        shapes = shuffled_shapes
        print(f"[prepare_shapes_with_qwen3] 未添加Qwen3 shape，总数量: {len(shapes)}")
    
    # 对打乱并插入qwen3_shapes后的shapes进行切片（如果提供了start_idx或end_idx）
    # start_idx和end_idx是全局索引，指的是在整个列表（包含qwen3_shapes）中的位置
    total_shapes = len(shapes)
    if start_idx is not None or end_idx is not None:
        # 检查并设置start_idx
        if start_idx is None:
            start_idx = 0
        elif start_idx < 0:
            raise ValueError(f"start_idx ({start_idx}) 必须 >= 0")
        elif start_idx >= total_shapes:
            raise ValueError(f"start_idx ({start_idx}) 超出范围 [0, {total_shapes-1}]")
        
        # 检查并设置end_idx
        if end_idx is None:
            end_idx = total_shapes
        elif end_idx < 0:
            raise ValueError(f"end_idx ({end_idx}) 必须 >= 0")
        elif end_idx > total_shapes:
            raise ValueError(f"end_idx ({end_idx}) 超出范围 [0, {total_shapes}]")
        
        # 检查start_idx < end_idx
        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) 必须 < end_idx ({end_idx})")
        
        # 执行切片
        shapes = shapes[start_idx:end_idx]
        print(f"[prepare_shapes_with_qwen3] 切片: [{start_idx}:{end_idx}], 总数量: {total_shapes}, 切片后数量: {len(shapes)}")
    
    return shapes


# 默认shape组（如果未提供shapes.xlsx文件则使用）
default_shape_group = [
    # M、N、K
    # [8, 4096, 7168], # 1h 15min   1240
    # [8, 7168, 18432], # 1h 15min   1240
    # [8, 18432, 7168], # 1h 15min   1240
    # [64, 4096, 7168], # 5h  5906
    # [64, 7168, 18432], # 5h  5906
    # [64, 18432, 7168], # 5h  5906
    # [64, 24576, 1536], # 5h  5906
    # [64, 32768, 512], # 5h  5906
    # [64, 7168, 16384], # 5h  5906
    # [128, 4096, 7168], # 9h   9660
    # [128, 7168, 18432], # 9h   9660
    # [128, 18432, 7168], # 9h   9660
    # [1024, 4096, 7168], # 14h   14520
    # [1024, 18432, 7168], # 14h   14520
    # [2048, 4096, 7168], # 14h   14520
    [1279, 5003, 7681],
    # [3511, 6151, 8191],
    # [5119, 6997, 9901]
]


def load_shapes_from_excel(
    shapes_file: str,
    operator_name: Optional[str] = None,
    layout_tag_a: Optional[int] = None,
    layout_tag_b: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> List[List[int]]:
    """
    从shapes.xlsx文件中读取并筛选shape数据
    
    Args:
        shapes_file: shapes.xlsx文件路径
        operator_name: 算子名称筛选条件（可选）
        layout_tag_a: LayoutTagA筛选条件（可选）
        layout_tag_b: LayoutTagB筛选条件（可选）
        start_idx: 切片起始位置（可选，从0开始，包含该位置）
        end_idx: 切片结束位置（可选，不包含该位置，None表示到末尾）
        
    Returns:
        筛选后的shape列表，每个shape为[M, N, K]格式
        
    筛选条件（可选）：
        - Op Name == operator_name（默认SmallMatmulKernel）
        - LayoutTagA == layout_tag_a（默认0）
        - LayoutTagB == layout_tag_b（默认0）
        
    切片说明：
        - 在筛选完成后，对筛选后的shapes列表进行切片
        - start_idx: 起始索引，必须 >= 0
        - end_idx: 结束索引，必须 <= len(shapes)，且 > start_idx
        - 如果 start_idx 或 end_idx 为 None，则不进行切片
    """
    if not os.path.exists(shapes_file):
        print(f"Warning: shapes file not found: {shapes_file}")
        print("Using default shape_group instead")
        return default_shape_group
    
    try:
        # 读取Excel文件，第一行作为表头
        df = pd.read_excel(shapes_file, engine='openpyxl', header=0)
        
        print(f"[load_shapes_from_excel] 读取Excel文件: 形状={df.shape}, 列名={list(df.columns)}")
        
        if df.empty:
            print(f"[load_shapes_from_excel] 警告: Excel文件为空")
            return default_shape_group
        
        # 检查必需的列是否存在
        required_columns = ['Op Name', 'LayoutTagA', 'LayoutTagB', 'M', 'N', 'K']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[load_shapes_from_excel] 错误: 缺少必需的列: {missing_columns}")
            print(f"[load_shapes_from_excel] 可用列: {list(df.columns)}")
            return default_shape_group
        
        filtered_df = df
        if operator_name is None:
            operator_name = "SmallMatmulKernel"
        if operator_name:
            filtered_df = filtered_df[filtered_df['Op Name'] == operator_name]
        if layout_tag_a is None:
            layout_tag_a = 0
        if layout_tag_a is not None:
            filtered_df = filtered_df[filtered_df['LayoutTagA'] == layout_tag_a]
        if layout_tag_b is None:
            layout_tag_b = 0
        if layout_tag_b is not None:
            filtered_df = filtered_df[filtered_df['LayoutTagB'] == layout_tag_b]
        
        if filtered_df.empty:
            print(f"[load_shapes_from_excel] 警告: 筛选后没有符合条件的行")
            return default_shape_group
        
        # 提取M, N, K列，转换为整数列表
        shapes = []
        filtered_n1 = 0
        for _, row in filtered_df.iterrows():
            try:
                m = int(row['M'])
                n = int(row['N'])
                k = int(row['K'])
                if n == 1:
                    filtered_n1 += 1
                    continue
                shapes.append([m, n, k])
            except (ValueError, TypeError):
                # 静默跳过无效行
                continue
        
        if filtered_n1:
            print(f"[load_shapes_from_excel] 已额外剔除 N == 1 的 shape 数量: {filtered_n1}")
        
        # CommonMatmulKernel的shape约束筛选
        shapes = filter_common_matmul_shapes(shapes, operator_name)
        
        # 判断是否为CommonMatmulKernel算子，只有Common算子才添加qwen3_shapes
        is_common_matmul = operator_name and operator_name.lower() == 'commonmatmulkernel'
        
        # 使用统一的处理函数：打乱、插入qwen3_shapes、切片
        shapes = prepare_shapes_with_qwen3(shapes, start_idx, end_idx, add_qwen3_shapes=is_common_matmul)
        
        print(f"[load_shapes_from_excel] 成功提取 {len(shapes)} 个shape")
        if len(shapes) > 0:
            print(f"[load_shapes_from_excel] 前5个shape示例: {shapes[:5]}")
        
        return shapes if shapes else default_shape_group
        
    except Exception as e:
        print(f"[load_shapes_from_excel] 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return default_shape_group


class CheckpointManager:
    """Checkpoint管理器，负责断点文件的读写"""
    
    def __init__(self, checkpoint_path: str):
        """
        初始化CheckpointManager
        
        Args:
            checkpoint_path: checkpoint文件路径
        """
        self.checkpoint_path = Path(checkpoint_path)
    
    def load(self) -> Optional[int]:
        """
        加载checkpoint，返回最后处理的索引
        
        Returns:
            最后处理的索引，如果文件不存在或读取失败返回None
        """
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                # 优化：直接读取第一行（紧凑JSON格式）
                line = f.readline().strip()
                if line:
                    checkpoint = json.loads(line)
                    return checkpoint.get('last_process_idx', -1)
        except (json.JSONDecodeError, IOError):
            # 兼容旧格式
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                    return checkpoint.get('last_process_idx', -1)
            except Exception:
                return None
        
        return None
    
    def save(self, last_process_idx: int) -> None:
        """
        保存checkpoint
        
        Args:
            last_process_idx: 最后处理的索引
        """
        try:
            # 优化：使用紧凑JSON格式，减少I/O开销
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                f.write(f'{{"last_process_idx":{last_process_idx}}}\n')
        except Exception as e:
            print(f"[CheckpointManager] 保存checkpoint失败: {e}")


class ResultWriter:
    """结果写入器，负责批量写入测试结果"""
    
    def __init__(self, result_path: str, buffer_size: int = 10):
        """
        初始化ResultWriter
        
        Args:
            result_path: 结果文件路径
            buffer_size: 缓冲区大小，默认10
        """
        self.result_path = Path(result_path)
        self.buffer_size = buffer_size
        self.result_buffer: List[CatlassResult] = []
    
    def add_result(self, result: CatlassResult) -> None:
        """
        添加结果到缓冲区，如果缓冲区满则自动写入
        
        Args:
            result: 测试结果
        """
        self.result_buffer.append(result)
        if len(self.result_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self) -> None:
        """批量写入结果到文件，减少I/O开销"""
        if not self.result_buffer:
            return
        
        try:
            # 优化：批量写入，减少文件打开/关闭次数
            # 使用更大的缓冲区（8KB）减少系统调用
            with open(self.result_path, 'a', encoding='utf-8', buffering=8192) as f:
                for result in self.result_buffer:
                    # 优化：使用紧凑JSON格式，减少文件大小和I/O时间
                    json.dump(asdict(result), f, ensure_ascii=False, separators=(',', ':'))
                    f.write('\n')
            self.result_buffer.clear()
        except IOError as e:
            print(f"[ResultWriter] 保存文件错误: {e}")
        except Exception as e:
            print(f"[ResultWriter] 处理数据错误: {e}")

