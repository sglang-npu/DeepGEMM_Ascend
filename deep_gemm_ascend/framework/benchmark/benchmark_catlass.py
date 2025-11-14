from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import math
from dataclasses import dataclass, field, asdict
import subprocess
import re
import os
import json
import argparse
from tqdm import tqdm
import glob
import shutil
import time
import pandas as pd

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

def load_shapes_from_excel(shapes_file: str) -> List[List[int]]:
    """
    从shapes.xlsx文件中读取并筛选shape数据
    
    Args:
        shapes_file: shapes.xlsx文件路径
        
    Returns:
        筛选后的shape列表，每个shape为[M, N, K]格式
        
    筛选条件：
        - Op Name == "SmallMatmulKernel"
        - LayoutTagA == 0
        - LayoutTagB == 0
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
        
        # 筛选条件：
        # 1. Op Name == "SmallMatmulKernel"
        # 2. LayoutTagA == 0
        # 3. LayoutTagB == 0
        filtered_df = df[
            (df['Op Name'] == 'SmallMatmulKernel') &
            (df['LayoutTagA'] == 0) &
            (df['LayoutTagB'] == 0)
        ]
        
        if filtered_df.empty:
            print(f"[load_shapes_from_excel] 警告: 筛选后没有符合条件的行")
            return default_shape_group
        
        # 提取M, N, K列，转换为整数列表
        shapes = []
        for _, row in filtered_df.iterrows():
            try:
                m = int(row['M'])
                n = int(row['N'])
                k = int(row['K'])
                shapes.append([m, n, k])
            except (ValueError, TypeError):
                # 静默跳过无效行
                continue
        # TODO:测试用，从120开始
        shapes = shapes[120:]
        print(f"[load_shapes_from_excel] 成功提取 {len(shapes)} 个shape")
        if len(shapes) > 0:
            print(f"[load_shapes_from_excel] 前5个shape示例: {shapes[:5]}")
        
        return shapes if shapes else default_shape_group
        
    except Exception as e:
        print(f"[load_shapes_from_excel] 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return default_shape_group

class CatlassParameter():
    """
    为 catlass 生成 tiling 网格参数
    参数：mTile, nTile, kTile
    约束条件：
        - 各维度元素数量为 tile × 16，矩阵元素数量需考虑两次 16
        - L0C = (mTile×16) × (nTile×16) × 4 字节 ≤ 128KB
        - L1 = (mTile×16 + nTile×16) × kTile×16 × 2 字节 < 512KB
    """
    def __init__(self, operator_type=None, core_num=24):
        """
        Args:
            operator_type: 算子类型，可选值: 'smallmatmul', 'commonmatmul', 'paddingcommonmatmul', None(所有)
            core_num: AI Core数量，默认24
        """
        self.operator_type = operator_type
        self.core_num = core_num
        self.grid_parameters = self.grid_generate_parameters()

    def generate_mn_tile_values(self):
        """
        生成mTile和nTile的值（范围1-512）
        策略：多粒度采样，考虑L0C约束（mTile × nTile ≤ 128）
        优化：压缩采样点，确保总组合数在500以内
        
        推理过程：
        1. L0C约束：mTile × nTile ≤ 128，意味着大部分大值组合会被过滤
        2. 步长1范围：1-4（4个值），覆盖最关键的极小值
        3. 其他区域：适当增大步长，减少采样点
        4. 目标：mn_tile_values约15-20个值，k_tile_values约20-25个值
        5. 理论最大组合：20² × 25 = 400 × 25 = 10,000
        6. 实际有效组合（考虑L0C约束）：约300-500
        """
        values = []
        
        # 极小值区域（1-4）：步长1，最密集采样
        # 这个区域对性能影响最大，且L0C约束下最常用
        values.extend(range(1, 5, 1))  # 1,2,3,4 (4个值)
        
        # 小值区域（6-16）：步长2，密集采样
        # 包含关键值8, 12, 16
        values.extend(range(6, 17, 2))  # 6,8,10,12,14,16 (6个值)
        
        # 中小值区域（20-32）：步长4，中等密度
        # 包含关键值24, 28, 32
        values.extend(range(20, 33, 4))  # 20,24,28,32 (4个值)
        
        # 中值区域（40-64）：步长8，稀疏采样
        # 包含关键值48, 56, 64
        values.extend(range(40, 65, 8))  # 40,48,56,64 (4个值)
        
        # 中大值区域（80-128）：步长16，很稀疏采样
        # 包含关键值96, 112, 128
        values.extend(range(80, 129, 16))  # 80,96,112,128 (4个值)
        
        # 大值区域（160-256）：步长32，极稀疏采样
        # 包含关键值192, 224, 256
        values.extend(range(160, 257, 32))  # 160,192,224,256 (4个值)
        
        # 超大值区域（320-512）：步长64，最稀疏采样
        # 包含关键值384, 448, 512
        values.extend(range(320, 513, 64))  # 320,384,448,512 (4个值)
        
        # 去重并排序
        values = sorted(list(set(values)))
        return values
    
    def generate_k_tile_values(self):
        """
        生成kTile的值（范围1-1024）
        策略：多粒度采样，压缩采样点数量
        
        推理过程：
        1. kTile范围更大（1-1024），可以更稀疏采样
        2. 步长1范围：1-4（4个值），覆盖最关键的极小值
        3. 其他区域：适当增大步长
        4. 目标：约20-25个值
        """
        values = []
        
        # 极小值区域（1-4）：步长1，最密集采样
        # kTile对L1缓存影响大，极小值需要完整覆盖
        values.extend(range(1, 5, 1))  # 1,2,3,4 (4个值)
        
        # 小值区域（6-16）：步长2，密集采样
        # 包含关键值8, 12, 16
        values.extend(range(6, 17, 2))  # 6,8,10,12,14,16 (6个值)
        
        # 中小值区域（20-32）：步长4，中等密度
        # 包含关键值24, 28, 32
        values.extend(range(20, 33, 4))  # 20,24,28,32 (4个值)
        
        # 中值区域（40-64）：步长8，稀疏采样
        # 包含关键值48, 56, 64
        values.extend(range(40, 65, 8))  # 40,48,56,64 (4个值)
        
        # 中大值区域（80-128）：步长16，很稀疏采样
        # 包含关键值96, 112, 128
        values.extend(range(80, 129, 16))  # 80,96,112,128 (4个值)
        
        # 大值区域（160-256）：步长32，极稀疏采样
        # 包含关键值192, 224, 256
        values.extend(range(160, 257, 32))  # 160,192,224,256 (4个值)
        
        # 超大值区域（320-512）：步长64，最稀疏采样
        # 包含关键值384, 448, 512
        values.extend(range(320, 513, 64))  # 320,384,448,512 (4个值)
        
        # 极大值区域（640-1024）：步长128，极稀疏采样
        # 包含关键值768, 896, 1024
        values.extend(range(640, 1025, 128))  # 640,768,896,1024 (4个值)
        
        # 去重并排序
        values = sorted(list(set(values)))
        return values
    
    def generate_tile_values(self):
        """
        兼容性方法：返回mTile/nTile的值（保持向后兼容）
        新代码应该使用generate_mn_tile_values()和generate_k_tile_values()
        """
        return self.generate_mn_tile_values()

    def generate_mnk_tiles_linear(self):
        """
        线性生成 mTile, nTile, kTile 的所有组合
        约束条件：
        1. L0C = (mTile×16) × (nTile×16) × 4 字节 ≤ 128KB
        2. L1 = (mTile×16 + nTile×16) × kTile×16 × 2 字节 x double_buffer < 512KB
        3. mTile/nTile范围：1-512，kTile范围：1-1024
        """
        mn_tile_values = self.generate_mn_tile_values()
        k_tile_values = self.generate_k_tile_values()
        double_buffer = 2
        L1_CACHE_MAX_BYTES = 512 * 1024 // double_buffer  # 256KB
        L0C_MAX_BYTES = 128 * 1024  # 128KB
        
        # 预计算常量，避免重复计算
        L0C_FACTOR = 16 * 16 * 4  # 1024
        L0C_MAX_TILE_PRODUCT = L0C_MAX_BYTES // L0C_FACTOR  # 128
        
        for m_tile in mn_tile_values:
            for n_tile in mn_tile_values:
                # L0C bytes = (mTile*16) * (nTile*16) * 4 = mTile * nTile * 1024
                # 优化：直接比较 mTile * nTile，避免乘法运算
                if m_tile * n_tile > L0C_MAX_TILE_PRODUCT:
                    continue
                
                # 预计算 (mTile*16 + nTile*16)，避免在k循环中重复计算
                m_plus_n_16 = (m_tile + n_tile) * 16
                
                for k_tile in k_tile_values:
                    # L1 bytes = (mTile*16 + nTile*16) * kTile*16 * 2
                    # 优化：使用预计算的值
                    l1_bytes = m_plus_n_16 * k_tile * 32  # 32 = 16 * 2
                    if l1_bytes > L1_CACHE_MAX_BYTES:
                        continue
                    
                    yield m_tile, n_tile, k_tile

    def grid_generate_parameters(self):
        """
        生成所有有效的参数组合
        返回参数列表，每个元素包含 mTile, nTile, kTile
        """
        parameters = []
        mn_tile_values = self.generate_mn_tile_values()
        k_tile_values = self.generate_k_tile_values()
        
        for m_tile, n_tile, k_tile in self.generate_mnk_tiles_linear():
            param_dict = {
                'mTile': m_tile,
                'nTile': n_tile,
                'kTile': k_tile
            }
            parameters.append(param_dict)
        
        # 输出统计信息
        print(f'=' * 60)
        print(f'Tile Values Statistics:')
        print(f'  mTile/nTile values: {len(mn_tile_values)} (range: {min(mn_tile_values)}-{max(mn_tile_values)})')
        print(f'    Values: {mn_tile_values[:10]}...{mn_tile_values[-5:]}')
        print(f'  kTile values: {len(k_tile_values)} (range: {min(k_tile_values)}-{max(k_tile_values)})')
        print(f'    Values: {k_tile_values[:10]}...{k_tile_values[-5:]}')
        print(f'')
        print(f'Combination Analysis:')
        print(f'  Theoretical max (before constraints): {len(mn_tile_values)}² × {len(k_tile_values)} = {len(mn_tile_values)**2 * len(k_tile_values):,}')
        print(f'  Valid combinations (after L0C/L1 constraints): {len(parameters):,}')
        print(f'  Constraint filtering rate: {(1 - len(parameters) / (len(mn_tile_values)**2 * len(k_tile_values))) * 100:.1f}%')
        print(f'=' * 60)
        
        if len(parameters) > 500:
            print(f'⚠️  Warning: Total combinations ({len(parameters)}) exceeds target of 500!')
            print(f'   Consider further reducing tile values or adjusting sampling strategy.')
        elif len(parameters) < 100:
            print(f'⚠️  Warning: Total combinations ({len(parameters)}) is very low (< 100).')
            print(f'   Consider increasing tile values for better coverage.')
        else:
            print(f'✓ Total combinations ({len(parameters)}) is within reasonable range (100-500).')
        
        return parameters

    def check_smallmatmul_constraints(self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0):
        """
        检查shape和tiling参数是否满足SmallMatmul算子的约束条件
        
        SmallMatmul约束条件：
        1. Cache约束: (m1 + n1) × k1 × 2 ≤ 512KB, m1 × n1 × 4 ≤ 128KB
        2. Tile数量: ⌈m/m1⌉ × ⌈n/n1⌉ ≤ coreNum
        3. K轴: k ≤ k1
        4. Padding: paddingTagA/B/C == PADDING_NONE (简化检查)
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: Tiling参数（实际值，不是tile编号）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
        
        Returns:
            bool: 是否满足SmallMatmul约束
        """
        # 约束1: Cache约束
        # L1约束: (m1 + n1) × k1 × 2 ≤ 512KB = 524288字节
        l1_bytes = (m1 + n1) * k1 * 2
        if l1_bytes > 524288:
            return False
        
        # L0C约束: m1 × n1 × 4 ≤ 128KB = 131072字节
        l0c_bytes = m1 * n1 * 4
        if l0c_bytes > 131072:
            return False
        
        # 约束2: Tile数量约束
        # ⌈m/m1⌉ × ⌈n/n1⌉ ≤ coreNum
        task_blocks = math.ceil(m / m1) * math.ceil(n / n1)
        if task_blocks > self.core_num:
            return False
        
        # 约束3: K轴约束
        if k > k1:
            return False
        
        # 约束4: Padding约束（简化版本）
        # 这里只做基本的对齐检查，完整的padding检查需要模拟GetPaddingTag的完整逻辑
        # 简化策略：检查可能导致padding的常见情况
        
        # PaddingTagA检查（简化）
        # 如果layoutTagA == RowMajor: innerAxisA = k, outterAxisA = m
        # 如果layoutTagA == ColumnMajor: innerAxisA = m, outterAxisA = k
        if layout_tag_a == 0:  # RowMajor
            inner_axis_a = k
            outter_axis_a = m
        else:  # ColumnMajor
            inner_axis_a = m
            outter_axis_a = k
        
        # 如果innerAxisA < 8 或 (innerAxisA < 32 且 innerAxisA % 16 != 0) 且 outterAxisA > 512
        # 则paddingTagA = PADDING_NZ
        if ((inner_axis_a < 8 or (inner_axis_a < 32 and inner_axis_a % 16 != 0)) and 
            outter_axis_a > 512):
            return False
        
        # PaddingTagB检查（简化）
        if layout_tag_b == 0:  # RowMajor
            inner_axis_b = n
            outter_axis_b = k
        else:  # ColumnMajor
            inner_axis_b = k
            outter_axis_b = n
        
        if ((inner_axis_b < 8 or (inner_axis_b < 32 and inner_axis_b % 16 != 0)) and 
            outter_axis_b > 512):
            return False
        
        # PaddingTagC检查（简化）
        # paddingTagC = PADDING_ND 当: m×n > 2048² 且 n > 256 且 n % 128 != 0
        if (m * n > 2048 * 2048 and n > 256 and n % 128 != 0):
            # 还需要检查totalDataSize < 192MB，这里简化处理
            # 如果满足前面的条件，很可能需要padding
            return False
        
        return True
    
    def filter_parameters(self, shape):
        """
        根据shape和算子类型过滤参数
        
        Args:
            shape: [m, n, k] 矩阵维度
        
        Returns:
            过滤后的参数列表
        """
        m, n, k = shape[0], shape[1], shape[2]
        
        # 如果没有指定算子类型，返回所有参数组合
        if self.operator_type is None:
            return self.grid_parameters
        
        # 根据算子类型进行筛选
        if self.operator_type.lower() == 'smallmatmul':
            filtered_params = []
            for param in self.grid_parameters:
                # 将tile编号转换为实际值（tile × 16）
                m1 = param['mTile'] * 16
                n1 = param['nTile'] * 16
                k1 = param['kTile'] * 16
                
                # 检查是否满足SmallMatmul约束
                # 默认使用RowMajor布局（layout_tag_a=0, layout_tag_b=0）
                # 注意：如果需要支持其他layout组合，可以在这里循环检查所有组合
                if self.check_smallmatmul_constraints(m, n, k, m1, n1, k1, 0, 0):
                    filtered_params.append(param)
            
            if len(filtered_params) == 0:
                print(f"Warning: No valid tiling parameters found for SmallMatmul with shape {shape}")
            
            return filtered_params
        elif self.operator_type.lower() in ['commonmatmul', 'paddingcommonmatmul']:
            # 对于其他算子类型，暂时返回所有参数
            # 可以根据需要添加相应的筛选逻辑
            return self.grid_parameters
        else:
            # 未知的算子类型，返回所有参数
            print(f"Warning: Unknown operator type '{self.operator_type}', returning all parameters")
            return self.grid_parameters
        

    def get_params_with_idx(self, shape, idx):
        """
        根据索引获取特定shape的参数
        """
        params = self.filter_parameters(shape)
        if idx >= len(params):
            raise IndexError(f"Index {idx} out of range for {len(params)} parameters")
        return params[idx]


@dataclass
class CatlassResult():
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

class GEMMBenchmarkRunner():
    def __init__(self, shape_group, rank_id, num_processes, catlass_bin_path, 
                 result_dir="./results", msp_dir="./msp", operator_type=None, core_num=24):
        self.shape_group = shape_group
        self.result_dir = result_dir
        self.parameters = CatlassParameter(operator_type=operator_type, core_num=core_num)
        self.parameter_cache = []
        self.msp_dir = msp_dir
        self.catlass_bin_path = catlass_bin_path
        self.rank_id = rank_id
        self.num_processes = num_processes
        self.operator_type = operator_type
        # 为每个进程创建独立的msp目录，避免多进程冲突
        self.rank_msp_dir = os.path.join(msp_dir, f"rank_{rank_id}")
        os.makedirs(self.rank_msp_dir, exist_ok=True)
        # 预编译正则表达式，避免重复编译
        self.error_pattern = re.compile(r'Max Relative Error:\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)')
        self.time_pattern = re.compile(r'Task Duration\(us\): (\d+\.\d+)')
        self.kernel_func_name_pattern = re.compile(r'Kernel Func Name:\s*(.+)', re.MULTILINE)
        # Checkpoint更新频率：每10个任务更新一次（优化：减少I/O操作）
        self.checkpoint_interval = 10
        # 预构建msprof命令模板，使用进程特定的输出目录
        self.msprof_cmd_template = f"msprof op --output={self.rank_msp_dir} --aic-metrics='PipeUtilization' {self.catlass_bin_path}"
        # 结果缓冲区：批量写入减少I/O开销（优化：增大缓冲区）
        self.result_buffer = []
        self.result_buffer_size = 10
        # 缓存shape字符串，避免重复计算
        self.shape_str_cache = {}
    
    # ms_prof -> save_result
    def benchmark_shape(self, shape: list) -> None:
        # 确保结果目录存在
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 生成当前进程的文件路径（使用缓存）
        shape_tuple = tuple(shape)
        if shape_tuple not in self.shape_str_cache:
            self.shape_str_cache[shape_tuple] = '_'.join(map(str, shape))
        shape_str = self.shape_str_cache[shape_tuple]
        result_filename = f'shape_{shape_str}_rank_{self.rank_id}.jsonl'
        result_path = str(Path(self.result_dir) / result_filename)
        checkpoint_filename = f'shape_{shape_str}_rank_{self.rank_id}_checkpoint.jsonl'
        checkpoint_path = str(Path(self.result_dir) / checkpoint_filename)
        
        # 分配rank_id对应的【tiling参数组合 & 任务范围】
        filter_params = self.parameters.filter_parameters(shape)
        
        # 输出筛选信息
        if self.operator_type:
            print(f"Rank {self.rank_id}: Operator type '{self.operator_type}', "
                  f"filtered {len(filter_params)} valid tiling combinations for shape {shape}")
        
        tasks_per_process = math.ceil(len(filter_params) / self.num_processes)
        total_tasks = len(filter_params)
        start_idx = self.rank_id * tasks_per_process
        end_idx = min(start_idx + tasks_per_process, total_tasks)
        process_params = filter_params[start_idx:end_idx]
        process_task_count = len(process_params)

        # 优化：加载断点信息，使用更高效的读取方式
        last_process_idx = -1
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    # 优化：直接读取第一行（紧凑JSON格式）
                    line = f.readline().strip()
                    if line:
                        checkpoint = json.loads(line)
                        last_process_idx = checkpoint.get('last_process_idx', -1)
            except (json.JSONDecodeError, IOError):
                # 兼容旧格式
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                    last_process_idx = checkpoint.get('last_process_idx', -1)

        # 计算rank_id进程对应的start_local_idx
        start_local_idx = 0
        if last_process_idx >= start_idx:
            start_local_idx = last_process_idx - start_idx
            if start_local_idx + 1 >= process_task_count:
                print(f"Rank {self.rank_id} 已完成所有任务，无需继续处理")
                return
        
        completed_count = max(0, last_process_idx - start_idx + 1) if last_process_idx >= start_idx else 0
        with tqdm(total=process_task_count, initial=completed_count, desc=f"Rank {self.rank_id} Testing shape {shape}", postfix={"Processed": completed_count}) as pbar:
            # 从断点开始处理任务
            local_idx = start_local_idx
            while local_idx < process_task_count:
                global_idx = start_idx + local_idx
                parameters = process_params[local_idx]
                
                # 检查是否需要跳过的错误索引
                if global_idx == last_process_idx:
                    print(f"Rank {self.rank_id} 跳过异常 Tiling组合索引: {global_idx}")
                    wrong_result = CatlassResult(
                        idx=global_idx,
                        M=shape[0],
                        N=shape[1],
                        K=shape[2],
                        time=-1,
                        diff=float('inf'),
                        kernel_func_name="",
                        parameters=parameters,
                    )
                    # 优化：使用缓冲区批量写入
                    self.result_buffer.append(wrong_result)
                    if len(self.result_buffer) >= self.result_buffer_size:
                        self.flush_results(result_path)
                    local_idx += 1
                    pbar.update(1)
                    continue

                # 更新checkpoint（降低频率，每N个任务更新一次）
                if local_idx % self.checkpoint_interval == 0:
                    # 优化：使用紧凑JSON格式，减少I/O开销
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        f.write(f'{{"last_process_idx":{global_idx}}}\n')

                # 【核心计算】使用msprof同时收集精度、耗时和PipeUtilization数据，verify开关打开
                # 优化：预构建参数字符串模板，减少格式化开销
                param_str = f" {shape[0]} {shape[1]} {shape[2]} {parameters['mTile']} {parameters['nTile']} {parameters['kTile']} 0 0 1 {self.rank_id}"
                time_us, diff, kernel_func_name, pipe_utilization_data = self.ms_prof(param_str)

                # 检查是否超时：如果time_us为None，说明命令超时，跳过该组合，不记录结果
                if time_us is None:
                    print(f"Rank {self.rank_id} 跳过超时的tile组合: mTile={parameters['mTile']}, nTile={parameters['nTile']}, kTile={parameters['kTile']}")
                    local_idx += 1
                    pbar.update(1)
                    continue

                # 【保存结果】使用缓冲区批量写入
                result = CatlassResult(
                    idx=global_idx,
                    M=shape[0],
                    N=shape[1],
                    K=shape[2],
                    time=time_us,
                    diff=diff,
                    kernel_func_name=kernel_func_name,
                    parameters=parameters,
                    pipe_utilization=pipe_utilization_data
                )
                self.result_buffer.append(result)
                
                # 缓冲区满时批量写入
                if len(self.result_buffer) >= self.result_buffer_size:
                    self.flush_results(result_path)

                local_idx += 1
                pbar.update(1)
                pbar.set_postfix({
                    'Processed': local_idx,
                    'Global Index': global_idx
                })
        
        # 循环结束时，刷新剩余结果并更新最终的checkpoint
        if len(self.result_buffer) > 0:
            self.flush_results(result_path)
        if process_task_count > 0:
            # 优化：使用紧凑JSON格式
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.write(f'{{"last_process_idx":{start_idx + process_task_count - 1}}}\n')
        return

    def find_opprof_folder(self, max_wait_time: float = 2.0) -> Optional[str]:
        """
        在当前进程的msp目录中查找最新的OPPROF文件夹
        增加重试机制，等待文件生成完成
        
        Args:
            max_wait_time: 最大等待时间（秒），默认2秒
        
        Returns:
            OPPROF文件夹路径，如果找不到则返回None
        """
        if not os.path.exists(self.rank_msp_dir):
            return None
        
        # 重试查找，等待文件生成（优化：减少等待时间）
        start_time = time.time()
        sleep_interval = 0.05  # 减少到50ms
        while time.time() - start_time < max_wait_time:
            # 查找所有OPPROF_开头的文件夹
            opprof_pattern = os.path.join(self.rank_msp_dir, "OPPROF_*")
            opprof_folders = glob.glob(opprof_pattern)
            
            if not opprof_folders:
                time.sleep(sleep_interval)
                continue
            
            # 按修改时间排序，获取最新的文件夹
            try:
                opprof_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_folder = opprof_folders[0]
                
                # 检查文件夹中是否有PipeUtilization.xlsx或PipeUtilization.csv文件
                xlsx_path = os.path.join(latest_folder, "PipeUtilization.xlsx")
                csv_path = os.path.join(latest_folder, "PipeUtilization.csv")
                
                # 优先检查xlsx，然后检查csv
                file_path = None
                if os.path.exists(xlsx_path):
                    file_path = xlsx_path
                elif os.path.exists(csv_path):
                    file_path = csv_path
                
                if file_path:
                    # 额外检查文件是否可读（确保文件已完全写入）
                    try:
                        # 尝试打开文件，确保文件已完全生成
                        with open(file_path, 'rb') as f:
                            f.read(1)  # 读取第一个字节，检查文件是否可读
                        return latest_folder
                    except (IOError, PermissionError):
                        # 文件可能还在写入中，继续等待
                        time.sleep(sleep_interval)
                        continue
            except Exception as e:
                # 只在verbose模式下输出错误
                if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                    print(f"[find_opprof_folder] Rank {self.rank_id} 错误: {e}")
                time.sleep(sleep_interval)
                continue
        
        return None
    
    def parse_pipe_utilization(self, opprof_folder: str) -> Dict[str, Any]:
        """
        解析OPPROF文件夹中的PipeUtilization文件（支持xlsx和csv格式）
        第一行是表头，数据从第二行开始
        
        Args:
            opprof_folder: OPPROF文件夹路径
        
        Returns:
            解析后的数据字典，如果解析失败返回空字典
        """
        
        # 优先查找xlsx文件，然后查找csv文件
        xlsx_path = os.path.join(opprof_folder, "PipeUtilization.xlsx")
        csv_path = os.path.join(opprof_folder, "PipeUtilization.csv")
        
        file_path = None
        file_type = None
        
        if os.path.exists(xlsx_path):
            file_path = xlsx_path
            file_type = "xlsx"
        elif os.path.exists(csv_path):
            file_path = csv_path
            file_type = "csv"
        else:
            print(f"[parse_pipe_utilization] Rank {self.rank_id} 警告: 文件不存在（xlsx和csv都不存在）")
            return {}
        
        try:
            # 根据文件类型选择相应的读取方法
            if file_type == "xlsx":
                # 读取xlsx文件，第一行作为表头（header=0是默认值）
                # 使用上下文管理器确保文件及时关闭
                with pd.ExcelFile(file_path, engine='openpyxl') as excel_file:
                    df = pd.read_excel(excel_file, header=0)
            else:  # csv
                # 读取csv文件，第一行作为表头
                df = pd.read_csv(file_path, header=0, encoding='utf-8')
            
            if df.empty:
                print(f"[parse_pipe_utilization] Rank {self.rank_id} 警告: 文件为空（只有表头，没有数据行）")
                # 即使没有数据行，也返回列名信息
                return {col: None for col in df.columns}
            
            # 将DataFrame转换为字典
            # 取第一行数据（索引0），因为第一行是表头，数据从第二行开始
            if len(df) > 0:
                result = df.iloc[0].to_dict()
            else:
                print(f"[parse_pipe_utilization] Rank {self.rank_id} 警告: 文件没有数据行")
                return {col: None for col in df.columns}
            
            # 处理NaN值，转换为None
            def clean_value(v):
                try:
                    if pd.isna(v):
                        return None
                    # 尝试转换为Python原生类型
                    if isinstance(v, (int, float)):
                        return int(v) if isinstance(v, (int, float)) and v == int(v) else float(v)
                    return v
                except:
                    return None
            
            result = {k: clean_value(v) for k, v in result.items()}
            return result
        except Exception as e:
            # 只在verbose模式下输出详细错误
            if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                print(f"[parse_pipe_utilization] Rank {self.rank_id} 解析失败: {e}")
                import traceback
                traceback.print_exc()
            return {}
    
    def cleanup_opprof_folder(self, opprof_folder: str) -> None:
        """
        删除OPPROF文件夹以节省空间
        增加重试机制，确保删除成功
        
        Args:
            opprof_folder: 要删除的OPPROF文件夹路径
        """
        if not opprof_folder or not os.path.exists(opprof_folder):
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 先尝试关闭可能打开的文件句柄
                shutil.rmtree(opprof_folder)
                # 验证删除是否成功
                if not os.path.exists(opprof_folder):
                    return
                else:
                    # 删除失败，等待后重试
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
                        continue
            except PermissionError as e:
                # 文件可能被占用，等待后重试
                if attempt < max_retries - 1:
                    time.sleep(0.3)
                    continue
                else:
                    print(f"[cleanup_opprof_folder] Rank {self.rank_id} 删除失败（权限错误）: {e}")
            except Exception as e:
                print(f"[cleanup_opprof_folder] Rank {self.rank_id} 删除失败: {e}")
                if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                    import traceback
                    traceback.print_exc()
                break
    
    def ms_prof(self, param_str, max_retries: int = 3, timeout: int = 15) -> Tuple[Optional[float], float, str, Dict[str, Any]]:
        """
        执行msprof命令并同时解析时间、精度、kernel函数名和PipeUtilization数据
        如果解析到的时间是0.0，会重新执行命令（最多重试max_retries次）
        如果命令执行超过timeout秒，会跳过该组合，返回None标记
        
        Args:
            param_str: 命令参数字符串
            max_retries: 最大重试次数，默认3次
            timeout: 命令执行超时时间（秒），默认15秒
        
        Returns:
            (time_us, diff, kernel_func_name, pipe_utilization_data)
            如果超时，time_us为None，其他值保持默认
        """
        # 优化：使用预构建的命令模板，避免重复字符串拼接
        full_cmd = self.msprof_cmd_template + param_str
        
        # 打印将要执行的命令（可选，可以通过环境变量控制）
        if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
            print(f"[ms_prof] 执行命令: {full_cmd}")
        
        def parse_output(output: str):
            """解析输出中的时间、精度和kernel函数名（优化版本）"""
            time_us = None
            diff = float('inf')
            kernel_func_name = ""
            
            # 优化：一次性搜索所有模式，减少字符串遍历次数
            time_match = self.time_pattern.search(output)
            if time_match:
                time_us = float(time_match.group(1))
            
            error_match = self.error_pattern.search(output)
            if error_match:
                try:
                    diff = float(error_match.group(1))
                except ValueError:
                    pass  # 保持默认值 float('inf')
            
            kernel_match = self.kernel_func_name_pattern.search(output)
            if kernel_match:
                # 优化：减少字符串操作，直接使用strip和split的组合
                kernel_func_name = ' '.join(kernel_match.group(1).split())
            
            return time_us, diff, kernel_func_name
        
        # 重试逻辑
        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    full_cmd, 
                    shell=True, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=timeout  # 添加15秒超时
                )
                
                # 优化：合并 stdout 和 stderr，减少字符串操作
                # 因为 msprof 可能将信息输出到 stderr，先检查stderr是否为空
                if result.stderr:
                    stdout_text = result.stdout.decode('utf-8', errors='ignore')
                    stderr_text = result.stderr.decode('utf-8', errors='ignore')
                    combined_output = stdout_text + stderr_text
                else:
                    combined_output = result.stdout.decode('utf-8', errors='ignore')
                
                time_us, diff, kernel_func_name = parse_output(combined_output)
                
                if time_us is None:
                    # 解析失败，静默重试
                    if attempt < max_retries:
                        continue
                    else:
                        # 只在verbose模式下输出错误
                        if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                            print(f"[ms_prof] Rank {self.rank_id} 解析失败，输出内容（前200字符）: {combined_output[:200]}")
                        return (999999999, diff, kernel_func_name, {})
                
                # 检查时间值是否为0.0或异常值
                if time_us == 0.0:
                    if attempt < max_retries:
                        continue
                    else:
                        # 只在verbose模式下输出警告
                        if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                            print(f"[ms_prof] Rank {self.rank_id} 警告：重试{max_retries}次后仍为0.0，返回999999999")
                        return (999999999, diff, kernel_func_name, {})
                
                # 时间值有效，查找并解析PipeUtilization.xlsx
                pipe_utilization_data = {}
                # 等待一小段时间，确保OPPROF文件夹已完全生成（优化：减少等待时间）
                time.sleep(0.3)  # 从500ms减少到300ms
                
                # 查找OPPROF文件夹（带重试机制，最多等待1.5秒，优化：减少等待时间）
                opprof_folder = self.find_opprof_folder(max_wait_time=1.5)
                if opprof_folder:
                    # 先解析文件（支持xlsx和csv格式）
                    pipe_utilization_data = self.parse_pipe_utilization(opprof_folder)
                    # 等待一小段时间，确保文件句柄已关闭（优化：减少等待时间）
                    time.sleep(0.05)  # 从100ms减少到50ms
                    # 读取后立即删除文件夹以节省空间
                    self.cleanup_opprof_folder(opprof_folder)
                else:
                    # 只在verbose模式下输出警告
                    if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                        print(f"[ms_prof] Rank {self.rank_id} 警告：未找到OPPROF文件夹或PipeUtilization文件（xlsx/csv）")
                
                return (time_us, diff, kernel_func_name, pipe_utilization_data)
                
            except subprocess.TimeoutExpired:
                # 超时情况：不重试，直接返回None标记，让调用者跳过该组合
                print(f"[ms_prof] Rank {self.rank_id} 命令执行超时（>{timeout}秒），跳过该tile组合")
                return (None, float('inf'), "", {})
                
            except Exception as e:
                if attempt < max_retries:
                    # 静默重试，只在verbose模式下输出
                    if os.getenv('BENCHMARK_VERBOSE', '0') == '1':
                        print(f"[ms_prof] Rank {self.rank_id} 第{attempt+1}次尝试失败: {e}，重试中...")
                    continue
                else:
                    # 只在最后一次失败时输出错误
                    print(f"[ms_prof] Rank {self.rank_id} 执行失败，错误: {e}")
                    return (999999999, float('inf'), "", {})
        
        # 理论上不会到达这里，但为了安全起见
        return (999999999, float('inf'), "", {})

    def flush_results(self, path: str) -> None:
        """
        批量写入结果到文件，减少I/O开销
        """
        if not self.result_buffer:
            return
        
        try:
            # 优化：批量写入，减少文件打开/关闭次数
            # 使用更大的缓冲区（8KB）减少系统调用
            with open(path, 'a', encoding='utf-8', buffering=8192) as f:
                for result in self.result_buffer:
                    # 优化：使用紧凑JSON格式，减少文件大小和I/O时间
                    json.dump(asdict(result), f, ensure_ascii=False, separators=(',', ':'))
                    f.write('\n')
            self.result_buffer.clear()
        except IOError as e:
            print(f"save files error: {e}")
        except Exception as e:
            print(f"process data error: {e}")
    
    def save_result(self, result: CatlassResult, path: str) -> None:
        """
        保存单个结果（兼容性方法，实际使用flush_results批量写入）
        """
        self.result_buffer.append(result)
        if len(self.result_buffer) >= self.result_buffer_size:
            self.flush_results(path)
    
    def run_benchmarks(self) -> None:
        print("=====STARTING GEMM BENCHMARK=====")
        for shape in self.shape_group:
            self.benchmark_shape(shape)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='%(prog)s --rank_id [num] --process_num [num] [--operator_type TYPE]'
    )
    parser.add_argument('--rank_id', required=True, type=int, help='进程的rank ID，从0开始')
    parser.add_argument('--process_num', required=True, type=int, help='总进程数')
    parser.add_argument('--catlass_bin_path', type=str, default="/home/q30063557/code/cutlass/21_dynamic_tiling_matmul",
                       help='catlass可执行文件路径')
    parser.add_argument('--result_dir', type=str, default="./catlass_results",
                       help='结果保存目录，默认: ./catlass_results')
    parser.add_argument('--msp_dir', type=str, default="./catlass_msp",
                       help='msprof输出目录，默认: ./catlass_msp')
    parser.add_argument('--operator_type', type=str, default=None,
                       choices=['smallmatmul', 'commonmatmul', 'paddingcommonmatmul'],
                       help='算子类型，可选: smallmatmul, commonmatmul, paddingcommonmatmul. 默认None表示所有算子')
    parser.add_argument('--core_num', type=int, default=20,
                       help='AI Core数量，默认20')
    parser.add_argument('--shapes_file', type=str, default=None,
                       help='shapes.xlsx文件路径，如果提供则从文件读取shape，否则使用默认shape_group')
    args = parser.parse_args()
    
    # 根据是否提供shapes_file来决定使用哪个shape_group
    if args.shapes_file:
        print(f"=====Loading shapes from {args.shapes_file}=====")
        shape_group = load_shapes_from_excel(args.shapes_file)
    else:
        print(f"=====Using default shape_group=====")
        shape_group = default_shape_group
    
    print(f"=====STARTING GEMM BENCHMARK (Rank {args.rank_id}/{args.process_num})=====")
    print(f"Total shapes to test: {len(shape_group)}")
    if args.operator_type:
        print(f"Operator type: {args.operator_type}")
        print(f"Core number: {args.core_num}")
    
    # 运行完整基准测试
    runner = GEMMBenchmarkRunner(
        shape_group,
        rank_id=args.rank_id,
        num_processes=args.process_num,
        catlass_bin_path=args.catlass_bin_path,
        result_dir=args.result_dir,
        msp_dir=args.msp_dir,
        operator_type=args.operator_type,
        core_num=args.core_num
    )
    runner.run_benchmarks()    
    print(f"=====GEMM BENCHMARK FINISHED (Rank {args.rank_id})=====")
