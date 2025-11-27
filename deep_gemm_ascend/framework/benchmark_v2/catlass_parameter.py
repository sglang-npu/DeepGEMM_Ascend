"""
Catlass参数生成模块

提供tiling参数生成和过滤功能
"""

import math
from typing import List, Dict

from .utils.common import ceil_div
from .padding_calculator import PaddingCalculator, PaddingTag


class CatlassParameter:
    """
    为 catlass 生成 tiling 网格参数
    参数：mTile, nTile, kTile
    约束条件：
        - 各维度元素数量为 tile × 16，矩阵元素数量需考虑两次 16
        - L0C = (mTile×16) × (nTile×16) × 4 字节 ≤ 128KB
        - L0A = (mTile×16) × (1×16) × double_buffer × 2 字节 ≤ 64KB（kTile固定为1）
        - L0B = (nTile×16) × (1×16) × double_buffer × 2 字节 ≤ 64KB（kTile固定为1）
        - L1 = (mTile×16 + nTile×16) × kTile×16 × 2 字节 < 512KB
    """
    
    def __init__(self, operator_type=None, core_num=20):
        """
        Args:
            operator_type: 算子类型，可选值: 'SmallMatmulKernel', 'CommonMatmulKernel', 'PaddingMatmulKernel', None(所有)
            core_num: AI Core数量，默认20
        """
        self.operator_type = operator_type
        self.core_num = core_num
        self.grid_parameters = self.grid_generate_parameters()

    def generate_mn_tile_values(self):
        """
        生成mTile和nTile的值（范围1-64）
        策略：多粒度采样，考虑L0C/L0A/L0B约束
        
        约束分析：
        1. L0C约束：mTile × nTile ≤ 64
        2. L0A/L0B约束：kTile固定为1，因此 mTile ≤ 64, nTile ≤ 64
        
        """
        representative_tiles = [
            1, 2, 3, 4, 6, 8,
            12, 16, 20, 24, 28, 32,
            40, 48, 56, 64
        ]
        return representative_tiles
    
    def generate_k_tile_values(self):
        """
        生成kTile的值（范围1-128）

        """
        representative_tiles = [
            1, 2, 3, 4, 6, 8,
            12, 16, 20, 24, 28, 32,
            40, 48, 56, 64,
            80, 96, 112, 128
        ]
        return representative_tiles

    def generate_mnk_tiles_linear(self):
        """
        线性生成 mTile, nTile, kTile 的所有组合
        约束条件：
        1. L0C = (mTile×16) × (nTile×16) × 4 字节 ≤ 128KB
        2. L0A = (mTile×16) × (1×16) × double_buffer × 2 字节 ≤ 64KB（kTile固定1）
        3. L0B = (nTile×16) × (1×16) × double_buffer × 2 字节 ≤ 64KB（kTile固定1）
        4. L1 = (mTile×16 + nTile×16) × kTile×16 × 2 字节 x double_buffer < 512KB
        5. mTile/nTile范围：1-64，kTile范围：1-128
        """
        mn_tile_values = self.generate_mn_tile_values()
        k_tile_values = self.generate_k_tile_values()
        double_buffer = 2
        L1_CACHE_MAX_BYTES = 512 * 1024 // double_buffer  # 256KB
        L0C_MAX_BYTES = 128 * 1024  # 128KB
        L0A_MAX_BYTES = 64 * 1024  # 64KB
        L0B_MAX_BYTES = 64 * 1024  # 64KB
        
        # 预计算常量，避免重复计算
        L0C_FACTOR = 16 * 16 * 4  # 1024
        L0C_MAX_TILE_PRODUCT = L0C_MAX_BYTES // L0C_FACTOR  # 128
        
        # L0A/L0B约束：简化计算
        # L0A = (mTile*16) * (kTile*16) * double_buffer * 2
        #     = mTile * kTile * 16 * 16 * 2 * 2
        #     = mTile * kTile * 1024
        # 约束：mTile * kTile * 1024 <= 64KB = 65536
        # 简化：mTile * kTile <= 64
        L0A_MAX_TILE_PRODUCT = L0A_MAX_BYTES // (16 * 16 * double_buffer * 2)  # 64
        
        # L0B = (nTile*16) * (kTile*16) * double_buffer * 2
        #     = nTile * kTile * 1024
        # 约束：nTile * kTile <= 64
        L0B_MAX_TILE_PRODUCT = L0B_MAX_BYTES // (16 * 16 * double_buffer * 2)  # 64
        
        for m_tile in mn_tile_values:
            for n_tile in mn_tile_values:
                # L0C bytes = (mTile*16) * (nTile*16) * 4 = mTile * nTile * 1024
                # 优化：直接比较 mTile * nTile，避免乘法运算
                if m_tile * n_tile > L0C_MAX_TILE_PRODUCT:
                    continue
                
                # 预计算 (mTile*16 + nTile*16)，避免在k循环中重复计算
                m_plus_n_16 = (m_tile + n_tile) * 16
                
                for k_tile in k_tile_values:
                    # L0A约束：mTile * 1 <= 64
                    if m_tile > L0A_MAX_TILE_PRODUCT:
                        continue
                    
                    # L0B约束：nTile * 1 <= 64
                    if n_tile > L0B_MAX_TILE_PRODUCT:
                        continue
                    
                    # L1 bytes = (mTile*16 + nTile*16) * kTile*16 * 2
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
        print(f'  Valid combinations (after L0C/L0A/L0B/L1 constraints): {len(parameters):,}')
        print(f'  Constraint filtering rate: {(1 - len(parameters) / (len(mn_tile_values)**2 * len(k_tile_values))) * 100:.1f}%')
        print(f'  Constraints applied:')
        print(f'    - L0C: mTile × nTile ≤ 128')
        print(f'    - L0A: mTile ≤ 64 (kTile=1)')
        print(f'    - L0B: nTile ≤ 64 (kTile=1)')
        print(f'    - L1: (mTile×16 + nTile×16) × kTile×16 × 2 < 256KB')
        print(f'=' * 60)
        
        return parameters

    def check_smallmatmul_constraints(self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0):
        """
        检查shape和tiling参数是否满足SmallMatmul算子的约束条件
        
        SmallMatmul约束条件：
        1. Tile数量: ⌈m/m1⌉ × ⌈n/n1⌉ ≤ coreNum
        2. K轴: k ≤ k1
        3. Padding: 通过 calc_padding_tags 模拟 C++ GetPaddingTag，要求 PaddingTagA/B/C == PADDING_NONE
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: Tiling参数（实际值，不是tile编号）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
        
        Returns:
            bool: 是否满足SmallMatmul约束
        """
        # 约束1: Tile数量约束
        # ⌈m/m1⌉ × ⌈n/n1⌉ ≤ coreNum
        task_blocks = math.ceil(m / m1) * math.ceil(n / n1)
        if task_blocks > self.core_num:
            return False
        
        # 约束2: K轴约束
        if k > k1:
            return False
        
        padding_a, padding_b, padding_c = PaddingCalculator.calc_padding_tags(
            m,
            n,
            k,
            m1,
            n1,
            k1,
            layout_tag_a,
            layout_tag_b,
            splitk_factor=1,
            core_num=self.core_num,
        )

        if any(
            tag != PaddingTag.PADDING_NONE
            for tag in (padding_a, padding_b, padding_c)
        ):
            return False

        return True

    def check_paddingmatmul_constraints(
        self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0
    ):
        """
        检查 shape 与 tiling 是否更适合 PaddingCommonMatmul

        要求：
        1. 至少一个 PaddingTag ≠ PADDING_NONE，用于触发 Padding kernel
        2. 其余基础约束已由网格生成阶段保证
        """
        padding_a, padding_b, padding_c = PaddingCalculator.calc_padding_tags(
            m,
            n,
            k,
            m1,
            n1,
            k1,
            layout_tag_a,
            layout_tag_b,
            splitk_factor=1,
            core_num=self.core_num,
        )

        return any(
            tag != PaddingTag.PADDING_NONE for tag in (padding_a, padding_b, padding_c)
        )
    
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
        operator_type_lower = self.operator_type.lower() if self.operator_type else None
        
        if operator_type_lower == 'smallmatmulkernel':
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
        elif operator_type_lower == 'paddingmatmulkernel':
            filtered_params = []
            for param in self.grid_parameters:
                m1 = param['mTile'] * 16
                n1 = param['nTile'] * 16
                k1 = param['kTile'] * 16

                if self.check_paddingmatmul_constraints(
                    m, n, k, m1, n1, k1, 0, 0
                ):
                    filtered_params.append(param)

            if len(filtered_params) == 0:
                print(
                    f"Warning: No valid tiling parameters found for {self.operator_type} with shape {shape}"
                )

            return filtered_params
        elif operator_type_lower == 'commonmatmulkernel':
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

