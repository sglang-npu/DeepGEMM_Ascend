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
    
    def __init__(self, operator_type, core_num=20, layout_tag_a=None, layout_tag_b=None):
        """
        Args:
            operator_type: 算子类型，必须提供，可选值: 
                - 'SmallMatmulKernel': SmallMatmul算子
                - 'CommonMatmulKernel': CommonMatmul算子
                - 'PaddingCommonMatmulKernel': PaddingCommonMatmul算子
                - 'PaddingMultiCoreSplitkMatmulKernel': PaddingMultiCoreSplitkMatmul算子
                - 'PaddingStreamkMatmulKernel': PaddingStreamkMatmul算子
            core_num: AI Core数量，默认20
            layout_tag_a: Layout A标签，必须提供，0=RowMajor, 1=ColumnMajor
            layout_tag_b: Layout B标签，必须提供，0=RowMajor, 1=ColumnMajor
        
        Raises:
            ValueError: 如果 operator_type、layout_tag_a 或 layout_tag_b 未提供
        """
        if operator_type is None or layout_tag_a is None or layout_tag_b is None:
            missing = []
            if operator_type is None:
                missing.append("operator_type")
            if layout_tag_a is None:
                missing.append("layout_tag_a")
            if layout_tag_b is None:
                missing.append("layout_tag_b")
            raise ValueError(f"以下参数必须提供，不能为 None: {', '.join(missing)}")
        
        self.operator_type = operator_type
        self.core_num = core_num
        self.layout_tag_a = layout_tag_a
        self.layout_tag_b = layout_tag_b
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
        """生成kTile的值（范围1-128）"""
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
        
        # L0A/L0B约束：mTile * kTile <= 64, nTile * kTile <= 64
        L0A_MAX_TILE_PRODUCT = L0A_MAX_BYTES // (16 * 16 * double_buffer * 2)  # 64
        L0B_MAX_TILE_PRODUCT = L0B_MAX_BYTES // (16 * 16 * double_buffer * 2)  # 64
        
        for m_tile in mn_tile_values:
            for n_tile in mn_tile_values:
                if m_tile * n_tile > L0C_MAX_TILE_PRODUCT:
                    continue
                
                m_plus_n_16 = (m_tile + n_tile) * 16
                
                for k_tile in k_tile_values:
                    if m_tile > L0A_MAX_TILE_PRODUCT or n_tile > L0B_MAX_TILE_PRODUCT:
                        continue
                    
                    l1_bytes = m_plus_n_16 * k_tile * 32
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
        padding_a, padding_b, padding_c = PaddingCalculator.calc_padding_tags(
            m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b,
            splitk_factor=1, core_num=self.core_num,
        )

        if any(tag != PaddingTag.PADDING_NONE for tag in (padding_a, padding_b, padding_c)):
            return False

        task_blocks = math.ceil(m / m1) * math.ceil(n / n1)
        return task_blocks <= self.core_num and k <= k1

        return True

    def check_padding_common_matmul_constraints(
        self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0
    ):
        """
        检查 shape 与 tiling 是否更适合 PaddingCommonMatmul

        要求：
        1. 至少一个 PaddingTag ≠ PADDING_NONE，用于触发 Padding kernel
        2. 其余基础约束已由网格生成阶段保证
        """
        padding_a, padding_b, padding_c = PaddingCalculator.calc_padding_tags(
            m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b,
            splitk_factor=1, core_num=self.core_num,
        )
        return any(tag != PaddingTag.PADDING_NONE for tag in (padding_a, padding_b, padding_c))

    def check_padding_multicore_splitk_constraints(
        self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0
    ):
        """
        检查是否满足 PaddingMultiCoreSplitkMatmul 的条件
        
        根据 C++ 代码 PaddingMultiCoreSplitkMatmulB16Handler：
        1. 根据 layout 确定 m1t, n1t, k1t (默认 128, 256, 256)
        2. 计算 blocks = CeilDiv(m, m1t) * CeilDiv(n, n1t)
        3. 根据 k 确定 maxSplitkFactor
        4. 条件：(blocks <= coreNum/2 && k > 5120) || (blocks <= 2 && k > 1024)
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: 当前 tiling 参数（实际值）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
        
        Returns:
            bool: 是否满足 PaddingMultiCoreSplitkMatmul 条件
        """
        m1t, n1t = 128, 256
        if (layout_tag_a == 1 and layout_tag_b == 1) or (layout_tag_a == 1 and layout_tag_b == 0 and m > n):
            m1t, n1t = 256, 128
        
        blocks = ceil_div(m, m1t) * ceil_div(n, n1t)
        return (blocks <= self.core_num // 2 and k > 5120) or (blocks <= 2 and k > 1024)

    def check_padding_streamk_constraints(
        self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0
    ):
        """
        检查是否满足 PaddingStreamkMatmul 的条件
        
        根据 C++ 代码 PaddingStreamkMatmulB16Handler：
        1. 根据 layout 确定 m1t, n1t, k1t (默认 128, 256, 256)
        2. 计算 blocks = CeilDiv(m, m1t) * CeilDiv(n, n1t)
        3. 计算 skBlocks = blocks % coreNum
        4. 条件：blocks > coreNum && blocks < 8*coreNum && skBlocks > 0 
                && skBlocks < 0.8*coreNum && k > 3072
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: 当前 tiling 参数（实际值）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
        
        Returns:
            bool: 是否满足 PaddingStreamkMatmul 条件
        """
        m1t, n1t = 128, 256
        if (layout_tag_a == 1 and layout_tag_b == 1) or (layout_tag_a == 1 and layout_tag_b == 0 and m > n):
            m1t, n1t = 256, 128
        
        blocks = ceil_div(m, m1t) * ceil_div(n, n1t)
        sk_blocks = blocks % self.core_num
        return (blocks > self.core_num and blocks < 8 * self.core_num 
                and sk_blocks > 0 and sk_blocks < 0.8 * self.core_num and k > 3072)

    def check_commonmatmul_constraints(
        self, m, n, k, m1, n1, k1, layout_tag_a=0, layout_tag_b=0
    ):
        """
        检查 shape 和 tiling 参数是否会被选择为 CommonMatmul 算子
        
        CommonMatmul 是兜底 handler，只有当所有前面的 handler 都返回 false 时才会被选择。
        
        选择顺序（按优先级）：
        1. SmallMatmulB16Handler
        2. PaddingMultiCoreSplitkMatmulB16Handler
        3. PaddingStreamkMatmulB16Handler
        4. PaddingCommonMatmulB16Handler
        5. CommonMatmulB16Handler (兜底，总是返回 true)
        
        CommonMatmul 被选择的条件是：
        - 不满足 SmallMatmul 条件
        - 不满足 PaddingMultiCoreSplitkMatmul 条件
        - 不满足 PaddingStreamkMatmul 条件
        - 不满足 PaddingCommonMatmul 条件
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: Tiling参数（实际值，不是tile编号）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
        
        Returns:
            bool: 是否会被选择为 CommonMatmul
        """
        checkers = [
            self.check_smallmatmul_constraints,
            self.check_padding_multicore_splitk_constraints,
            self.check_padding_streamk_constraints,
            self.check_padding_common_matmul_constraints,
        ]
        return not any(checker(m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b) for checker in checkers)
    
    def filter_parameters(self, shape, layout_tag_a=None, layout_tag_b=None):
        """
        根据shape和算子类型过滤参数
        
        Args:
            shape: [m, n, k] 矩阵维度
            layout_tag_a: Layout A标签，0=RowMajor, 1=ColumnMajor，如果为None则使用实例的layout_tag_a
            layout_tag_b: Layout B标签，0=RowMajor, 1=ColumnMajor，如果为None则使用实例的layout_tag_b
        
        Returns:
            过滤后的参数列表
        
        支持的算子类型：
            - 'SmallMatmulKernel': SmallMatmul算子
            - 'CommonMatmulKernel': CommonMatmul算子
            - 'PaddingCommonMatmulKernel': PaddingCommonMatmul算子
            - 'PaddingMultiCoreSplitkMatmulKernel': PaddingMultiCoreSplitkMatmul算子
            - 'PaddingStreamkMatmulKernel': PaddingStreamkMatmul算子
        
        Raises:
            ValueError: 如果 layout_tag_a 或 layout_tag_b 未提供且实例中也没有设置
        """
        m, n, k = shape[0], shape[1], shape[2]
        
        layout_a = layout_tag_a if layout_tag_a is not None else self.layout_tag_a
        layout_b = layout_tag_b if layout_tag_b is not None else self.layout_tag_b
        
        if layout_a is None or layout_b is None:
            missing = []
            if layout_a is None:
                missing.append("layout_tag_a")
            if layout_b is None:
                missing.append("layout_tag_b")
            raise ValueError(f"以下参数必须提供: {', '.join(missing)}")
        
        operator_type_lower = self.operator_type.lower()
        constraint_checkers = {
            'smallmatmulkernel': self.check_smallmatmul_constraints,
            'paddingcommonmatmulkernel': self.check_padding_common_matmul_constraints,
            'paddingmulticoresplitkmatmulkernel': self.check_padding_multicore_splitk_constraints,
            'paddingstreamkmatmulkernel': self.check_padding_streamk_constraints,
            'commonmatmulkernel': self.check_commonmatmul_constraints,
        }
        
        if operator_type_lower not in constraint_checkers:
            print(f"Warning: Unknown operator type '{self.operator_type}', returning all parameters")
            return self.grid_parameters
        
        checker = constraint_checkers[operator_type_lower]
        filtered_params = []
        for param in self.grid_parameters:
            m1 = param['mTile'] * 16
            n1 = param['nTile'] * 16
            k1 = param['kTile'] * 16
            if checker(m, n, k, m1, n1, k1, layout_a, layout_b):
                filtered_params.append(param)
        
        if len(filtered_params) == 0:
            print(f"Warning: No valid tiling parameters found for {self.operator_type} with shape {shape}, layout_a={layout_a}, layout_b={layout_b}")
        
        return filtered_params

    def get_params_with_idx(self, shape, idx, layout_tag_a=None, layout_tag_b=None):
        """
        根据索引获取特定shape的参数
        
        Args:
            shape: [m, n, k] 矩阵维度
            idx: 参数索引
            layout_tag_a: Layout A标签，0=RowMajor, 1=ColumnMajor，如果为None则使用实例的layout_tag_a
            layout_tag_b: Layout B标签，0=RowMajor, 1=ColumnMajor，如果为None则使用实例的layout_tag_b
        
        Returns:
            指定索引的参数字典
        """
        params = self.filter_parameters(shape, layout_tag_a, layout_tag_b)
        if idx >= len(params):
            raise IndexError(f"Index {idx} out of range for {len(params)} parameters")
        return params[idx]

