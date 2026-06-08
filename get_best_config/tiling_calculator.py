"""
Small Matmul Tiling 计算器
封装版本 - 提供简洁的 API 接口
"""

import importlib.util
import os
import sys
import types
from enum import IntEnum
from typing import Any, Dict, Tuple

from padding_calculator import PaddingCalculator, PaddingTag
from utils.common import ceil_div, round_up

class LayoutTag(IntEnum):
    """矩阵布局标签"""
    TagRowMajor = 0
    TagColumnMajor = 1


class PlatformInfo:
    """平台信息，默认值来自 platform_info.h"""
    def __init__(self):
        self.coreNum = 20          # AI Core 数量
        self.ubSize = 192 * 1024   # UB 大小 (字节)
        self.l1Size = 512 * 1024   # L1 大小 (字节)
        self.l0ASize = 64 * 1024   # L0 A 大小 (字节)
        self.l0BSize = 64 * 1024   # L0 B 大小 (字节)
        self.l0CSize = 128 * 1024  # L0 C 大小 (字节)


class OperatorType:
    """算子类型枚举"""
    SMALL_MATMUL = "SmallMatmul"
    COMMON_MATMUL = "CommonMatmul"
    PADDING_COMMON_MATMUL = "PaddingCommonMatmul"
    PADDING_MULTICORE_SPLITK_MATMUL = "PaddingMultiCoreSplitkMatmul"
    PADDING_STREAMK_MATMUL = "PaddingStreamkMatmul"


class MatmulTilingCalculator:
    """
    Small Matmul Tiling 计算器

    封装了完整的 tiling 计算和算子类型选择逻辑
    """

    def __init__(self, platform_info: PlatformInfo = None):
        """
        初始化计算器

        参数:
            platform_info: 平台信息，如果为 None 则使用默认值
        """
        if platform_info is None:
            self.platform = PlatformInfo()
        else:
            self.platform = platform_info

        # 布局映射表
        self.layout_map = {
            "RowMajor": LayoutTag.TagRowMajor,
            "ColumnMajor": LayoutTag.TagColumnMajor,
            "row": LayoutTag.TagRowMajor,
            "col": LayoutTag.TagColumnMajor,
            "0": LayoutTag.TagRowMajor,
            "1": LayoutTag.TagColumnMajor,
        }

    def balance_workload(self, m: int, n: int, m1: int, n1: int, threshold: int) -> Tuple[int, int]:
        """
        平衡工作负载，调整 m1 和 n1

        参数:
            m, n: 矩阵维度
            m1, n1: 当前的 tile 大小（会被修改）
            threshold: m1 的最小阈值
        返回:
            (m1, n1): 调整后的 tile 大小
        """
        max_blocks = round_up(ceil_div(m, m1) * ceil_div(n, n1), self.platform.coreNum)

        # 逐步减小 m1，直到达到阈值或无法满足 maxBlocks 约束
        while m1 > threshold and (ceil_div(m, m1 - 16) * ceil_div(n, n1) <= max_blocks):
            m1 -= 16

        # 如果实际维度小于 tile 大小，则调整为实际大小的对齐值
        if m < m1:
            m1 = round_up(m, 16)
        if n < n1:
            n1 = round_up(n, 16)

        return m1, n1

    def judge_space(self, m1: int, n1: int, k1: int, dtype_size: int = 2) -> bool:
        """
        判断给定的 tile 大小是否满足内存约束

        参数:
            m1, n1, k1: tile 大小
            dtype_size: 数据类型大小（字节），fp16/bf16 为 2
        返回:
            True 如果满足约束，False 否则
        """
        # L1 约束: A 和 B 矩阵的数据量不能超过 L1 大小
        # A 矩阵: m1 * k1 * 2 * dtype_size (2 是因为可能有 double buffer)
        # B 矩阵: k1 * n1 * 2 * dtype_size
        judge_l1 = (m1 * k1 * 2 * dtype_size + k1 * n1 * 2 * dtype_size <= self.platform.l1Size)

        # L0 C 约束: C 矩阵的数据量不能超过 L0 C 大小
        # C 矩阵: m1 * n1 * 4 (accumulator 通常是 float，4 字节)
        judge_l0c = (m1 * n1 * 4 <= self.platform.l0CSize)

        return judge_l1 and judge_l0c

    def get_max_k1(self, m1: int, n1: int, dtype_size: int = 2) -> int:
        """
        获取在给定 m1, n1 下能使用的最大 k1

        参数:
            m1, n1: tile 大小
            dtype_size: 数据类型大小（字节）
        返回:
            最大可用的 k1 值
        """
        k1_list = [1024, 512, 256, 128]
        k1_default = 512 // dtype_size  # fp16/bf16 为 256

        for k1t in k1_list:
            if self.judge_space(m1, n1, k1t, dtype_size):
                return k1t

        return k1_default

    def do_tiling_b16_layout00(self, m: int, n: int, k: int) -> Tuple[int, int, int]:
        """
        Layout 00: RowMajor A, RowMajor B
        对应 DoTilingB16Layout00
        """
        m1, n1, k1 = 128, 256, 256

        if n >= 256:
            # n0 = 256 提供最优带宽性能
            max_blocks = round_up(ceil_div(m, m1) * ceil_div(n, n1), self.platform.coreNum)
            m1, n1 = self.balance_workload(m, n, m1, n1, 32)
            blocks = ceil_div(m, 64) * ceil_div(n, 512)
            if blocks <= max_blocks - self.platform.coreNum and k <= 128:
                m1 = 64
                n1 = 512
        else:
            m1 = 128
            n1 = round_up(n, 16)
            max_blocks = round_up(ceil_div(m, m1) * ceil_div(n, n1), self.platform.coreNum)
            m1t = m1
            while self.judge_space(m1t + 16, n1, k1):
                m1t += 16
                blocks = ceil_div(m, m1t) * ceil_div(n, n1)
                if blocks <= max_blocks - self.platform.coreNum:
                    m1 = m1t
            m1, n1 = self.balance_workload(m, n, m1, n1, 32)

        if k >= 65536 or n >= 65536:
            m1 = 128
            n1 = 256

        k1 = self.get_max_k1(m1, n1)
        return m1, n1, k1

    def do_tiling_b16_layout01(self, m: int, n: int, k: int) -> Tuple[int, int, int]:
        """
        Layout 01: RowMajor A, ColumnMajor B
        对应 DoTilingB16Layout01
        """
        m1, n1, k1 = 128, 256, 256

        # 当 LayoutA 是 RowMajor 且 LayoutB 是 ColumnMajor 时，
        # 可以完全忽略带宽问题，只需选择最平衡的工作负载配置
        ratio = (m * k + k * n) / (m * n) if (m * n) > 0 else 0.0

        if m > n and (ratio > 0.1 or n < 256):
            m1 = 256
            n1 = 128
            m1, n1 = self.balance_workload(m, n, m1, n1, 64)
            n1, m1 = self.balance_workload(n, m, n1, m1, 64)
        else:
            n1, m1 = self.balance_workload(n, m, n1, m1, 64)
            m1, n1 = self.balance_workload(m, n, m1, n1, 64)

        max_blocks = round_up(ceil_div(m, m1) * ceil_div(n, n1), self.platform.coreNum)

        if m < n:
            n1t = n1
            while self.judge_space(m1, n1t + 16, k1):
                n1t += 16
                blocks = ceil_div(m, m1) * ceil_div(n, n1t)
                if blocks <= max_blocks - self.platform.coreNum:
                    n1 = n1t
            m1, n1 = self.balance_workload(m, n, m1, n1, 64)
            n1, m1 = self.balance_workload(n, m, n1, m1, 64)
        else:
            m1t = m1
            while self.judge_space(m1t + 16, n1, k1):
                m1t += 16
                blocks = ceil_div(m, m1t) * ceil_div(n, n1)
                if blocks <= max_blocks - self.platform.coreNum:
                    m1 = m1t
            n1, m1 = self.balance_workload(n, m, n1, m1, 64)
            m1, n1 = self.balance_workload(m, n, m1, n1, 64)

        if k >= 65536:
            if m < n or (ratio < 0.1 and n >= 256):
                m1 = 128
                n1 = 256
            else:
                m1 = 256
                n1 = 128

        k1 = self.get_max_k1(m1, n1)
        return m1, n1, k1

    def do_tiling_b16_layout10(self, m: int, n: int, k: int) -> Tuple[int, int, int]:
        """
        Layout 10: ColumnMajor A, RowMajor B
        对应 DoTilingB16Layout10
        """
        m1, n1, k1 = 128, 256, 256
        ratio = (m * k + k * n) / (m * n) if (m * n) > 0 else 0.0

        if m > n and (ratio > 0.1 or n < 256):
            m1 = 256
            n1 = 128
            k1 = 256

        if m < m1:
            m1 = round_up(m, 16)
        if n < n1:
            n1 = round_up(n, 16)

        blocks = ceil_div(m, m1) * ceil_div(n, n1)
        if blocks <= self.platform.coreNum // 4:
            if n1 > 16:
                n1 //= 2
            if m1 > 16:
                m1 //= 2
        elif blocks <= self.platform.coreNum // 2:
            if m1 > n1:
                m1 //= 2
            elif n1 > 16:
                n1 //= 2

        if n >= 65536 or m >= 65536:
            if m < n or (ratio < 0.1 and n >= 256):
                m1 = 128
                n1 = 256
            else:
                m1 = 256
                n1 = 128

        m1 = m1 // 16 * 16
        n1 = n1 // 16 * 16
        k1 = self.get_max_k1(m1, n1)
        return m1, n1, k1

    def do_tiling_b16_layout11(self, m: int, n: int, k: int) -> Tuple[int, int, int]:
        """
        Layout 11: ColumnMajor A, ColumnMajor B
        对应 DoTilingB16Layout11
        """
        m1, n1, k1 = 256, 128, 256

        if m >= 256:
            # n0 = 256 提供最优带宽性能
            max_blocks = round_up(ceil_div(m, m1) * ceil_div(n, n1), self.platform.coreNum)
            n1, m1 = self.balance_workload(n, m, n1, m1, 32)
            blocks = ceil_div(n, 64) * ceil_div(m, 512)
            if blocks <= max_blocks - self.platform.coreNum and k <= 128:
                n1 = 64
                m1 = 512
        else:
            n1 = 128
            m1 = round_up(m, 16)
            max_blocks = round_up(ceil_div(m, m1) * ceil_div(n, n1), self.platform.coreNum)
            n1t = n1
            while self.judge_space(n1t + 16, m1, k1):
                n1t += 16
                blocks = ceil_div(n, n1t) * ceil_div(m, m1)
                if blocks <= max_blocks - self.platform.coreNum:
                    n1 = n1t
            n1, m1 = self.balance_workload(n, m, n1, m1, 32)

        if k >= 65536 or m >= 65536:
            m1 = 256
            n1 = 128

        # fixpipe bound
        ratio = (m * k + k * n) / (m * n) if (m * n) > 0 else 0.0
        if ratio < 0.1 and n >= 256:
            m1 = 128
            n1 = 256

        k1 = self.get_max_k1(m1, n1)
        return m1, n1, k1

    def calculate_tiling(self, m: int, n: int, k: int,
                         layout_tag_a: int = LayoutTag.TagRowMajor,
                         layout_tag_b: int = LayoutTag.TagRowMajor) -> Tuple[int, int, int]:
        """
        计算 small matmul 的 tiling 参数

        参数:
            m, n, k: 矩阵维度
            layout_tag_a: A 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: B 矩阵布局 (0=RowMajor, 1=ColumnMajor)
        返回:
            (m1, n1, k1): L1 tile 大小
        """
        # 根据 layout 组合选择对应的 tiling 函数
        layout_key = (layout_tag_a, layout_tag_b)

        if layout_key == (LayoutTag.TagRowMajor, LayoutTag.TagRowMajor):
            return self.do_tiling_b16_layout00(m, n, k)
        elif layout_key == (LayoutTag.TagRowMajor, LayoutTag.TagColumnMajor):
            return self.do_tiling_b16_layout01(m, n, k)
        elif layout_key == (LayoutTag.TagColumnMajor, LayoutTag.TagRowMajor):
            return self.do_tiling_b16_layout10(m, n, k)
        elif layout_key == (LayoutTag.TagColumnMajor, LayoutTag.TagColumnMajor):
            return self.do_tiling_b16_layout11(m, n, k)
        else:
            raise ValueError(f"Invalid layout tags: layout_tag_a={layout_tag_a}, layout_tag_b={layout_tag_b}")

    def get_padding_tags(self, m: int, n: int, k: int, m1: int, n1: int, k1: int,
                         layout_tag_a: int, layout_tag_b: int,
                         splitk_factor: int) -> Tuple[int, int, int]:
        """
        计算 Padding 标签，对应 C++ 的 GetPaddingTag 函数

        参数:
            m, n, k: 矩阵维度
            m1, n1, k1: tiling 参数
            layout_tag_a: A 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: B 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            splitk_factor: SplitK 因子
        返回:
            (padding_tag_a, padding_tag_b, padding_tag_c)
        """
        padding_a, padding_b, padding_c = PaddingCalculator.calc_padding_tags(
            m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b,
            splitk_factor, self.platform.coreNum
        )
        return padding_a.value, padding_b.value, padding_c.value

    def calculate_block_dim(self, m: int, n: int, k: int, m1: int, n1: int, k1: int,
                            layout_tag_a: int, layout_tag_b: int,
                            padding_tag_a: int, padding_tag_b: int,
                            splitk_factor: int) -> int:
        """
        计算 blockDim，对应 C++ 的 GetPaddingTag 函数中的 blockDim 计算逻辑

        参数:
            m, n, k: 矩阵维度
            m1, n1, k1: tiling 参数
            layout_tag_a: A 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: B 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            padding_tag_a: A 矩阵 padding 标签
            padding_tag_b: B 矩阵 padding 标签
            splitk_factor: SplitK 因子
        返回:
            blockDim: 块维度
        """
        tasks_aic = ceil_div(m, m1) * ceil_div(n, n1) * splitk_factor
        block_dim_aic = min(tasks_aic, self.platform.coreNum)

        # 如果有 padding，需要计算 AIV 的 blockDim
        if padding_tag_a != PaddingTag.PADDING_NONE.value or padding_tag_b != PaddingTag.PADDING_NONE.value:
            # 计算 AIV 任务数
            outter_axis_a = m
            inner_axis_a = k
            if layout_tag_a == LayoutTag.TagColumnMajor:
                outter_axis_a = k
                inner_axis_a = m

            outter_axis_b = k
            inner_axis_b = n
            if layout_tag_b == LayoutTag.TagColumnMajor:
                outter_axis_b = n
                inner_axis_b = k

            def calc_tasks_aiv(outter_axis: int, inner_axis: int) -> int:
                """计算 AIV 任务数"""
                task_rows = min(16, outter_axis)
                task_cols = (48 * 1024) // 2 // task_rows
                if inner_axis < task_cols:
                    task_cols = inner_axis
                tiles_per_axis = max(1, ceil_div(inner_axis, task_cols))
                task_cols = round_up(inner_axis // tiles_per_axis, 16)
                return ceil_div(outter_axis, task_rows) * ceil_div(inner_axis, task_cols)

            actual_tasks_aiv_a = 0
            actual_tasks_aiv_b = 0

            if padding_tag_a != PaddingTag.PADDING_NONE.value and inner_axis_a > 192:
                actual_tasks_aiv_a = calc_tasks_aiv(outter_axis_a, inner_axis_a)

            if padding_tag_b != PaddingTag.PADDING_NONE.value and inner_axis_b > 192:
                actual_tasks_aiv_b = calc_tasks_aiv(outter_axis_b, inner_axis_b)

            actual_tasks_aiv = max(actual_tasks_aiv_a, actual_tasks_aiv_b)
            block_dim_aiv = min(ceil_div(actual_tasks_aiv, 2), self.platform.coreNum)

            return max(block_dim_aic, block_dim_aiv)

        return block_dim_aic

    def get_kernel_serial(self, operator_type: str) -> int:
        """
        根据算子类型获取 kernelSerial

        参数:
            operator_type: 算子类型字符串
        返回:
            kernelSerial: 内核序列号
        """
        kernel_serial_map = {
            OperatorType.SMALL_MATMUL: 1,
            OperatorType.PADDING_COMMON_MATMUL: 2,
            OperatorType.PADDING_MULTICORE_SPLITK_MATMUL: 3,
            OperatorType.PADDING_STREAMK_MATMUL: 4,
            OperatorType.COMMON_MATMUL: 0,
        }
        return kernel_serial_map.get(operator_type, 0)

    def get_dispatch_policy_tag(self, operator_type: str) -> str:
        """
        根据算子类型获取 dispatchPolicyTag

        参数:
            operator_type: 算子类型字符串
        返回:
            dispatchPolicyTag: 调度策略标签
        """
        dispatch_policy_map = {
            OperatorType.SMALL_MATMUL: "DynamicSmall",
            OperatorType.PADDING_STREAMK_MATMUL: "DynamicStreamk",
            OperatorType.PADDING_MULTICORE_SPLITK_MATMUL: "DynamicCommon",
            OperatorType.PADDING_COMMON_MATMUL: "DynamicCommon",
            OperatorType.COMMON_MATMUL: "DynamicCommon",
        }
        return dispatch_policy_map.get(operator_type, "DynamicCommon")

    def select_kernel_type(self, m: int, n: int, k: int, m1: int, n1: int, k1: int,
                           layout_tag_a: int, layout_tag_b: int) -> Tuple[str, int, int, int, int, int, int, int, int, str]:
        """
        根据 shape 和 tiling 参数选择算子类型，完全对应 C++ 的 SelectKernelB16 函数

        参数:
            m, n, k: 矩阵维度
            m1, n1, k1: tiling 参数（可能会被修改）
            layout_tag_a: A 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: B 矩阵布局 (0=RowMajor, 1=ColumnMajor)
        返回:
            (operator_type, final_m1, final_n1, final_k1, splitk_factor, padding_tag_a, padding_tag_b, padding_tag_c, block_dim, dispatch_policy_tag):
            算子类型、最终的 tiling 参数、splitk 因子、padding 标签、blockDim 和 dispatchPolicyTag
        """
        # 临时存储原始 layout（对应 C++ 代码中的 layoutTagATmp, layoutTagBTmp）
        layout_tag_a_tmp = layout_tag_a
        layout_tag_b_tmp = layout_tag_b

        # 当 m=1 或 n=1 时，调整 layout（对应 C++ 代码）
        if m == 1 and layout_tag_a == LayoutTag.TagColumnMajor:
            layout_tag_a = LayoutTag.TagRowMajor
        if n == 1 and layout_tag_b == LayoutTag.TagRowMajor:
            layout_tag_b = LayoutTag.TagColumnMajor

        final_m1, final_n1, final_k1 = m1, n1, k1
        splitk_factor = 1  # 默认值

        # Handler 1: SmallMatmulB16Handler (对应 C++ 代码)
        padding_a, padding_b, padding_c = self.get_padding_tags(
            m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b, 1
        )

        if (padding_a == PaddingTag.PADDING_NONE.value and
            padding_b == PaddingTag.PADDING_NONE.value and
            padding_c == PaddingTag.PADDING_NONE.value):
            task_blocks = ceil_div(m, m1) * ceil_div(n, n1)
            if task_blocks <= self.platform.coreNum and k <= k1:
                block_dim = min(task_blocks, self.platform.coreNum)
                operator_type = OperatorType.SMALL_MATMUL
                kernel_serial = self.get_kernel_serial(operator_type)
                dispatch_policy_tag = self.get_dispatch_policy_tag(operator_type)
                return (operator_type, final_m1, final_n1, final_k1, splitk_factor,
                       padding_a, padding_b, padding_c, block_dim, dispatch_policy_tag)

        # Handler 2: PaddingMultiCoreSplitkMatmulB16Handler (对应 C++ 代码)
        m1t, n1t, k1t = 128, 256, 256
        layout_a = LayoutTag(layout_tag_a)
        layout_b = LayoutTag(layout_tag_b)

        cond1 = (layout_a == LayoutTag.TagColumnMajor and layout_b == LayoutTag.TagColumnMajor)
        cond2 = (layout_a == LayoutTag.TagColumnMajor and layout_b == LayoutTag.TagRowMajor) and (m > n)
        if cond1 or cond2:
            m1t = 256
            n1t = 128

        blocks = ceil_div(m, m1t) * ceil_div(n, n1t)
        # max_splitk_factor = 2
        # if k > 1024:
        #     max_splitk_factor = 4
        # if k > 2048:
        #     max_splitk_factor = 8
        # if k > 4096:
        #     max_splitk_factor = 16
        # if k >= 12288:
        #     max_splitk_factor = self.platform.coreNum
        #
        # if (blocks <= self.platform.coreNum // 2 and k > 5120) or (blocks <= 2 and k > 1024):
        #     # 修改 tiling 参数（对应 C++ 代码）
        #     final_m1 = m1t
        #     final_n1 = n1t
        #     final_k1 = k1t
        #     # 计算 splitkFactor（对应 C++ 代码）
        #     splitk_factor = min(self.platform.coreNum // blocks, max_splitk_factor)
        #     # 重新计算 padding（对应 C++ 代码中的 GetPaddingTag）
        #     padding_a, padding_b, padding_c = self.get_padding_tags(
        #         m, n, k, final_m1, final_n1, final_k1, layout_tag_a, layout_tag_b, splitk_factor
        #     )
        #     # 计算 blockDim（对应 C++ 代码，PaddingMultiCoreSplitk 使用 platform.coreNum）
        #     block_dim = self.platform.coreNum
        #     operator_type = OperatorType.PADDING_MULTICORE_SPLITK_MATMUL
        #     kernel_serial = self.get_kernel_serial(operator_type)
        #     dispatch_policy_tag = self.get_dispatch_policy_tag(operator_type)
        #     return (operator_type, final_m1, final_n1, final_k1, splitk_factor,
        #            padding_a, padding_b, padding_c, block_dim, dispatch_policy_tag)

        # Handler 3: PaddingStreamkMatmulB16Handler (对应 C++ 代码)
        # StreamK 可以选择带宽最优的 tiling 配置，如果默认配置不满足条件，尝试另一种配置
        sk_blocks = blocks % self.platform.coreNum
        streamk_satisfied = False
        if (blocks > self.platform.coreNum and blocks < 8 * self.platform.coreNum and
            sk_blocks > 0 and sk_blocks < int(0.8 * self.platform.coreNum) and k > 3072):
            # 修改 tiling 参数（对应 C++ 代码）
            final_m1 = m1t
            final_n1 = n1t
            final_k1 = k1t
            # 重新计算 padding（对应 C++ 代码中的 GetPaddingTag）
            padding_a, padding_b, padding_c = self.get_padding_tags(
                m, n, k, final_m1, final_n1, final_k1, layout_tag_a, layout_tag_b, 1
            )
            # 计算 blockDim（对应 C++ 代码，PaddingStreamk 使用 platform.coreNum）
            block_dim = self.platform.coreNum
            operator_type = OperatorType.PADDING_STREAMK_MATMUL
            kernel_serial = self.get_kernel_serial(operator_type)
            dispatch_policy_tag = self.get_dispatch_policy_tag(operator_type)
            return (operator_type, final_m1, final_n1, final_k1, splitk_factor,
                   padding_a, padding_b, padding_c, block_dim, dispatch_policy_tag)

        # 如果默认配置不满足条件，尝试另一种配置（交换 m1t 和 n1t）
        # 这对应 C++ 代码中 StreamK 可以选择带宽最优配置的逻辑
        # 对于 layout (RowMajor, ColumnMajor) 或 (ColumnMajor, RowMajor) 且不满足 cond1/cond2 的情况
        if not (cond1 or cond2):
            m1t_alt, n1t_alt = 256, 128
            blocks_alt = ceil_div(m, m1t_alt) * ceil_div(n, n1t_alt)
            sk_blocks_alt = blocks_alt % self.platform.coreNum
            if (blocks_alt > self.platform.coreNum and blocks_alt < 8 * self.platform.coreNum and
                sk_blocks_alt > 0 and sk_blocks_alt < int(0.8 * self.platform.coreNum) and k > 3072):
                # 修改 tiling 参数，但最终可能走到 CommonMatmul（如果 padding 不满足）
                final_m1 = m1t_alt
                final_n1 = n1t_alt
                final_k1 = k1t
                # 注意：这里不返回，继续检查后续 handler，但 tiling 已经被修改

        # Handler 4: PaddingCommonMatmulB16Handler (对应 C++ 代码)
        # 使用当前的 tiling 参数（可能已被前面的 handler 修改）重新计算 padding
        padding_a, padding_b, padding_c = self.get_padding_tags(
            m, n, k, final_m1, final_n1, final_k1, layout_tag_a, layout_tag_b, splitk_factor
        )
        if (padding_a != PaddingTag.PADDING_NONE.value or
            padding_b != PaddingTag.PADDING_NONE.value or
            padding_c != PaddingTag.PADDING_NONE.value):
            # 计算 blockDim（对应 C++ 代码中的 GetPaddingTag）
            block_dim = self.calculate_block_dim(
                m, n, k, final_m1, final_n1, final_k1, layout_tag_a, layout_tag_b,
                padding_a, padding_b, splitk_factor
            )
            operator_type = OperatorType.PADDING_COMMON_MATMUL
            kernel_serial = self.get_kernel_serial(operator_type)
            dispatch_policy_tag = self.get_dispatch_policy_tag(operator_type)
            return (operator_type, final_m1, final_n1, final_k1, splitk_factor,
                   padding_a, padding_b, padding_c, block_dim, dispatch_policy_tag)

        # Handler 5: CommonMatmulB16Handler (对应 C++ 代码，兜底)
        task_blocks = ceil_div(m, final_m1) * ceil_div(n, final_n1)
        block_dim = min(task_blocks, self.platform.coreNum)
        operator_type = OperatorType.COMMON_MATMUL
        kernel_serial = self.get_kernel_serial(operator_type)
        dispatch_policy_tag = self.get_dispatch_policy_tag(operator_type)
        return (operator_type, final_m1, final_n1, final_k1, splitk_factor,
               padding_a, padding_b, padding_c, block_dim, dispatch_policy_tag)

    def calculate(self, m: int, n: int, k: int,
                  layout_tag_a: int = LayoutTag.TagRowMajor,
                  layout_tag_b: int = LayoutTag.TagRowMajor) -> Dict[str, Any]:
        """
        主要接口：计算 tiling 和算子类型

        参数:
            m, n, k: 矩阵维度
            layout_tag_a: A 矩阵布局 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: B 矩阵布局 (0=RowMajor, 1=ColumnMajor)
        返回:
            字典，包含:
            - tiling: (m1, n1, k1) tiling 参数
            - operator_type: 算子类型字符串
            - shape: (m, n, k) 原始 shape
            - layout: (layout_tag_a, layout_tag_b) layout 信息
            - splitkFactor: SplitK 因子
            - paddingTagA/B/C: padding 标签
            - blockDim: block 维度
            - kernelSerial: 内核序列号
            - dispatchPolicyTag: 调度策略标签
        """
        # 计算 tiling
        m1, n1, k1 = self.calculate_tiling(m, n, k, layout_tag_a, layout_tag_b)

        # 选择算子类型（可能会修改 tiling 参数）
        (operator_type, final_m1, final_n1, final_k1, splitk_factor,
         padding_tag_a, padding_tag_b, padding_tag_c, block_dim, dispatch_policy_tag) = self.select_kernel_type(
            m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b
        )

        # 获取 kernelSerial
        kernel_serial = self.get_kernel_serial(operator_type)

        return {
            "tiling": (final_m1, final_n1, final_k1),
            "operator_type": operator_type,
            "shape": (m, n, k),
            "layout": (layout_tag_a, layout_tag_b),
            "original_m1": final_m1,
            "original_n1": final_n1,
            "original_k1": final_k1,
            "splitkFactor": splitk_factor,
            "paddingTagA": padding_tag_a,
            "paddingTagB": padding_tag_b,
            "paddingTagC": padding_tag_c,
            "blockDim": block_dim,
            "kernelSerial": kernel_serial,
            "dispatchPolicyTag": dispatch_policy_tag,
        }

    def calculate_str(self, m: int, n: int, k: int,
                      layout_a: str = "RowMajor",
                      layout_b: str = "RowMajor") -> Dict[str, Any]:
        """
        使用字符串指定 layout 的计算函数

        参数:
            m, n, k: 矩阵维度
            layout_a: A 矩阵布局 ("RowMajor" 或 "ColumnMajor")
            layout_b: B 矩阵布局 ("RowMajor" 或 "ColumnMajor")
        返回:
            字典，包含所有计算参数
        """
        layout_tag_a = self.layout_map.get(layout_a, LayoutTag.TagRowMajor)
        layout_tag_b = self.layout_map.get(layout_b, LayoutTag.TagRowMajor)

        result = self.calculate(m, n, k, layout_tag_a, layout_tag_b)
        result["layout"] = (layout_a, layout_b)  # 使用字符串格式
        return result

    def set_platform_info(self, platform_info: PlatformInfo):
        """
        设置平台信息

        参数:
            platform_info: 新的平台信息
        """
        self.platform = platform_info

    def get_platform_info(self) -> PlatformInfo:
        """
        获取当前平台信息

        返回:
            当前平台信息
        """
        return self.platform


# ============================================================================
# 使用示例
# ============================================================================

def main():
    calculator = MatmulTilingCalculator()

    result1 = calculator.calculate(m=8192, n=8192, k=8192)
    print("示例 1 - (8192, 8192, 8192), RowMajor, RowMajor:")
    print(f"  Tiling: {result1['tiling']}")
    print(f"  算子类型: {result1['operator_type']}")
    print(f"  blockDim: {result1['blockDim']}")
    for key, value in result1.items():
        print(f"  {key}: {value}")
    print()

    result2 = calculator.calculate_str(
        m=1472, n=64, k=320,
        layout_a="RowMajor",
        layout_b="ColumnMajor"
    )
    print("示例 2 - (1472, 64, 320), RowMajor, ColumnMajor:")
    print(f"  Tiling: {result2['tiling']}")
    print(f"  算子类型: {result2['operator_type']}")
    print(f"  splitkFactor: {result2['splitkFactor']}")
    for key, value in result2.items():
        print(f"  {key}: {value}")
    print()

    custom_platform = PlatformInfo()
    custom_platform.coreNum = 32
    calculator.set_platform_info(custom_platform)

    result3 = calculator.calculate(m=82014, n=1294, k=1294)
    print("示例 3 - 自定义平台 (32 cores):")
    print(f"  Tiling: {result3['tiling']}")
    print(f"  算子类型: {result3['operator_type']}")
    print(f"  平台核心数: {calculator.get_platform_info().coreNum}")
    for key, value in result3.items():
        print(f"  {key}: {value}")
    return result1, result2, result3


if __name__ == "__main__":
    main()