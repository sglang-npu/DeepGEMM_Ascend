"""
Padding计算模块

提供Padding标签计算和带宽估算功能
"""

from enum import IntEnum
from typing import Tuple
from .utils.common import ceil_div, round_up


class PaddingTag(IntEnum):
    """Padding标签枚举"""
    PADDING_NONE = 0
    PADDING_ND = 1
    PADDING_BLOCK_ND = 2
    PADDING_NZ = 3


class PaddingCalculator:
    """Padding计算器类"""
    
    @staticmethod
    def estimate_bandwidth(n_value: int, d_value: int, src_d_value: int) -> float:
        """
        估算带宽
        
        Args:
            n_value: N值
            d_value: D值
            src_d_value: 源D值
        
        Returns:
            估算的带宽值
        """
        a6 = 0.000000000000020146121020
        a5 = -0.000000000012456944162142
        a4 = -0.000000006738536427145036
        a3 = 0.000007301215580838747961
        a2 = -0.002146456956750821074703
        a1 = 0.312849910814454512664184
        a0 = 0.1

        unalign_band = (
            a6 * (d_value ** 6)
            + a5 * (d_value ** 5)
            + a4 * (d_value ** 4)
            + a3 * (d_value ** 3)
            + a2 * (d_value ** 2)
            + a1 * d_value
            + a0
        )

        if d_value == src_d_value and d_value <= 128 and d_value % 16 == 0:
            unalign_band = 60
        if src_d_value >= 65536:
            unalign_band = 1

        if src_d_value % 256 == 0:
            unalign_band = 100 / 30 * unalign_band
        elif src_d_value % 128 == 0:
            unalign_band = 80 / 30 * unalign_band
        elif src_d_value % 64 == 0:
            unalign_band = 50 / 30 * unalign_band
        elif src_d_value % 16 == 0:
            unalign_band = 40 / 30 * unalign_band

        unalign_band = min(unalign_band, 80.0)

        if d_value % 256 == 0 and n_value < 16:
            b2 = -0.003332381309698882569659
            b1 = 0.113578920178116271610946
            b0 = 0.016102868630357251855667
            unalign_band *= b2 * (n_value ** 2) + b1 * n_value + b0
        elif d_value % 32 == 0 and n_value < 32:
            b2 = -0.000298086120946179481978
            b1 = 0.045309519479127147167929
            b0 = 0.035130178145161221336945
            unalign_band *= b2 * (n_value ** 2) + b1 * n_value + b0
        elif n_value < 64:
            b3 = 0.000001809180573350345869
            b2 = -0.000469676727179688081274
            b1 = 0.038963259596073690493867
            b0 = 0.003942641759904389614499
            unalign_band *= (
                b3 * (n_value ** 3)
                + b2 * (n_value ** 2)
                + b1 * n_value
                + b0
            )
        return unalign_band
    
    @staticmethod
    def calc_padding_tags(
        m: int,
        n: int,
        k: int,
        m1: int,
        n1: int,
        k1: int,
        layout_tag_a: int,
        layout_tag_b: int,
        splitk_factor: int,
        core_num: int,
    ) -> Tuple[PaddingTag, PaddingTag, PaddingTag]:
        """
        计算Padding标签
        
        Args:
            m, n, k: 矩阵维度
            m1, n1, k1: Tiling参数（实际值）
            layout_tag_a: Layout A标签 (0=RowMajor, 1=ColumnMajor)
            layout_tag_b: Layout B标签 (0=RowMajor, 1=ColumnMajor)
            splitk_factor: SplitK因子
            core_num: AI Core数量
        
        Returns:
            (padding_tag_a, padding_tag_b, padding_tag_c)
        """
        splitk_factor = max(1, splitk_factor)
        core_num = max(1, core_num)

        outter_axis_a = m
        inner_axis_a = k
        n_value_a = min(m, m1)
        d_value_a = min(k, k1)
        if layout_tag_a == 1:
            outter_axis_a = k
            inner_axis_a = m
            n_value_a = min(k, k1)
            d_value_a = min(m, m1)

        outter_axis_b = k
        inner_axis_b = n
        n_value_b = min(k, k1)
        d_value_b = min(n, n1)
        if layout_tag_b == 1:
            outter_axis_b = n
            inner_axis_b = k
            n_value_b = min(n, n1)
            d_value_b = min(k, k1)

        matrix_a_size = m * k * 2
        a_bandwidth_aiv = 10 if matrix_a_size > 192 * 1024 * 1024 else 30
        a_bandwidth_before_padding = PaddingCalculator.estimate_bandwidth(n_value_a, d_value_a, inner_axis_a)

        tasks_aic = ceil_div(m, m1) * ceil_div(n, n1) * splitk_factor
        block_dim_aic = min(tasks_aic, core_num)
        if (
            ceil_div(m, m1) < block_dim_aic // 2
            and k <= k1
            and ceil_div(m, m1) <= 2
        ):
            ratio = block_dim_aic / ceil_div(m, m1)
            if ratio != 0:
                a_bandwidth_before_padding = (
                    a_bandwidth_before_padding / ratio * 1.5
                )

        a_bandwidth_after_padding = 80.0
        if n_value_a < 16:
            a_bandwidth_after_padding *= n_value_a / 16.0

        matrix_b_size = k * n * 2
        b_bandwidth_aiv = 10 if matrix_b_size > 192 * 1024 * 1024 else 30
        b_bandwidth_before_padding = PaddingCalculator.estimate_bandwidth(n_value_b, d_value_b, inner_axis_b)
        if (
            ceil_div(n, n1) < block_dim_aic // 2
            and k <= k1
            and ceil_div(n, n1) <= 2
        ):
            ratio = block_dim_aic / ceil_div(n, n1)
            if ratio != 0:
                b_bandwidth_before_padding = (
                    b_bandwidth_before_padding / ratio * 1.5
                )

        b_bandwidth_after_padding = 80.0
        if n_value_b < 16:
            b_bandwidth_after_padding *= n_value_b / 16.0

        actual_m = min(m, m1)
        actual_n = min(n, n1)
        round_max = ceil_div(
            ceil_div(m, m1) * ceil_div(n, n1) * splitk_factor,
            core_num,
        )
        a_max_data_size_aic = (
            round_max * actual_m * ceil_div(k, splitk_factor) * 2
        )
        b_max_data_size_aic = (
            round_max * actual_n * ceil_div(k, splitk_factor) * 2
        )

        def calc_padding_simulator(outter_axis: int, inner_axis: int) -> int:
            outter_axis = max(1, outter_axis)
            inner_axis = max(1, inner_axis)

            task_rows = min(16, outter_axis)
            task_cols = (48 * 1024) // 2 // task_rows
            if inner_axis < task_cols:
                task_cols = inner_axis

            tiles_per_axis = max(1, ceil_div(inner_axis, task_cols))
            task_cols = max(16, round_up(inner_axis // tiles_per_axis, 16))

            tasks_aiv = ceil_div(outter_axis, task_rows) * ceil_div(inner_axis, task_cols)
            max_tasks_per_core = ceil_div(tasks_aiv, core_num * 2)
            return max_tasks_per_core * task_cols * task_rows * 2

        a_max_data_size_aiv = calc_padding_simulator(outter_axis_a, inner_axis_a)
        b_max_data_size_aiv = calc_padding_simulator(outter_axis_b, inner_axis_b)

        head_cost = 1 + 7 * (block_dim_aic / core_num)
        if splitk_factor > 1:
            head_cost = 1

        t00 = (
            a_max_data_size_aic / a_bandwidth_before_padding / 1000
            + b_max_data_size_aic / b_bandwidth_before_padding / 1000
        )
        t01 = (
            a_max_data_size_aic / a_bandwidth_before_padding / 1000
            + b_max_data_size_aic / b_bandwidth_after_padding / 1000
            + b_max_data_size_aiv / b_bandwidth_aiv / 1000
            + head_cost
        )
        t10 = (
            a_max_data_size_aic / a_bandwidth_after_padding / 1000
            + b_max_data_size_aic / b_bandwidth_before_padding / 1000
            + a_max_data_size_aiv / a_bandwidth_aiv / 1000
            + head_cost
        )
        t11 = (
            a_max_data_size_aic / a_bandwidth_after_padding / 1000
            + b_max_data_size_aic / b_bandwidth_after_padding / 1000
            + a_max_data_size_aiv / a_bandwidth_aiv / 1000
            + b_max_data_size_aiv / b_bandwidth_aiv / 1000
            + head_cost
            + 2
        )

        min_cost = float("inf")
        padding_tag_a = PaddingTag.PADDING_NONE
        padding_tag_b = PaddingTag.PADDING_NONE

        if min_cost > t00:
            min_cost = t00
        if min_cost > t01:
            min_cost = t01
            padding_tag_a = PaddingTag.PADDING_NONE
            padding_tag_b = PaddingTag.PADDING_NZ
        if min_cost > t10:
            min_cost = t10
            padding_tag_a = PaddingTag.PADDING_NZ
            padding_tag_b = PaddingTag.PADDING_NONE
        if min_cost > t11:
            min_cost = t11
            padding_tag_a = PaddingTag.PADDING_NZ
            padding_tag_b = PaddingTag.PADDING_NZ

        if (
            (inner_axis_a < 8 or (inner_axis_a < 32 and inner_axis_a % 16 != 0))
            and outter_axis_a > 512
        ):
            padding_tag_a = PaddingTag.PADDING_NZ
        if (
            (inner_axis_b < 8 or (inner_axis_b < 32 and inner_axis_b % 16 != 0))
            and outter_axis_b > 512
        ):
            padding_tag_b = PaddingTag.PADDING_NZ

        padding_tag_c = PaddingTag.PADDING_NONE
        if m * n > 2048 * 2048 and n > 256 and n % 128 != 0:
            total_data_size = m * k * ceil_div(n, n1) * 2 + k * n * ceil_div(m, m1) * 2 + m * n * 2
            if total_data_size < 192 * 1024 * 1024:
                padding_tag_c = PaddingTag.PADDING_ND

        return padding_tag_a, padding_tag_b, padding_tag_c


# 向后兼容的函数接口
def estimate_bandwidth(n_value: int, d_value: int, src_d_value: int) -> float:
    """向后兼容的函数接口"""
    return PaddingCalculator.estimate_bandwidth(n_value, d_value, src_d_value)


def calc_padding_tags(
    m: int,
    n: int,
    k: int,
    m1: int,
    n1: int,
    k1: int,
    layout_tag_a: int,
    layout_tag_b: int,
    splitk_factor: int,
    core_num: int,
) -> Tuple[PaddingTag, PaddingTag, PaddingTag]:
    """向后兼容的函数接口"""
    return PaddingCalculator.calc_padding_tags(
        m, n, k, m1, n1, k1, layout_tag_a, layout_tag_b, splitk_factor, core_num
    )

