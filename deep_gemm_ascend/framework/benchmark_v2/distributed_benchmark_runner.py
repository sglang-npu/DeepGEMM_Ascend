"""
基准测试运行器模块

提供基准测试流程协调功能
"""

import os
import math
import csv
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from tqdm import tqdm

from .catlass_parameter import CatlassParameter
from .models import CatlassResult
from .file_io import CheckpointManager, ResultWriter
from .utils.msprof_executor import MsProfExecutor
from .utils.result_parse import ResultParse
from .utils.logger import logger


class GEMMBenchmarkRunner:
    """GEMM基准测试运行器（协调者模式）"""
    
    def __init__(
        self, 
        shape_group: List[List[int]], 
        rank_id: int,
        npu_ids: List[int],
        num_processes: int, 
        catlass_bin_path: str, 
        result_dir: str = "./results", 
        msp_dir: str = "./msp", 
        operator_type: str = None, 
        core_num: int = 20,
        layout_tag_a: int = None,
        layout_tag_b: int = None
    ):
        """
        初始化基准测试运行器
        
        Args:
            shape_group: Shape列表，每个元素为[M, N, K]
            rank_id: 进程编号（从0开始，用于shape分配计算和获取对应的NPU ID）
            npu_ids: NPU设备ID列表，rank_id对应的NPU ID为npu_ids[rank_id]
            num_processes: 总进程数（用于shape分配计算）
            catlass_bin_path: Catlass可执行文件路径
            result_dir: 结果保存目录
            msp_dir: msprof输出目录
            operator_type: 算子类型，必须提供
            core_num: AI Core数量，默认20
            layout_tag_a: Layout A标签，必须提供，0=RowMajor, 1=ColumnMajor
            layout_tag_b: Layout B标签，必须提供，0=RowMajor, 1=ColumnMajor
        
        Raises:
            ValueError: 如果 operator_type、layout_tag_a 或 layout_tag_b 未提供
        """
        if operator_type is None:
            raise ValueError("operator_type 必须提供，不能为 None")
        if layout_tag_a is None:
            raise ValueError("layout_tag_a 必须提供，不能为 None")
        if layout_tag_b is None:
            raise ValueError("layout_tag_b 必须提供，不能为 None")
        self.shape_group = shape_group
        self.result_dir = result_dir
        self.rank_id = rank_id
        self.npu_ids = npu_ids
        self.num_processes = num_processes
        self.operator_type = operator_type
        self.layout_tag_a = layout_tag_a
        self.layout_tag_b = layout_tag_b
        self.checkpoint_interval = 10
        
        # 从npu_ids列表获取当前进程对应的NPU ID
        self.npu_id = npu_ids[rank_id]
        
        # 依赖注入：创建各个组件
        self.parameters = CatlassParameter(
            operator_type=operator_type, 
            core_num=core_num,
            layout_tag_a=layout_tag_a,
            layout_tag_b=layout_tag_b
        )
        
        # 为每个进程创建独立的msp目录，避免多进程冲突
        self.rank_msp_dir = os.path.join(msp_dir, f"npu_{self.npu_id}")
        os.makedirs(self.rank_msp_dir, exist_ok=True)
        
        # 保存二进制路径，用于构建program命令
        self.catlass_bin_path = catlass_bin_path
        
        # 初始化MsProfExecutor，使用正确的参数
        self.msprof_executor = MsProfExecutor(
            output=self.rank_msp_dir,
            aic_metrics="PipeUtilization",
            kernel_name="_Z",  # 默认kernel名称过滤
            launch_count=0,  # 不限制launch次数
            timeout=15  # 15秒超时
        )
        
        # 初始化ResultParse，根据operator_type和layout确定kernel_list
        kernel_list = self._get_kernel_list(operator_type, layout_tag_a, layout_tag_b)
        self.result_parser = ResultParse(kernel_list=kernel_list, device_id=self.npu_id)
        
        # 缓存shape字符串，避免重复计算
        self.shape_str_cache = {}
    
    def _clear_msp_dir(self) -> None:
        """
        清空msprof输出目录
        
        删除目录下所有文件和子目录，但保留目录本身
        """
        if not os.path.exists(self.rank_msp_dir):
            return
        
        try:
            # 删除目录下所有内容
            for item in os.listdir(self.rank_msp_dir):
                item_path = os.path.join(self.rank_msp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            logger.debug(f"Rank {self.rank_id} (NPU {self.npu_id}) 已清空msp目录: {self.rank_msp_dir}")
        except Exception as e:
            logger.warning(f"Rank {self.rank_id} (NPU {self.npu_id}) 清空msp目录失败: {e}")
    
    def _get_kernel_list(self, operator_type: str, layout_tag_a: int, layout_tag_b: int) -> List[str]:
        """
        根据operator_type和layout确定需要解析的kernel列表
        
        Args:
            operator_type: 算子类型，必须提供，可选值: 
                'SmallMatmulKernel', 'CommonMatmulKernel', 
                'PaddingCommonMatmulKernel', 
                'PaddingMultiCoreSplitkMatmulKernel',
                'PaddingStreamkMatmulKernel'
            layout_tag_a: Layout A标签，0=RowMajor, 1=ColumnMajor
            layout_tag_b: Layout B标签，0=RowMajor, 1=ColumnMajor
            
        Returns:
            kernel名称列表
        """
        # 根据layout组合确定kernel名称后缀
        layout_suffix = f"Layout{layout_tag_a}{layout_tag_b}"
        
        if operator_type == 'SmallMatmulKernel':
            return [f"SmallMatmulKernelHalf{layout_suffix}"]
        elif operator_type == 'CommonMatmulKernel':
            return [f"CommonMatmulKernelHalf{layout_suffix}"]
        elif operator_type == 'PaddingCommonMatmulKernel':
            return [
                f"PaddingMatmulKernelHalf{layout_suffix}Padding001",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding030",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding300",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding031",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding301",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding330",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding331",
            ]
        elif operator_type == 'PaddingMultiCoreSplitkMatmulKernel':
            return [
                f"PaddingMatmulKernelHalf{layout_suffix}Padding030",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding300",
                f"PaddingMatmulKernelHalf{layout_suffix}Padding330",
            ]
        elif operator_type == 'PaddingStreamkMatmulKernel':
            return [f"PaddingStreamkMatmulKernelHalf{layout_suffix}"]
        else:
            raise ValueError(f"未知的算子类型: {operator_type}")
    
    def generate_tiling_csv(self, shape: List[int], filter_params: List[Dict[str, int]], csv_path: str) -> None:
        """
        生成包含tiling参数的CSV文件
        
        Args:
            shape: [M, N, K] 矩阵维度
            filter_params: tiling参数列表，每个元素包含 mTile, nTile, kTile
            csv_path: CSV文件保存路径
        """
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头：添加layoutA和layoutB列，保留None列
            writer.writerow(['M', 'N', 'K', 'mTile', 'nTile', 'kTile', 'layoutA', 'layoutB', 'None'])
            # 写入数据行
            for params in filter_params:
                writer.writerow([
                    shape[0],  # M
                    shape[1],  # N
                    shape[2],  # K
                    params['mTile'],
                    params['nTile'],
                    params['kTile'],
                    self.layout_tag_a,  # layoutA
                    self.layout_tag_b,  # layoutB
                    'None'
                ])
    
    def _is_result_abnormal(self, output: str, expected_count: int) -> bool:
        """
        检测结果是否异常
        
        Args:
            output: msprof_executor.process() 返回的输出字符串
            expected_count: 期望的结果数量
            
        Returns:
            True表示异常，False表示正常
        """
        # 检查1: 返回空字符串
        if not output or output.strip() == "":
            logger.error(f"Rank {self.rank_id} (NPU {self.npu_id}) msprof输出为空字符串，期望结果数量: {expected_count}")
            return True
        
        # 检查2: 包含stderr（通过检查是否包含常见错误关键词）
        # 注意：msprof_executor会将stderr合并到输出中，我们需要检查是否有错误信息
        error_indicators = [
            "error", "Error", "ERROR",
            "failed", "Failed", "FAILED",
            "exception", "Exception", "EXCEPTION"
        ]
        # 如果输出中包含错误关键词，可能是异常
        # 但需要小心，因为某些正常输出也可能包含这些词
        # 更可靠的方法是检查返回码，但process方法已经合并了输出
        # 这里我们主要依赖空字符串和结果数量检查
        
        # 检查3: 尝试解析结果数量
        try:
            # 使用ResultParse解析结果，检查数量是否匹配
            parsed_results = self.result_parser.parse_multi_result(output, expected_count)
            if parsed_results is None:
                logger.error(f"Rank {self.rank_id} (NPU {self.npu_id}) 解析结果返回None，期望结果数量: {expected_count}")
                return True
            if len(parsed_results) != expected_count:
                logger.error(f"Rank {self.rank_id} (NPU {self.npu_id}) 解析结果数量不匹配: 期望 {expected_count}，实际 {len(parsed_results)}")
                return True
        except Exception as e:
            # 解析失败也视为异常
            logger.error(f"Rank {self.rank_id} (NPU {self.npu_id}) 解析结果时抛出异常: {e}，期望结果数量: {expected_count}")
            return True
        
        return False
    
    def _execute_with_span_reduction(
        self, 
        csv_path: str, 
        start_idx: int, 
        end_idx: int, 
        span: int,
        shape: Optional[List[int]] = None
    ) -> List[Tuple]:
        """
        递归执行批量任务，如果异常则缩小跨度重试
        
        Args:
            csv_path: CSV文件路径
            start_idx: 起始索引（包含）
            end_idx: 终止索引（不包含）
            span: 当前跨度
            
        Returns:
            解析后的结果列表，格式为 [(kernel_name, accuracy, duration, block_dim, mix_block_dim), ...]
            如果跨度为1仍异常，返回包含-1的结果列表，格式为 [("", -1, -1, -1, -1), ...]
        """
        # 计算期望的结果数量
        expected_count = end_idx - start_idx
        if expected_count <= 0:
            logger.error(
                "Rank %s (NPU %s) 发现无效的区间 [%s, %s)，expected_count=%s",
                self.rank_id,
                self.npu_id,
                start_idx,
                end_idx,
                expected_count,
            )
            return []
        range_len = expected_count

        def _split_mid() -> Optional[int]:
            """根据当前区间计算安全的拆分点，确保 start < mid < end。"""
            if range_len <= 1:
                return None
            split_step = max(1, range_len // 2)
            mid = start_idx + split_step
            if mid >= end_idx:
                mid = end_idx - 1
            if mid <= start_idx:
                mid = start_idx + 1
            if mid <= start_idx or mid >= end_idx:
                return None
            return mid

        program = f"{self.catlass_bin_path} {self.npu_id} 2 {csv_path} {start_idx} {end_idx}"
        
        # 根据 span 动态设置 launch-count
        effective_span = span if span > 0 else expected_count
        effective_span = max(1, min(effective_span, expected_count))
        
        # 根据计算量（M * N * K）和span动态计算超时时间
        # 基于规则的档位设置，参考：[1000,1000,1000] 单条超时时间为3秒
        if shape is not None and len(shape) >= 3:
            m, n, k = shape[0], shape[1], shape[2]
            computation_size = m * n * k
            
            # 根据计算量分档，设置单条超时时间
            if computation_size < 1e8:  # < 100M
                base_timeout_per_item = 2  # 小shape：2秒/条
            elif computation_size < 1e9:  # < 1B
                base_timeout_per_item = 3  # 中shape：3秒/条（参考值：[1000,1000,1000]）
            elif computation_size < 1e10:  # < 10B
                base_timeout_per_item = 5  # 大shape：5秒/条
            elif computation_size < 1e11:  # < 100B
                base_timeout_per_item = 10  # 超大shape：10秒/条
            elif computation_size < 1e12:  # < 1T
                base_timeout_per_item = 20  # 极大shape：20秒/条
            else:  # >= 1T
                base_timeout_per_item = 30  # 特大shape：30秒/条
            
            # 批次超时时间 = 单条超时时间 * 批次大小 * 批次因子
            # 批次因子考虑批次执行的开销，默认1.2（即增加20%的缓冲）
            batch_factor = 1.2
            dynamic_timeout = max(5, int(base_timeout_per_item * effective_span * batch_factor))
            
            logger.debug(f"Shape {shape}, computation_size={computation_size:.2e}, span={effective_span}, "
                        f"base_timeout_per_item={base_timeout_per_item}s, timeout={dynamic_timeout}s")
        else:
            # 如果没有shape信息，回退到原来的计算方式
            dynamic_timeout = max(1, 6 * effective_span)

        # 执行msprof命令
        output = self.msprof_executor.process(
            program,
            launch_count=effective_span,
            timeout=dynamic_timeout
        )
        
        # 检查结果是否异常
        if self._is_result_abnormal(output, expected_count):
            # 如果跨度为1，无法再缩小，返回包含-1的结果列表
            if span <= 1 or range_len <= 1:
                error_msg = f"Rank {self.rank_id} (NPU {self.npu_id}) 跨度为1仍异常，为索引范围 [{start_idx}, {end_idx}) 返回-1标记"
                logger.error(error_msg)
                # 返回包含-1的结果列表，参考result_parse.py的格式: (kernel_name, accuracy, duration, block_dim, mix_block_dim)
                return [("", -1, -1, -1, -1) for _ in range(expected_count)]
            
            # 跨度减半，递归处理。需要确保拆分点位于 (start_idx, end_idx) 之间
            new_span = max(1, span // 2)
            mid_idx = _split_mid()
            if mid_idx is None:
                error_msg = f"Rank {self.rank_id} (NPU {self.npu_id}) 无法继续拆分区间 [{start_idx}, {end_idx})，返回-1标记"
                logger.error(error_msg)
                return [("", -1, -1, -1, -1) for _ in range(expected_count)]
            
            # 递归处理前半部分（传入new_span以继续缩小）
            results_part1 = self._execute_with_span_reduction(csv_path, start_idx, mid_idx, new_span, shape)
            # 递归处理后半部分（传入new_span以继续缩小）
            results_part2 = self._execute_with_span_reduction(csv_path, mid_idx, end_idx, new_span, shape)
            
            # 合并结果（现在两部分都不会是None，都会返回列表）
            return results_part1 + results_part2
        
        # 结果正常，解析并返回
        try:
            parsed_results = self.result_parser.parse_multi_result(output, expected_count)
            if parsed_results is None:
                # 如果解析返回None，且跨度为1，返回-1标记
                if span <= 1 or range_len <= 1:
                    error_msg = f"Rank {self.rank_id} (NPU {self.npu_id}) 解析返回None，跨度为1，为索引范围 [{start_idx}, {end_idx}) 返回-1标记"
                    logger.error(error_msg)
                    return [("", -1, -1, -1, -1) for _ in range(expected_count)]
                else:
                    # 递归缩小
                    new_span = max(1, span // 2)
                    mid_idx = _split_mid()
                    if mid_idx is None:
                        error_msg = f"Rank {self.rank_id} (NPU {self.npu_id}) 解析返回None但无法拆分区间 [{start_idx}, {end_idx})，返回-1标记"
                        logger.error(error_msg)
                        return [("", -1, -1, -1, -1) for _ in range(expected_count)]
                    results_part1 = self._execute_with_span_reduction(csv_path, start_idx, mid_idx, new_span, shape)
                    results_part2 = self._execute_with_span_reduction(csv_path, mid_idx, end_idx, new_span, shape)
                    return results_part1 + results_part2
            return parsed_results
        except Exception as e:
            error_msg = f"Rank {self.rank_id} (NPU {self.npu_id}) 解析结果失败: {e}"
            logger.error(error_msg)
            # 解析失败，如果跨度为1则返回-1标记，否则递归缩小
            if span <= 1 or range_len <= 1:
                error_msg2 = f"Rank {self.rank_id} (NPU {self.npu_id}) 解析异常，跨度为1，为索引范围 [{start_idx}, {end_idx}) 返回-1标记"
                print(error_msg2)
                logger.error(error_msg2)
                return [("", -1, -1, -1, -1) for _ in range(expected_count)]
            else:
                new_span = max(1, span // 2)
                mid_idx = _split_mid()
                if mid_idx is None:
                    error_msg2 = f"Rank {self.rank_id} (NPU {self.npu_id}) 解析异常且无法拆分区间 [{start_idx}, {end_idx})，返回-1标记"
                    logger.error(error_msg2)
                    return [("", -1, -1, -1, -1) for _ in range(expected_count)]
                results_part1 = self._execute_with_span_reduction(csv_path, start_idx, mid_idx, new_span, shape)
                results_part2 = self._execute_with_span_reduction(csv_path, mid_idx, end_idx, new_span, shape)
                return results_part1 + results_part2
    
    def benchmark_shape(self, shape: List[int], shape_order: int, total_shapes: int) -> None:
        """
        对指定的shape进行基准测试（批量执行模式）
        
        使用CSV文件批量执行tiling参数组合，初始跨度根据shape大小动态调整：
        - 如果 M * N * K > 10^10，初始跨度为50
        - 否则初始跨度为100
        异常时递归缩小跨度。
        
        Args:
            shape: [M, N, K] 矩阵维度
            shape_order: 当前shape在本进程中的序号（从1开始）
            total_shapes: 本进程需处理的shape总数
        """
        # 确保结果目录存在
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 生成当前进程的文件路径（使用缓存）
        shape_tuple = tuple(shape)
        if shape_tuple not in self.shape_str_cache:
            self.shape_str_cache[shape_tuple] = '_'.join(map(str, shape))
        shape_str = self.shape_str_cache[shape_tuple]
        result_filename = f'shape_{shape_str}_npu_{self.npu_id}.jsonl'
        result_path = str(Path(self.result_dir) / result_filename)
        checkpoint_filename = f'shape_{shape_str}_npu_{self.npu_id}_checkpoint.jsonl'
        checkpoint_path = str(Path(self.result_dir) / checkpoint_filename)
        
        # 获取该shape的所有有效tiling参数组合
        filter_params = self.parameters.filter_parameters(
            shape, 
            layout_tag_a=self.layout_tag_a, 
            layout_tag_b=self.layout_tag_b
        )
        
        # 输出筛选信息
        if self.operator_type:
            print(f"Rank {self.rank_id} (NPU {self.npu_id}): Operator type '{self.operator_type}', "
                  f"filtered {len(filter_params)} valid tiling combinations for shape {shape}")
        
        process_task_count = len(filter_params)
        if process_task_count == 0:
            print(f"Rank {self.rank_id} (NPU {self.npu_id}) Shape {shape} 没有有效的tiling参数组合，跳过")
            return
        
        # 生成CSV文件路径
        tiling_csv_dir = os.path.join(self.result_dir, "tiling_csvs")
        csv_filename = f'shape_{shape_str}.csv'
        csv_path = str(Path(tiling_csv_dir) / csv_filename)
        
        # 生成CSV文件（如果不存在）
        if not os.path.exists(csv_path):
            print(f"Rank {self.rank_id} (NPU {self.npu_id}) 生成CSV文件: {csv_path}")
            self.generate_tiling_csv(shape, filter_params, csv_path)
        
        # 使用CheckpointManager加载断点
        checkpoint_manager = CheckpointManager(checkpoint_path)
        last_process_idx = checkpoint_manager.load()
        if last_process_idx is None:
            last_process_idx = -1
        
        # 计算从checkpoint恢复的起始索引
        start_local_idx = 0
        if last_process_idx >= 0:
            start_local_idx = last_process_idx + 1  # 从下一个索引开始
            if start_local_idx >= process_task_count:
                print(f"Rank {self.rank_id} (NPU {self.npu_id}) 已完成所有任务，无需继续处理")
                return
        
        # 使用ResultWriter管理结果写入
        result_writer = ResultWriter(result_path, buffer_size=10)
        
        completed_count = max(0, start_local_idx)
        progress_desc = f"Rank {self.rank_id} (NPU {self.npu_id}) Shape [{shape_order}/{total_shapes}]"
        
        # 根据shape大小动态调整初始跨度，避免大shape时NPU显存占满导致超时
        # 基于计算量的档位设置，计算量越大，初始跨度越小
        m, n, k = shape[0], shape[1], shape[2]
        computation_size = m * n * k
        
        # 根据计算量分档，设置初始跨度
        if computation_size < 1e8:  # < 100M
            initial_span = 200  # 小shape：200条/批
        elif computation_size < 1e9:  # < 1B
            initial_span = 100  # 中shape：100条/批
        elif computation_size < 1e10:  # < 10B
            initial_span = 50  # 大shape：50条/批
        elif computation_size < 1e11:  # < 100B
            initial_span = 20  # 超大shape：20条/批
        elif computation_size < 1e12:  # < 1T
            initial_span = 10  # 极大shape：10条/批
        else:  # >= 1T
            initial_span = 5  # 特大shape：5条/批
        
        logger.debug(f"Shape {shape}, computation_size={computation_size:.2e}, initial_span={initial_span}")
        
        with tqdm(
            total=process_task_count,
            initial=completed_count,
            desc=f"{progress_desc} {shape}",
            postfix={"Processed": completed_count},
        ) as pbar:
            # 批量执行：按初始跨度分批处理（根据shape大小动态调整）
            current_idx = start_local_idx
            while current_idx < process_task_count:
                # 计算当前批次的结束索引（左闭右开）
                end_idx = min(current_idx + initial_span, process_task_count)
                
                # 更新checkpoint（每批次更新一次）
                checkpoint_manager.save(current_idx)
                
                # 执行批量任务（递归缩小跨度）
                batch_results = self._execute_with_span_reduction(
                    csv_path, 
                    current_idx, 
                    end_idx, 
                    initial_span,
                    shape
                )
                
                # 处理批量结果（batch_results现在总是返回列表，不会为None）
                # 将解析结果映射回对应的tiling参数并保存
                for result_idx, parsed_result in enumerate(batch_results):
                    actual_idx = current_idx + result_idx
                    if actual_idx >= process_task_count:
                        break
                    
                    # parsed_result格式: (kernel_name, accuracy, duration, block_dim, mix_block_dim)
                    kernel_name, accuracy, duration, block_dim, mix_block_dim = parsed_result
                    
                    # 获取对应的tiling参数
                    if actual_idx < len(filter_params):
                        parameters = filter_params[actual_idx]
                        
                        # 检查是否为异常标记（duration == -1表示异常）
                        if duration == -1:
                            # 创建失败标记
                            error_msg = f"Rank {self.rank_id} (NPU {self.npu_id}) Shape {shape} tiling参数索引 {actual_idx} (mTile={parameters['mTile']}, nTile={parameters['nTile']}, kTile={parameters['kTile']}) 执行失败，标记为time=-1"
                            logger.error(error_msg)
                            result = CatlassResult(
                                idx=actual_idx,
                                M=shape[0],
                                N=shape[1],
                                K=shape[2],
                                time=-1,
                                diff=float('inf'),
                                kernel_func_name="",
                                parameters=parameters,
                            )
                        else:
                            # 正常结果
                            result = CatlassResult(
                                idx=actual_idx,
                                M=shape[0],
                                N=shape[1],
                                K=shape[2],
                                time=duration,
                                diff=accuracy if accuracy != -1 else float('inf'),
                                kernel_func_name=kernel_name if kernel_name else "",
                                parameters=parameters,
                                pipe_utilization={}  # PipeUtilization数据需要从CSV中提取，暂时留空
                            )
                        result_writer.add_result(result)
                        pbar.update(1)
                
                # 每批次完成后清空msp目录，释放磁盘空间
                self._clear_msp_dir()
                
                # 移动到下一批次
                current_idx = end_idx
                pbar.set_postfix({
                    'Processed': current_idx,
                    'Index': current_idx
                })
        
        # 循环结束时，刷新剩余结果并更新最终的checkpoint
        result_writer.flush()
        if process_task_count > 0:
            checkpoint_manager.save(process_task_count - 1)
    
    def run_benchmarks(self) -> None:
        """
        运行分配给当前进程的shape的基准测试
        
        """
        print("=====STARTING GEMM BENCHMARK=====")
        
        total_shapes = len(self.shape_group)
        if total_shapes == 0:
            print(f"Rank {self.rank_id} (NPU {self.npu_id}): 没有可处理的shape，直接返回")
            return
        
        # 按连续区间切分shape_group，保证每个进程处理一段连续的数据
        base_count = total_shapes // self.num_processes
        remainder = total_shapes % self.num_processes
        
        if self.rank_id < remainder:
            start_idx = self.rank_id * (base_count + 1)
            end_idx = start_idx + (base_count + 1)
        else:
            start_idx = remainder * (base_count + 1) + (self.rank_id - remainder) * base_count
            end_idx = start_idx + base_count
        
        start_idx = min(start_idx, total_shapes)
        end_idx = min(end_idx, total_shapes)
        assigned_shapes = self.shape_group[start_idx:end_idx]
        
        total_assigned = len(assigned_shapes)
        print(
            f"Rank {self.rank_id} (NPU {self.npu_id}): 分配了 {total_assigned} 个shape，区间范围 [{start_idx}, {end_idx})"
        )
        if total_assigned:
            print(f"  分配的shape列表: {assigned_shapes[:5]}{'...' if total_assigned > 5 else ''}")
        
        for idx, shape in enumerate(assigned_shapes, start=1):
            self.benchmark_shape(shape, idx, total_assigned)

