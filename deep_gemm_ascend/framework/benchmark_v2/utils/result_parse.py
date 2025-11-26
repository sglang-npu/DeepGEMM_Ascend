"""
msprof结果解析工具模块

该模块用于解析msprof工具的输出结果，包括：
1. 从标准输出中提取kernel名称、精度、耗时等信息
2. 从msprof生成的CSV文件中提取详细的性能数据
3. 支持单次和多次执行结果的解析

主要功能：
- 使用正则表达式解析msprof的标准输出
- 读取并解析msprof生成的OpBasicInfo CSV文件
- 将解析结果组织成结构化的数据格式
"""

import re
from typing import Tuple, List, Any, Optional, Dict
import glob
import os
import csv

from .logger import logger


# 字符串类型字段的正则表达式规则（用于从msprof输出中提取字符串信息）
str_rule = {
    "kernel_name": r'Kernel Func Name:\s*(.+)',  # 提取kernel函数名称
}

# 数值类型字段的正则表达式规则（用于从msprof输出中提取数值信息）
int_rule = {
    "accuracy": r'Max Relative Error:\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',  # 最大相对误差（支持科学计数法）
    "duration": r'Task Duration\(us\): (\d+\.\d+)',  # 任务耗时（微秒）
    "block_dim": r'Block Dim:\s*(\d+)',  # Block维度
    "mix_block_dim": r'Mix Block Dim:\s*(\d+)',  # Mix Block维度
}

# Kernel名称到操作符名称的映射表
# 用于将简化的kernel名称映射到msprof生成的目录中使用的完整操作符名称
kernel_to_op = {
    "PaddingMatmulKernelHalfLayout00Padding030":
        "_Z19PaddingMatmulKernelIDhN7Catlass6layout8RowMajorEDhS2_DhS2_LNS0_4Gemm6Kernel10PaddingTagE0ELS5_3EEvmPhS6_S6_S6_S6_S6__mix_aic",
    "PaddingMatmulKernelHalfLayout00Padding300":
        "_Z19PaddingMatmulKernelIDhN7Catlass6layout8RowMajorEDhS2_DhS2_LNS0_4Gemm6Kernel10PaddingTagE3ELS5_0EEvmPhS6_S6_S6_S6_S6__mix_aic",
    "PaddingMatmulKernelHalfLayout00Padding330":
        "_Z19PaddingMatmulKernelIDhN7Catlass6layout8RowMajorEDhS2_DhS2_LNS0_4Gemm6Kernel10PaddingTagE3ELS5_3EEvmPhS6_S6_S6_S6_S6__mix_aic",
    "SmallMatmulKernelHalfLayout00":
        "_Z17SmallMatmulKernelIDhN7Catlass6layout8RowMajorEDhS2_DhS2_EvPhS3_S3_S3_",
    "CommonMatmulKernelHalfLayout00":
        "_Z18CommonMatmulKernelIDhN7Catlass6layout8RowMajorEDhS2_DhS2_EvPhS3_S3_S3_",
}

# 默认的搜索空间（从标准输出中提取的字段列表）
# 注意：accuracy字段已被注释，当前只提取kernel_name
# DEFAULT_SEARCH_SPACE = ["kernel_name", "accuracy"]
DEFAULT_SEARCH_SPACE = ["kernel_name"]

# msprof生成的CSV文件名称模式（使用通配符匹配）
CSV_FILE_NAME = "OpBasicInfo_*.csv"

# CSV文件中的列名常量
TASK_DURATION = "Task Duration(us)"  # 任务耗时列名
BLOCK_DIM = "Block Dim"  # Block维度列名
Mix_Block_Dim = "Mix Block Dim"  # Mix Block维度列名


def get_sub_dir(parent_dir: str) -> List[str]:
    """
    获取指定目录下的所有子目录列表
    
    Args:
        parent_dir: 父目录路径
        
    Returns:
        子目录路径列表，如果目录不存在或没有子目录则返回空列表
    """
    if not os.path.isdir(parent_dir):
        logger.error(f"parent_dir {parent_dir} is not exist.")
        return []
    # 过滤出所有子目录（排除文件）
    subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not subdirs:
        logger.error(f"parent_dir {parent_dir} has no subdir.")
    return subdirs
    

def get_lastest_dir(parent_dir: str) -> Optional[str]:
    """
    获取指定目录下最新的子目录（按名称排序后的最后一个）
    
    通常msprof会按时间戳创建目录，排序后最后一个即为最新目录
    
    Args:
        parent_dir: 父目录路径
        
    Returns:
        最新子目录的完整路径，如果不存在子目录则返回None
    """
    subdirs = get_sub_dir(parent_dir)
    if not subdirs:
        logger.error(f"get_lastest_dir: 目录 {parent_dir} 下没有子目录")
        return None
    subdirs.sort()  # 按名称排序，通常时间戳目录会按时间顺序排列
    latest_dir = subdirs[-1]  # 取最后一个（最新的）
    return latest_dir


def string_to_float(input_str: str) -> float:
    """
    将字符串转换为浮点数，保留2位小数
    
    Args:
        input_str: 待转换的字符串
        
    Returns:
        转换后的浮点数，如果转换失败返回-1
    """
    try:
        return round(float(input_str), 2)
    except ValueError:
        return -1


def string_to_int(input_str: str) -> int:
    """
    将字符串转换为整数
    
    Args:
        input_str: 待转换的字符串
        
    Returns:
        转换后的整数，如果转换失败返回-1
    """
    try:
        return int(input_str)
    except ValueError:
        return -1


class ResultParse:
    """
    msprof结果解析器
    
    该类用于解析msprof工具的输出结果，包括：
    1. 从标准输出中提取信息（使用正则表达式）
    2. 从msprof生成的CSV文件中读取详细性能数据
    3. 支持单次和多次执行结果的解析
    """
    
    def __init__(self, kernel_list: List[str]):
        """
        初始化结果解析器
        
        Args:
            kernel_list: 需要解析的kernel名称列表，这些kernel对应的CSV文件会被读取
        """
        self.kernel_list = kernel_list
        # 预编译所有正则表达式规则，提高后续匹配效率
        self.re_searcher = {}
        # 编译字符串类型规则
        for rule_name, rule_str in str_rule.items():
            self.re_searcher[rule_name] = re.compile(rule_str)
        # 编译数值类型规则
        for rule_name, rule_str in int_rule.items():
            self.re_searcher[rule_name] = re.compile(rule_str)

    def parse_result(self, origin_result: str, search_space) -> Dict[str, Any]:
        """
        解析单次执行的结果（从标准输出中提取信息）
        
        使用预编译的正则表达式从msprof的标准输出中提取指定字段的值
        
        Args:
            origin_result: msprof的标准输出字符串
            search_space: 要提取的字段名称列表，如 ["kernel_name", "accuracy"]
            
        Returns:
            包含提取字段的字典，格式为 {字段名: 字段值}
            如果某个字段提取失败，该字段不会出现在返回字典中
        """
        res_dict = {}
        for search_name in search_space:
            # 使用正则表达式搜索第一个匹配项
            res = self.re_searcher[search_name].search(origin_result)
            if not res:
                logger.error(f"get search rule '{search_name}' result failed.")
                continue
            # 根据字段类型进行不同的处理
            if search_name in str_rule.keys():
                # 字符串类型：去除首尾空白
                res_dict[search_name] = res.group(1).strip()
            elif search_name in int_rule.keys():
                # 数值类型：转换为浮点数并保留2位小数
                res_dict[search_name] = round(float(res.group(1).strip()), 2)
            else:
                logger.error(f"cannot find search rule '{search_name}'.")
        return res_dict

    def _multi_parse_result(self, origin_result: str, search_space: List[str]) -> Dict[str, List[Any]]:
        """
        解析多次执行的结果（从标准输出中提取所有匹配项）
        
        与parse_result的区别：此方法使用findall提取所有匹配项，而不是只提取第一个
        
        Args:
            origin_result: msprof的标准输出字符串
            search_space: 要提取的字段名称列表
            
        Returns:
            包含提取字段的字典，格式为 {字段名: [字段值列表]}
            每个字段对应一个列表，包含所有匹配的值
        """
        res_dict = {}
        for search_name in search_space:
            # 使用findall提取所有匹配项（而不是只提取第一个）
            res = self.re_searcher[search_name].findall(origin_result)
            if not res:
                logger.error(f"get search rule '{search_name}' result failed.")
                continue
            # 根据字段类型进行不同的处理
            if search_name in str_rule.keys():
                # 字符串类型：直接使用匹配结果列表
                res_dict[search_name] = res
            elif search_name in int_rule.keys():
                # 数值类型：转换为浮点数并保留2位小数
                res_dict[search_name] = [round(float(s), 2) for s in res]
            else:
                logger.error(f"cannot find search rule '{search_name}'.")
        return res_dict

    def __get_csv_file(self, parent_dir: str, file_name: str) -> Optional[str]:
        """
        在指定目录下查找匹配的CSV文件（私有方法）
        
        Args:
            parent_dir: 父目录路径
            file_name: 文件名模式（支持通配符，如 "OpBasicInfo_*.csv"）
            
        Returns:
            找到的第一个CSV文件的完整路径，如果未找到则返回None
        """
        logger.debug(f"try to get csv file from directory {parent_dir}.")
        pattern_name = os.path.join(parent_dir, file_name)
        csv_file = glob.glob(pattern_name)  # 使用glob模式匹配文件
        if not csv_file:
            logger.error(f"cannot find any op time csv in {parent_dir}.")
            return None
        return csv_file[0]  # 返回第一个匹配的文件

    def __get_csv_list(self, max_count: int, msprof_res_dir: str, op_name: str) -> List[str]:
        """
        获取指定操作符的所有CSV文件列表（私有方法）
        
        msprof的目录结构通常是：
        msprof_res_dir/
          op_name/
            0/
              OpBasicInfo_*.csv
            1/
              OpBasicInfo_*.csv
            ...
        
        Args:
            max_count: 最大索引数量（通常对应执行次数）
            msprof_res_dir: msprof结果根目录
            op_name: 操作符名称（完整的C++ mangled名称）
            
        Returns:
            CSV文件路径列表，按索引顺序排列
        """
        logger.debug(f"try to get csv list for kernel {op_name}.")
        # 1、校验msprof目录结构
        op_res_dir = f"{msprof_res_dir}/{op_name}"
        if not os.path.isdir(op_res_dir):
            logger.error(f"op result directory '{op_res_dir}' is not exist.")
            return []
        # 2、按索引顺序获取所有CSV文件（0, 1, 2, ..., max_count-1）
        csv_list = []
        for idx in range(max_count):
            final_res_dir = f"{op_res_dir}/{idx}"
            if not os.path.isdir(final_res_dir):
                # 如果某个索引目录不存在，说明执行次数少于max_count
                logger.debug(f"op result directory '{op_res_dir}' only has {idx} sub directory.")
                break
            csv_file = self.__get_csv_file(final_res_dir, CSV_FILE_NAME)
            csv_list.append(csv_file)
        return csv_list

    def _get_csv_dict(self, max_count: int, msprof_res_dir: str) -> Dict[str, Optional[List[str]]]:
        """
        获取所有kernel对应的CSV文件字典（内部方法）
        
        为每个kernel名称查找对应的CSV文件列表，通过kernel_to_op映射表
        将简化的kernel名称转换为完整的操作符名称
        
        Args:
            max_count: 最大执行次数（对应CSV文件索引范围）
            msprof_res_dir: msprof结果根目录
            
        Returns:
            字典，格式为 {kernel_name: [csv_file_path_list]}
            如果某个kernel没有找到CSV文件，对应的值为空列表
        """
        logger.debug(f"try to get csv dict.")
        csv_dict = {}
        for kernel_name in self.kernel_list:
            # 通过映射表将kernel名称转换为完整的操作符名称
            csv_list = self.__get_csv_list(max_count, msprof_res_dir, kernel_to_op[kernel_name])
            csv_dict[kernel_name] = csv_list
        return csv_dict

    @staticmethod
    def _get_res_dict(csv_dict: Dict[str, List[str]]) -> Dict[str, Optional[List[List[str]]]]:
        """
        从CSV文件列表中解析性能数据（静态方法）
        
        读取每个CSV文件的第一行数据，提取任务耗时、Block维度、Mix Block维度等信息
        
        Args:
            csv_dict: CSV文件字典，格式为 {kernel_name: [csv_file_path_list]}
            
        Returns:
            解析结果字典，格式为 {kernel_name: [duration_list, block_dim_list, mix_block_dim_list]}
            每个kernel对应三个列表：
            - duration_list: 任务耗时列表（微秒）
            - block_dim_list: Block维度列表
            - mix_block_dim_list: Mix Block维度列表
            如果某个kernel没有有效数据，对应的值为None
        """
        logger.debug(f"try to get result dict.")
        res_dict = {}
        for kernel_name, csv_list in csv_dict.items():
            duration_list = []
            block_dim_list = []
            mix_block_dim_list = []
            # 遍历该kernel的所有CSV文件
            for csv_file in csv_list:
                with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                    csv_data = list(csv.DictReader(f))  # 读取CSV为字典列表
                    # 从第一行提取数据（通常每个CSV文件只有一行数据）
                    duration_list.append(string_to_float(csv_data[0][TASK_DURATION]))
                    block_dim_list.append(string_to_int(csv_data[0][BLOCK_DIM]))
                    mix_block_dim_list.append(string_to_int(csv_data[0][Mix_Block_Dim]))
            # 检查是否有有效数据
            if not duration_list and not block_dim_list and not mix_block_dim_list:
                res_dict[kernel_name] = None
                logger.debug(f"res dict for kernel {kernel_name} is empty.")
                continue
            # 将三个列表组合成一个列表返回
            res_dict[kernel_name] = [duration_list, block_dim_list, mix_block_dim_list]
        return res_dict

    def parse_multi_result(self,
                           origin_result: str,
                           count: int
    ) -> Optional[List[Tuple]]:
        """
        解析多次执行的结果（主入口方法）
        
        该方法综合解析msprof的输出：
        1. 从标准输出中提取kernel名称列表
        2. 从msprof生成的CSV文件中提取详细的性能数据
        3. 将两者关联，生成结构化的结果列表
        
        执行流程：
        - 从origin_result中提取msprof结果目录路径
        - 读取该目录下所有相关kernel的CSV文件
        - 从标准输出中提取kernel名称序列
        - 将kernel名称与CSV数据一一对应
        
        Args:
            origin_result: msprof的完整标准输出字符串（包含目录路径和kernel信息）
            count: 执行次数（用于确定要读取多少个CSV文件）
            
        Returns:
            结果元组列表，每个元组格式为：
            (kernel_name, accuracy, duration, block_dim, mix_block_dim)
            - kernel_name: kernel名称（字符串）
            - accuracy: 精度值（当前固定为0.0，已废弃）
            - duration: 任务耗时（微秒，浮点数）
            - block_dim: Block维度（整数）
            - mix_block_dim: Mix Block维度（整数）
            
            如果某个kernel不在kernel_list中或没有CSV数据，对应位置为-1
            如果解析失败，返回None
        """
        # 从标准输出中提取msprof结果目录路径（格式：saved in /path/to/dir）
        msprof_dir_match = re.search(r'saved in (\S+)', origin_result)
        if not msprof_dir_match:
            logger.error(f"cannot get msprof result directory.")
            return None
        
        # 步骤1：获取各种算子的CSV文件位置
        csv_dict = self._get_csv_dict(count, msprof_dir_match.group(1).strip())
        
        # 步骤2：解析CSV文件获取性能数据
        res_dict = self._get_res_dict(csv_dict)

        # 步骤3：从标准输出中提取kernel名称列表
        res_from_std = self._multi_parse_result(origin_result, DEFAULT_SEARCH_SPACE)
        list_kernel_name = res_from_std["kernel_name"]
        # 注意：accuracy字段已被注释，当前不使用
        # list_accuracy = res_from_std["accuracy"]

        # 步骤4：为每个kernel初始化索引计数器（用于从CSV数据中按顺序取数据）
        idx_dict = {}
        for kernel_name in res_dict.keys():
            idx_dict[kernel_name] = 0

        # 步骤5：遍历kernel名称列表，将标准输出中的kernel与CSV数据关联
        final_output_list = []
        for idx in range(len(list_kernel_name)):
            kernel_name = list_kernel_name[idx]
            # accuracy = list_accuracy[idx]  # 已废弃
            
            # 如果该kernel不在kernel_list中（不需要解析），则填充-1
            if kernel_name not in res_dict.keys():
                final_output_list.append((kernel_name, -1, -1, -1, -1))
                continue

            # 检查该kernel是否有有效数据
            if not res_dict[kernel_name]:
                logger.error(f"res dict for kernel {kernel_name} is empty.")
                return None
            
            # 从CSV数据中按顺序提取对应索引的数据
            duration = res_dict[kernel_name][0][idx_dict[kernel_name]]  # 耗时列表
            block_dim = res_dict[kernel_name][1][idx_dict[kernel_name]]  # Block维度列表
            mix_block_dim = res_dict[kernel_name][2][idx_dict[kernel_name]]  # Mix Block维度列表

            # 组合成结果元组（accuracy固定为0.0）
            # final_output_list.append((kernel_name, accuracy, duration, block_dim, mix_block_dim))
            final_output_list.append((kernel_name, 0.0, duration, block_dim, mix_block_dim))
            idx_dict[kernel_name] += 1  # 更新索引，为下一个同名kernel准备

        return final_output_list
