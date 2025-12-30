import re
from typing import Tuple, List, Any, Optional, Dict
import glob
import os
import csv

from .logger import logger


# 从msprof op的采集到的字符串结果对应的规则(import re)
str_rule = {
    "kernel_name": r'Kernel Func Name :\s*(\S+)',
}
# 从msprof op的采集到的数字结果对应的规则(import re)
int_rule = {
    "accuracy": r'Max Relative Error:\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)',
    "duration": r'Task Duration\(us\): (\d+\.\d+)',
    "block_dim": r'Block Dim:\s*(\d+)',
    "mix_block_dim": r'Mix Block Dim:\s*(\d+)',
}
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
    "CommonMatmulKernelHalfLayout01":
        "_Z18CommonMatmulKernelIDhN7Catlass6layout8RowMajorEDhNS1_11ColumnMajorEDhS2_EvPhS4_S4_S4_",
    "PaddingStreamkMatmulKernelHalfLayout01":
        "_Z19StreamkMatmulKernelIDhN7Catlass6layout8RowMajorEDhNS1_11ColumnMajorEDhS2_LNS0_4Gemm6Kernel10PaddingTagE0ELS6_0EEvmPhS7_S7_S7_S7_S7_S7__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding030":
        "_Z19PaddingMatmulKernelIDhN7Catlass6layout8RowMajorEDhNS1_11ColumnMajorEDhS2_LNS0_4Gemm6Kernel10PaddingTagE0ELS6_3EEvmPhS7_S7_S7_S7_S7__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding300":
        "_Z19PaddingMatmulKernelIDhN7Catlass6layout8RowMajorEDhNS1_11ColumnMajorEDhS2_LNS0_4Gemm6Kernel10PaddingTagE3ELS6_0EEvmPhS7_S7_S7_S7_S7__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding330":
        "_Z25PaddingCommonMatmulKernelIN7Catlass4Arch7AtlasA2EDhNS0_6layout8RowMajorEDhNS3_11ColumnMajorEDhS4_LNS0_4Gemm6Kernel10PaddingTagE3ELS8_3ELS8_0EEvmPhS9_S9_S9_S9_S9_S9__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding331":
        "_Z25PaddingCommonMatmulKernelIN7Catlass4Arch7AtlasA2EDhNS0_6layout8RowMajorEDhS4_DhS4_LNS0_4Gemm6Kernel10PaddingTagE3ELS7_3ELS7_1EEvmPhS8_S8_S8_S8_S8_S8__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding031":
        "_Z25PaddingCommonMatmulKernelIN7Catlass4Arch7AtlasA2EDhNS0_6layout8RowMajorEDhNS3_11ColumnMajorEDhS4_LNS0_4Gemm6Kernel10PaddingTagE0ELS8_3ELS8_1EEvmPhS9_S9_S9_S9_S9_S9__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding001":
        "_Z25PaddingCommonMatmulKernelIN7Catlass4Arch7AtlasA2EDhNS0_6layout8RowMajorEDhNS3_11ColumnMajorEDhS4_LNS0_4Gemm6Kernel10PaddingTagE0ELS8_0ELS8_1EEvmPhS9_S9_S9_S9_S9_S9__mix_aic",
    "PaddingMatmulKernelHalfLayout01Padding301":
        "_Z25PaddingCommonMatmulKernelIN7Catlass4Arch7AtlasA2EDhNS0_6layout8RowMajorEDhNS3_11ColumnMajorEDhS4_LNS0_4Gemm6Kernel10PaddingTagE3ELS8_0ELS8_1EEvmPhS9_S9_S9_S9_S9_S9__mix_aic",
}
# DEFAULT_SEARCH_SPACE = ["kernel_name", "accuracy"]
DEFAULT_SEARCH_SPACE = ["kernel_name"]
CSV_FILE_NAME = "OpBasicInfo_*.csv"
TASK_DURATION = "Task Duration(us)"
BLOCK_DIM = "Block Dim"
Mix_Block_Dim = "Mix Block Dim"


def get_sub_dir(parent_dir: str) -> List[str]:
    """
    功能: 获取父目录下的所有子目录
    输入: 父目录
    输出: 所有子目录的列表
    """
    if not os.path.isdir(parent_dir):
        logger.error(f"parent_dir {parent_dir} is not exist.")
        return []
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
    功能: 获取父目录下的所有子目录中, 最新创建的一个
    输入: 父目录
    输出: 最新创建的子目录名
    """
    subdirs = get_sub_dir(parent_dir)
    if not subdirs:
        return None
    subdirs.sort()
    latest_dir = subdirs[-1]
    return latest_dir


def string_to_float(input_str: str, dec=2) -> float:
    """
    功能: 转换字符串到浮点数, 默认保留两位小数
    输入: 需要转换的字符串
    输出: 转换结果, 如果转换失败则返回-1
    """
    try:
        return round(float(input_str), dec)
    except ValueError:
        return -1


def string_to_int(input_str: str) -> int:
    """
    功能: 转换字符串到整数
    输入: 需要转换的字符串
    输出: 转换结果, 如果转换失败则返回-1
    """
    try:
        return int(input_str)
    except ValueError:
        return -1


class ResultParse:
    """
    功能: 解析msprof op运行结果
    使用:
    1. 传入需要解析的算子名称列表 res_parse = ResultParse(['PaddingMatmulKernelHalfLayout00Padding030'])
    2. 调用 res_parse.parse_multi_result(mspros_res, count) 一次性解析采集到的多条算子信息
    3. 也可以调用 res_parse.parse_result(mspros_res, ["kernel_name", "duration"]) 解析单条算子信息, 如果count大于1, 不建议单条采集
    """
    def __init__(self, kernel_list: List[str], device_id: int):
        self.kernel_list = kernel_list
        self.device_dir = f"device{device_id}"
        self.re_searcher = {}
        for rule_name, rule_str in str_rule.items():
            self.re_searcher[rule_name] = re.compile(rule_str)
        for rule_name, rule_str in int_rule.items():
            self.re_searcher[rule_name] = re.compile(rule_str)

    def parse_result(self, origin_result: str, search_space: str) -> Dict[str, Any]:
        """
        功能: 采集msprof回显结果中的第一条结果
        输入:
        1. origin_result: msprof的回显信息
        2. search_space: 需要搜索的信息, 要保证规则存在于int_rule和str_rule中
        输出: 采集结果的字典, 类型已经转换
        """
        res_dict = {}
        for search_name in search_space:
            res = self.re_searcher[search_name].search(origin_result)
            if not res:
                logger.error(f"get search rule '{search_name}' result failed.")
                continue
            if search_name in str_rule.keys():
                res_dict[search_name] = res.group(1).strip()
            elif search_name in int_rule.keys():
                res_dict[search_name] = round(float(res.group(1).strip()), 2)
            else:
                logger.error(f"cannot find search rule '{search_name}'.")
        return res_dict

    def _multi_parse_result(self, origin_result: str, search_space: List[str]) -> Dict[str, List[Any]]:
        """
        功能: 批量采集msprof回显结果中结果
        输入:
        1. origin_result: msprof的回显信息
        2. search_space: 需要搜索的信息, 要保证规则存在于int_rule和str_rule中
        输出: 采集结果列表的字典, 类型已经转换
        注意: 由于msprof采集算子的输出结果顺序可能与执行顺序无关, 因此该接口不适用于算子信息的采集, 建议外部不要使用这个接口
        """
        res_dict = {}
        for search_name in search_space:
            res = self.re_searcher[search_name].findall(origin_result)
            if not res:
                logger.error(f"get search rule '{search_name}' result failed.")

                continue
            if search_name in str_rule.keys():
                res_dict[search_name] = res
            elif search_name in int_rule.keys():
                res_dict[search_name] = [round(float(s), 2) for s in res]
            else:
                logger.error(f"cannot find search rule '{search_name}'.")
        return res_dict

    def __get_csv_file(self, parent_dir: str, file_name: str) -> Optional[str]:
        """
        功能: 获取目录下某个文件(用于csv文件所以带csv标记, 实际也可用于其他文件)
        输入: 文件所在目录及文件名(带通配符用于标注时间戳, 所以无法直接获取)
        输出: 获取到的文件名
        注意: 最好确定目录下只有一个匹配的文件, 否则只会获取第一个匹配的文件
        """
        logger.debug(f"try to get csv file from directory {parent_dir}.")
        pattern_name = os.path.join(parent_dir, file_name)
        csv_file = glob.glob(pattern_name)
        if not csv_file:
            logger.error(f"cannot find any op time csv in {parent_dir}.")
            return None
        return csv_file[0]

    def __get_csv_list(self, max_count: int, msprof_res_dir: str, op_name: str) -> List[str]:
        """
        功能: 获取msprof结果目录下所有OpBasicInfo文件
        输入:
        1. max_count: 当次msprof预期采集的全部算子数量(保证所有算子的所有文件都能获取到)
        2. msprof_res_dir: msprof的结果目录
        3. op_name: 想要采集的msprof算子名(也是目录名)
        输出: 获取到的文件列表
        """
        logger.debug(f"try to get csv list for kernel {op_name}.")
        # 1、校验msprof目录
        op_res_dir = msprof_res_dir
        path_device = os.path.join(msprof_res_dir, self.device_dir)
        if os.path.exists(path_device):
            op_res_dir = path_device
        op_res_dir = os.path.join(op_res_dir, op_name)
        if not os.path.isdir(op_res_dir):
            logger.info(f"op result directory '{op_res_dir}' is not exist.")
            return []
        # 2、按count获取最新目录下的所有csv文件
        csv_list = []
        for idx in range(max_count):
            final_res_dir = f"{op_res_dir}/{idx}"
            if not os.path.isdir(final_res_dir):
                logger.debug(f"op result directory '{op_res_dir}' only has {idx} sub directory.")
                break
            csv_file = self.__get_csv_file(final_res_dir, CSV_FILE_NAME)
            csv_list.append(csv_file)
        return csv_list

    def _get_csv_dict(self, max_count: int, msprof_res_dir: str) -> Dict[str, Optional[List[str]]]:
        """
        功能: 获取msprof结果目录下所有OpBasicInfo文件
        输入:
        1. max_count: 当次msprof预期采集的全部算子数量(保证所有算子的所有文件都能获取到)
        2. msprof_res_dir: msprof的结果目录
        输出: 获取到的文件列表与算子名称的对应, 只采集创建ResultParse时输入的算子名
        """
        logger.debug(f"try to get csv dict.")
        csv_dict = {}
        for kernel_name in self.kernel_list:
            csv_list = self.__get_csv_list(max_count, msprof_res_dir, kernel_to_op[kernel_name])
            csv_dict[kernel_name] = csv_list
        return csv_dict

    @staticmethod
    def _get_res_dict(csv_dict: Dict[str, List[str]]) -> Dict[str, Optional[List[List[str]]]]:
        """
        功能: 获取msprof结果目录下所有OpBasicInfo文件中的信息
        输入: _get_csv_dict获取到的csv_dict
        输出: 获取到的文件列表与采集信息元组列表的对应
        注意: 这里需要获取的信息有duration、block dim、mix block dim, 自定义时需要多处进行修改
        """
        logger.debug(f"try to get result dict.")
        res_dict = {}
        for kernel_name, csv_list in csv_dict.items():
            duration_list = []
            block_dim_list = []
            mix_block_dim_list = []
            for csv_file in csv_list:
                with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                    csv_data = list(csv.DictReader(f))
                    duration_list.append(string_to_float(csv_data[0][TASK_DURATION]))
                    block_dim_list.append(string_to_int(csv_data[0][BLOCK_DIM]))
                    mix_block_dim_list.append(string_to_int(csv_data[0][Mix_Block_Dim]))
            if not duration_list and not block_dim_list and not mix_block_dim_list:
                res_dict[kernel_name] = None
                logger.debug(f"res dict for kernel {kernel_name} is empty.")
                continue
            res_dict[kernel_name] = [duration_list, block_dim_list, mix_block_dim_list]
        return res_dict

    def parse_multi_result(self, origin_result: str, count: int) -> Optional[List[Tuple]]:
        """
        功能: 采集count大于1时, msprof的结果
        输入:
        1. origin_result: msprof的回显结果
        2. count: msprof预期采集的算子数量
        输出: 解析到的信息的元组的列表, 已排好顺序
        注意: 暂不支持精度accuracy的采集
        """
        msprof_dir_match = re.search(r'saved in (\S+)', origin_result)
        if not msprof_dir_match:
            logger.error(f"cannot get msprof result directory.")
            return None
        # 获取各种算子的csv文件位置
        csv_dict = self._get_csv_dict(count, msprof_dir_match.group(1).strip())
        # 解析csv文件获取内容
        res_dict = self._get_res_dict(csv_dict)

        res_from_std = self._multi_parse_result(origin_result, DEFAULT_SEARCH_SPACE)
        list_kernel_name = res_from_std["kernel_name"]
        # list_accuracy = res_from_std["accuracy"]

        idx_dict = {}
        for kernel_name in res_dict.keys():
            idx_dict[kernel_name] = 0

        final_output_list = []
        for idx in range(len(list_kernel_name)):
            kernel_name = list_kernel_name[idx]
            # accuracy = list_accuracy[idx]
            # 如果不需要该kernel_name对应算子数据, 则全置为-1
            if kernel_name not in res_dict.keys():
                final_output_list.append((kernel_name, -1, -1, -1, -1))
                continue

            if not res_dict[kernel_name]:
                logger.error(f"res dict for kernel {kernel_name} is empty.")
                return None
            duration = res_dict[kernel_name][0][idx_dict[kernel_name]]
            block_dim = res_dict[kernel_name][1][idx_dict[kernel_name]]
            mix_block_dim = res_dict[kernel_name][2][idx_dict[kernel_name]]

            # final_output_list.append((kernel_name, accuracy, duration, block_dim, mix_block_dim))
            final_output_list.append((kernel_name, 0.0, duration, block_dim, mix_block_dim))
            idx_dict[kernel_name] += 1

        return final_output_list
