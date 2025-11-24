import re
from typing import Tuple, List, Any, Optional, Dict
import glob
import os
import csv

from logger import logger


str_rule = {
    "kernel_name": r'Kernel Func Name:\s*(.+)',
}
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
}
# DEFAULT_SEARCH_SPACE = ["kernel_name", "accuracy"]
DEFAULT_SEARCH_SPACE = ["kernel_name"]
CSV_FILE_NAME = "OpBasicInfo_*.csv"
TASK_DURATION = "Task Duration(us)"
BLOCK_DIM = "Block Dim"
Mix_Block_Dim = "Mix Block Dim"


def get_sub_dir(parent_dir: str) -> List[str]:
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
    subdirs = get_sub_dir(parent_dir)
    if not subdirs:
        return None
    subdirs.sort()
    latest_dir = subdirs[-1]
    return latest_dir


def string_to_float(input_str: str) -> float:
    try:
        return round(float(input_str), 2)
    except ValueError:
        return -1


def string_to_int(input_str: str) -> int:
    try:
        return int(input_str)
    except ValueError:
        return -1


class ResultParse:
    def __init__(self, kernel_list: List[str]):
        self.kernel_list = kernel_list
        self.re_searcher = {}
        for rule_name, rule_str in str_rule.items():
            self.re_searcher[rule_name] = re.compile(rule_str)
        for rule_name, rule_str in int_rule.items():
            self.re_searcher[rule_name] = re.compile(rule_str)

    def parse_result(self, origin_result: str, search_space) -> Dict[str, Any]:
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
        logger.debug(f"try to get csv file from directory {parent_dir}.")
        pattern_name = os.path.join(parent_dir, file_name)
        csv_file = glob.glob(pattern_name)
        if not csv_file:
            logger.error(f"cannot find any op time csv in {parent_dir}.")
            return None
        return csv_file[0]

    def __get_csv_list(self, max_count: int, msprof_res_dir: str, op_name: str) -> List[str]:
        logger.debug(f"try to get csv list for kernel {op_name}.")
        # 1、校验msprof目录
        op_res_dir = f"{msprof_res_dir}/{op_name}"
        if not os.path.isdir(op_res_dir):
            logger.error(f"op result directory '{op_res_dir}' is not exist.")
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
        logger.debug(f"try to get csv dict.")
        csv_dict = {}
        for kernel_name in self.kernel_list:
            csv_list = self.__get_csv_list(max_count, msprof_res_dir, kernel_to_op[kernel_name])
            csv_dict[kernel_name] = csv_list
        return csv_dict

    @staticmethod
    def _get_res_dict(csv_dict: Dict[str, List[str]]) -> Dict[str, Optional[List[List[str]]]]:
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

    def parse_multi_result(self,
                           origin_result: str,
                           count: int
    ) -> Optional[List[Tuple]]:
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
            # 如果不需要该kernel_name对应算子数据，则全置为-1
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
