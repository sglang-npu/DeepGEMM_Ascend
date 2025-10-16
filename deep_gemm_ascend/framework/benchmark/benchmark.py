
from typing import List, Dict, Any 
import torch
from torch import Tensor
import numpy as np 
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from tqdm import tqdm
import os
import json
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import deep_gemm_ascend 
import subprocess
import math, argparse, re
import time


shape_group = [
    # M、N、K
    [8, 4096, 7168], # 1h 15min   1240
    [8, 7168, 18432], # 1h 15min   1240
    [8, 18432, 7168], # 1h 15min   1240
    [64, 4096, 7168], # 5h  5906
    [64, 7168, 18432], # 5h  5906
    [64, 18432, 7168], # 5h  5906
    [64, 24576, 1536], # 5h  5906
    [64, 32768, 512], # 5h  5906
    [64, 7168, 16384], # 5h  5906
    [128, 4096, 7168], # 9h   9660
    [128, 7168, 18432], # 9h   9660
    [128, 18432, 7168], # 9h   9660
    [1024, 4096, 7168], # 14h   14520
    [1024, 18432, 7168], # 14h   14520
    [2048, 4096, 7168], # 14h   14520
    [1279, 5003, 7681],
    [3511, 6151, 8191],
    [5119, 6997, 9901]
]

class Parameter():
    def __init__(self):
        # self.all_parameters = self.init_generate_parameters()
        self.grid_parameters = self.grid_generate_parameters()

    def generate_mn_sections(self):
        # Rule 1：m_sections × n_sections <= 24
        for m_sections in range(1, 25):
            max_n_sections = min(24, 24 // m_sections)
            for n_sections in range(1, max_n_sections + 1):
                yield (m_sections, n_sections)

    def generate_mn_sections_linear(self):
        m_sections_values = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
        n_sections_values = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]

        for m_sections in m_sections_values:
            max_n_sections = 24 // m_sections
            n_sections_valid_values = [n for n in n_sections_values if n <= max_n_sections]

            for n_sections in n_sections_valid_values:
                yield m_sections, n_sections

    def generate_mnk_db_o_blocks(self):
        for m_sec_o_blocks in range(1, 128):
            for n_sec_o_blocks in range(1, 128 - m_sec_o_blocks):
                # Rule 4: m × n <= 128
                if m_sec_o_blocks * n_sec_o_blocks > 128:
                    continue

                # Rule 2：(m + n) × k × 2 < 1024 → k < 1024/(m + n)/2
                max_k_o_iter_blocks = 1023 // (m_sec_o_blocks + n_sec_o_blocks) // 2

                # Rule 3：m_sec_o_blocks × db_o_blocks <= 128 && n_sec_o_blocks × db_o_blocks <= 128 → db_o_blocks <= min(128/m_sec_o_blocks, 128/m_sec_o_blocks)
                max_db_from_m = 128 // m_sec_o_blocks
                max_db_from_n = 128 // n_sec_o_blocks
                max_db = min(max_db_from_m, max_db_from_n)
                for k_o_iter_blocks in range(1, max_k_o_iter_blocks + 1):
                    for db_o_blocks in range(1, max_db + 1):
                        yield m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks

    def generate_mnk_db_o_blocks_linear(self):
        m_sec_o_blocks_values = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        n_sec_o_blocks_values = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        k_o_iter_blocks_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        db_o_blocks_values = [1, 2, 4, 8, 16, 32, 64]

        for m_sec_o_blocks in m_sec_o_blocks_values:
            for n_sec_o_blocks in n_sec_o_blocks_values:
                # Rule 4: m × n <= 128
                if m_sec_o_blocks * n_sec_o_blocks > 128:
                    continue

                # Rule 2：(m + n) × k × 2 < 1024 → k < 1024/(m + n)/2
                max_k_o_iter_blocks = 1023 // (m_sec_o_blocks + n_sec_o_blocks) // 2
                for k_o_iter_blocks in k_o_iter_blocks_values:
                    if k_o_iter_blocks > max_k_o_iter_blocks:
                        continue

                    # Rule 3：m_sec_o_blocks × db_o_blocks <= 128 && n_sec_o_blocks × db_o_blocks <= 128 → db_o_blocks <= min(128/m_sec_o_blocks, 128/m_sec_o_blocks)
                    max_db_o_blocks_from_m = 128 // m_sec_o_blocks
                    max_db_o_blocks_from_n = 128 // n_sec_o_blocks
                    max_db_o_blocks = min([max_db_o_blocks_from_m, max_db_o_blocks_from_n, k_o_iter_blocks])

                    for db_o_blocks in db_o_blocks_values:
                        if db_o_blocks > max_db_o_blocks:
                            continue
                        yield m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks

    def init_generate_parameters(self):
        mn_sections_list = list(self.generate_mn_sections())
        o_blocks_list = list(self.generate_mnk_db_o_blocks())
        parameters = []
        print(f'mn_sections_list len:{len(mn_sections_list)}')
        print(f'o_blocks_list len:{len(o_blocks_list)}')

        for (m_sections, n_sections) in mn_sections_list:
            for (m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks) in o_blocks_list:
                param_dic = {
                    'm_sections': m_sections,
                    'n_sections': n_sections,
                    'm_sec_o_blocks': m_sec_o_blocks,
                    'n_sec_o_blocks': n_sec_o_blocks,
                    'k_o_iter_blocks': k_o_iter_blocks,
                    'db_o_blocks': db_o_blocks
                }
                parameters.append(param_dic)

        print(f'Init valid parameter group {len(parameters)}')
        return parameters

    def grid_generate_parameters(self):
        mn_sections_list = list(self.generate_mn_sections_linear())
        o_blocks_list = list(self.generate_mnk_db_o_blocks_linear())
        parameters = []
        print(f'mn_sections_list len:{len(mn_sections_list)}')
        print(f'o_blocks_list len:{len(o_blocks_list)}')

        for (m_sections, n_sections) in mn_sections_list:
            for (m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks) in o_blocks_list:
                param_dic = {
                    'm_sections': m_sections,
                    'n_sections': n_sections,
                    'm_sec_o_blocks': m_sec_o_blocks,
                    'n_sec_o_blocks': n_sec_o_blocks,
                    'k_o_iter_blocks': k_o_iter_blocks,
                    'db_o_blocks': db_o_blocks
                }
                parameters.append(param_dic)

        print(f'Grid valid parameter group {len(parameters)}')
        return parameters
    
    def filter_parameters(self, shape):
        M, N, K = shape
        
        max_m_sections = math.ceil(M / 16)
        max_n_sections = math.ceil(N / 16)

        min_m_sec_o_blocks = min(2, math.ceil(M / 16))
        min_n_sec_o_blocks = min(2, math.ceil(N / 16))
        min_k_o_iter_blocks = min(2, math.ceil(K / 16))
        max_m_sec_o_blocks = math.ceil(M / 16)
        max_n_sec_o_blocks = math.ceil(N / 16)
        max_k_o_iter_blocks = math.ceil(K / 16)
        
        # 过滤符合条件的参数
        filtered_params = []
        for param in self.grid_parameters:
            if (param['m_sections'] <= max_m_sections and
                param['n_sections'] <= max_n_sections and
                param['m_sec_o_blocks'] >= min_m_sec_o_blocks and
                param['n_sec_o_blocks'] >= min_n_sec_o_blocks and
                param['k_o_iter_blocks'] >= min_k_o_iter_blocks and 
                param['m_sec_o_blocks'] <= max_m_sec_o_blocks and
                param['n_sec_o_blocks'] <= max_n_sec_o_blocks and
                param['k_o_iter_blocks'] <= max_k_o_iter_blocks):
                filtered_params.append(param)
        
        print(f'Filtered parameters count: {len(filtered_params)}')
        return filtered_params

    def get_params_with_idx(self, shape, idx):
        params = self.filter_parameters(shape)
        return params[idx]
