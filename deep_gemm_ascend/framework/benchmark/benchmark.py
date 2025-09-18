
from typing import List, Dict, Any 
import torch
from torch import Tensor
import numpy as np 
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np

import deep_gemm_ascend

shape_group = [
    [8, 4096, 7168],
    # [8, 7168, 18432],
    # [8, 18432, 7168],
    # [64, 4096, 7168],
    # [64, 7168, 18432],
    # [64, 18432, 7168],
    # [64, 24576, 1536],
    # [64, 32768, 512],
    # [64, 7168, 16384],
    # [128, 4096, 7168],
    # [128, 7168, 18432],
    # [128, 18432, 7168],
    # [1024, 4096, 7168],
    # [1024, 18432, 7168],
    # [2048, 4096, 7168],
    # [4096, 4096, 7168]
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

    # 网格调参
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
                sum_mn = m_sec_o_blocks + n_sec_o_blocks
                # Rule 2：(m + n) × k < 1024 → k < 1024/(m + n)
                max_k = 1023 // sum_mn // 2

                # Rule 3：m_sec_o_blocks × db_o_blocks <= 128 && n_sec_o_blocks × db_o_blocks <= 128 → db_o_blocks <= min(128/m_sec_o_blocks, 128/m_sec_o_blocks)
                max_db_from_m = 128 // m_sec_o_blocks
                max_db_from_n = 128 // n_sec_o_blocks
                max_db = min(max_db_from_m, max_db_from_n)
                for k_o_iter_blocks in range(1, max_k + 1):
                    for db_o_blocks in range(1, max_db + 1):
                        yield m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks

    def generate_mnk_db_o_blocks_linear(self):
        m_sec_o_blocks_values = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        n_sec_o_blocks_values = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
        k_o_iter_blocks_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        db_o_blocks_values = [1, 2, 4, 8, 16, 32, 64]

        for m_sec_o_blocks in m_sec_o_blocks_values:
            for n_sec_o_blocks in n_sec_o_blocks_values:
                sum_mn = m_sec_o_blocks + n_sec_o_blocks
                # Rule 2：(m + n) × k < 1024 → k < 1024/(m + n)
                max_k = 1023 // sum_mn // 2

                for k_o_iter_blocks in k_o_iter_blocks_values:
                    # Rule 3：m_sec_o_blocks × db_o_blocks <= 128 && n_sec_o_blocks × db_o_blocks <= 128 → db_o_blocks <= min(128/m_sec_o_blocks, 128/m_sec_o_blocks)
                    if k_o_iter_blocks > max_k:
                        continue

                    max_db_o_blocks_from_m = 128 // m_sec_o_blocks
                    max_db_o_blocks_from_n = 128 // n_sec_o_blocks
                    max_db_o_blocks = min(max_db_o_blocks_from_m, max_db_o_blocks_from_n)

                    for db_o_blocks in db_o_blocks_values:
                        if db_o_blocks > max_db_o_blocks:
                            continue
                        yield m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks

    # 满足约束条件的所有参数组合
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


@dataclass
class Result():
    idx: int
    M: int
    N: int
    K: int
    time: float
    idx: int
    # diff: float
    parameters: Dict[str, int] = field(default_factory=lambda:{
        'm_sections': 0,
        'n_sections': 0,
        'm_sec_o_blocks': 0,
        'n_sec_o_blocks': 0,
        'k_o_iter_blocks': 0,
        'db_o_blocks': 0
    })

    @classmethod
    def from_dict(cls, data: dict) -> 'Result':
        """Create a Result object from a dictionary"""
        return cls(
            M=data['M'],
            N=data['N'],
            K=data['K'],
            time=data['time'],
            idx=data['idx'],
            parameters=data.get('parameters', {})
        )


class GEMMBenchmarkRunner():
    def __init__(self, shape_group, result_dir="./results"):
        self.shape_group = shape_group
        self.result_dir = result_dir
        self.parameters = Parameter()
    
    def benchmark_shape(self, shape: list) -> None:
        # gen_data -> deepgemm_gemm && cann_gemm -> is_correct -> ms_prof -> save_result
        shape_str = '_'.join(map(str, shape))
        filename = f'shape_{shape_str}.jsonl'
        result_path = str(Path(result_dir) / filename)

        A, B = self.gen_data(shape)
        gold = self.cann_gemm(A, B)

        saved_count = 0      # 保存的结果数量
        max_saved_idx = -1   # 已保存的最大索引
        if os.path.exists(result_path):
            with jsonlines.open(result_path, mode='r') as reader:
                for result in reader:
                    saved_count += 1
                    if result['idx'] > max_saved_idx:
                        max_saved_idx = result['idx']

        start_idx = max_saved_idx + 1 if max_saved_idx >= 0 else 0
        results = []
        total_params = len(self.parameters.grid_parameters[start_idx:])
        with tqdm(total=total_params, initial=start_idx, desc=f"Testing shape {shape}", postfix={"Proccessed": start_idx, "Valid": saved_count}) as pbar:
            for idx, parameters in enumerate(self.parameters.grid_parameters):
                output = self.deepgemm_gemm(A, B, parameters)
                
                if self.is_correct(gold, output):
                    # TODO:
                    time_us = self.ms_prof()
                    result = Result(
                        idx=idx,
                        M=shape[0],
                        N=shape[1],
                        K=shape[2],
                        time=time_us,
                        idx=start_idx + idx,
                        # diff=?
                        parameters=parameters
                    )
                    results.append(result)
                    saved_count += 1
                
                if len(results) == 100 or idx == total_params - 1:
                    if results:
                        self.save_result(results, result_path)
                        results = []

                pbar.update(1)
                pbar.set_postfix({
                    'curren': idx + 1,
                    'valid': saved_count
                })

        return

    def gen_data(self, shape: list) -> tuple[Tensor, Tensor]:
        device = torch.device("npu" if torch.npu.is_available() else "cpu")
        if device.type == "cpu":
            assert False

        M, N, K = shape

        rng = np.random.default_rng()

        def heavy_tail(shape):
            v = rng.lognormal(mean=1.0, sigma=1.2, size=shape)
            return np.clip(v, 1, 10).astype(np.float16)

        A = heavy_tail([M, K])
        B = heavy_tail([K, N])

        A_npu = torch.tensor(A, device=device)
        B_npu = torch.tensor(B, device=device)
        return (A_npu, B_npu)

    def deepgemm_gemm(self, A: Tensor, B: Tensor, parameters: dict) -> Tensor:
        param_list = list(parameters.values())
        param_list.extend([0] * 22)
        param_npu = torch.tensor(param_list, device='npu', dtype=torch.int32)
        
        z_shape = [A.size(0), B.size(1)]
        z_npu = torch.zeros(z_shape, device='npu', dtype=torch.int32)

        deep_gemm_ascend.run_mmad_bench(A, B, z_npu, param_npu)
        return z_npu

    def cann_gemm(self, A: Tensor, B: Tensor) -> Tensor:
        out = torch.matmul(A, B)
        return out

    def is_correct(self, cann_result: Tensor, deepgemm_result: Tensor) -> bool:
        return True

    def ms_prof(self, output_path: str, kernel_path: str) -> float:
        pass
    
    def save_result(self, results: list, path: str) -> None:
        result_dicts = []
        for result in results:
            if is_dataclass(result):
                result_dicts.append(asdict(result))
            else:
                result_dicts.append(dict(result))
        
        try:
            with open(path, 'a', encoding='utf-8') as f:
                for result_dict in result_dicts:
                    json.dump(result_dict, f, ensure_ascii=False)
                    f.write('\n')
        except IOError as e:
            print(f"save files error: {e}")
        except Exception as e:
            print(f"process data error: {e}")
    
    def run_benchmarks(self) -> None:
        print("=====STARTING GEMM BENCHMARK=====")

        for shape in self.shape_group:
            self.benchmark_shape(shape)
            
    
    def visualize_time_with_single_parameter(self, shape: list, target_parameter: str, other_parameters: dict) -> None:
        if not os.path.exists(self.result_path):
            raise FileNotFoundError(f"File not found: {self.result_path}")
        
        results: List[Result] = []
        with open(self.result_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    results.append(Result.from_dict(data))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line content: {line}")
                    continue
        
        if not results:
            print("No valid Result data was read")
            return
        
        # 检查shape合法性
        if len(shape) != 3:
            raise ValueError(f"shape must contain exactly 3 elements [M, N, K], got {len(shape)}")
        try:
            target_M, target_N, target_K = map(int, shape)
        except (ValueError, TypeError):
            raise ValueError(f"shape must contain integer values, got {shape}")

        # 检查target_parameter合法性
        valid_parameters = results[0].parameters.keys()
        if target_parameter not in valid_parameters:
            raise ValueError(f"Target parameter '{target_parameter}' is not valid. Valid parameters: {valid_parameters}")
        
        # 检查other_parameters合法性
        if len(other_parameters) != 5:
            raise ValueError(f"other_parameters must contain exactly 5 parameters, got {len(other_parameters)}")
        for param in other_parameters:
            if param not in valid_parameters:
                raise ValueError(f"Parameter '{param}' in other_parameters is not valid. Valid parameters: {valid_parameters}")
            if param == target_parameter:
                raise ValueError(f"target_parameter '{target_parameter}' cannot be in other_parameters")
        
        filtered_data = []
        for result in results:
            if result.M != target_M or result.N != target_N or result.K != target_K:
                continue
                
            match = True
            for param, value in other_parameters.items():
                if result.parameters.get(param, None) != value:
                    match = False
                    break
            
            if match and target_parameter in result.parameters:
                filtered_data.append((
                    result.parameters[target_parameter],
                    result.time
                ))
        
        if not filtered_data:
            print(f"No data found matching criteria. Target parameter: {target_parameter}, Other parameters: {other_parameters}")
            return
        
        filtered_data.sort(key=lambda x: x[0])
        param_values, times = zip(*filtered_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, times, 'b-', marker='o', markersize=8, linewidth=2, label=f'Time vs {target_parameter}')
        other_params_str = ", ".join([f"{k}={v}" for k, v in other_parameters.items()])
        plt.title(f'Time vs {target_parameter}\nOther parameters: {other_params_str}', fontsize=14)
        plt.xlabel(target_parameter, fontsize=12)
        plt.ylabel('Time (us)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        filename = f"time_vs_{target_parameter}_M{target_M}_N{target_N}_K{target_K}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to current directory as: {filename}")

if __name__ == "__main__":
    # parameters = Parameter()
    benchmark_runner = GEMMBenchmarkRunner(shape_group)
    benchmark_runner.run_benchmarks()

    # shape = [1024, 512, 256]
    # param_dic = {
    #     'm_sections': 1,
    #     'n_sections': 1,
    #     'm_sec_o_blocks': 3,
    #     'n_sec_o_blocks': 8,
    #     'k_o_iter_blocks': 20,
    #     'db_o_blocks': 10
    # }
    # A, B = benchmark_runner.gen_data(shape)
    # benchmark_runner.deepgemm_gemm(A, B, param_dic)

    # target_parameter = "m_sections"
    # other_parameters = {"n_sections": 4, "m_sec_o_blocks": 8, "n_sec_o_blocks": 8, "k_o_iter_blocks": 16, "db_o_blocks": 4}
    # benchmark_runner.visualize_time_with_single_parameter(shape, target_parameter, other_parameters)
