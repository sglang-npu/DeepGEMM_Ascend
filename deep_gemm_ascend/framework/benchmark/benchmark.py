
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
import math

torch.npu.config.allow_internal_format = False
relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4

shape_group = [
    # M、N、K
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
error_tolerance = 1e-4

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


@dataclass
class Result():
    idx: int
    M: int
    N: int
    K: int
    time: float
    diff: float
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
    def __init__(self, shape_group, result_dir="./results", msp_dir="./msp", msp_bench_path='deep_gemm_ascend\\framework\\tests\\bench_main.py'):
        self.shape_group = shape_group
        self.result_dir = result_dir
        self.parameters = Parameter()
        self.parameter_cache = []
        self.msp_dir = msp_dir
        self.msp_bench_path = msp_bench_path
    
    def benchmark_shape(self, shape: list, checkpoints: str) -> None:
        # gen_data -> deepgemm_gemm && cann_gemm -> is_correct -> ms_prof -> save_result
        shape_str = '_'.join(map(str, shape))
        filename = f'shape_{shape_str}.jsonl'
        result_path = str(Path(self.result_dir) / filename)
        checkpoints_path = str(Path(self.result_dir) / checkpoints)
        
        check_content = dict()
        start_idx = 0
        break_idx = -1
        recall = False 

        if not checkpoints_path: # 如果没有checkpoint，则创建文件
           with open(checkpoints_path, "w", encoding="utf-8") as f:
                check_content['start_idx'] = start_idx
                check_content['break_idx'] = break_idx 
                check_content['recall'] = False
                json.dump(check_content, f, indent=3)
                pass 
        
        # 读取checkpoints内容
        with open(checkpoints_path, "r", encoding="utf-8") as f:
             check_content = json.load(f)
             recall = check_content['recall']
      
        exit(1)
        
        a_npu, b_npu, x_npu, y_npu = self.gen_data(shape)
        gold = self.cann_gemm(x_npu, y_npu)

        saved_count = 0      # 保存的结果数量
        max_saved_idx = -1   # 已保存的最大索引
        if os.path.exists(result_path):
            with jsonlines.open(result_path, mode='r') as reader:
                for result in reader:
                    saved_count += 1
                    if result['idx'] > max_saved_idx:
                        max_saved_idx = result['idx']

        start_idx = max_saved_idx + 1 if max_saved_idx >= 0 else 0
        check_content['start_idx'] = start_idx

        if recall: # 如果先前断过
           break_idx = check_content['break_idx']

        results = []

        total_params = len(self.parameters.filter_parameters(shape))
        with tqdm(total=total_params, initial=start_idx, desc=f"Testing shape {shape}", postfix={"Processed": start_idx, "Valid": saved_count}) as pbar:
            for idx, parameters in enumerate(self.parameters.grid_parameters[start_idx:], start=start_idx):
                if idx == break_idx: # todo 模拟 break_id
                    print(f'遇到break_id, {break_id=}')
                    continue 

                check_content['break_idx'] = idx
                check_content['recall'] = True 
                
                with open(checkpoints_path, "w", encoding="utf-8") as f:
                    json.dump(check_content, f, indent=3)

                # checkpoint 123  100-122， 124-149 150 /   ---- 1/600 
                output, param_npu = self.deepgemm_gemm(a_npu, b_npu, parameters)
                is_diff, diff_prop = self.is_correct(gold, output)
                time_us = self.ms_prof() if diff_prop < error_tolerance else float('inf')
                
                has_negative = torch.any(output < 0).item()
                # self.save_negative_debug_info(has_negative, a_npu, b_npu, output)
                # self.save_params_to_jsonl(param_npu, has_negative, diff_prop)
                result = Result(
                    idx=idx,
                    M=shape[0],
                    N=shape[1],
                    K=shape[2],
                    time=time_us,
                    diff=diff_prop,
                    parameters=parameters
                )
                results.append(result)
                saved_count += 1
                
                if len(results) == 100 or idx == total_params or recall: 
                    if results:
                        self.save_result(results, result_path)
                        results = []

                pbar.update(1)
                pbar.set_postfix({
                    'Processed': idx + 1,
                    'Value': saved_count
                })

        return

    def gen_data(self, shape: list) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        a_npu = torch.tensor(A, device=device, dtype=torch.float16)
        b_npu = torch.tensor(B, device=device, dtype=torch.float16)
        x_npu = torch.tensor(A, device=device, dtype=torch.float32)
        y_npu = torch.tensor(B, device=device, dtype=torch.float32)

        return a_npu, b_npu, x_npu, y_npu

    def deepgemm_gemm(self, a_npu: Tensor, b_npu: Tensor, parameters: dict) -> Tensor:
        param_list = list(parameters.values())
        param_list.extend([0] * 22)
        param_npu = torch.tensor(param_list, device='npu', dtype=torch.int32)

        z_shape = [a_npu.size(0), b_npu.size(1)]
        z_npu = torch.empty(z_shape, device='npu', dtype=torch.float32)

        deep_gemm_ascend.run_mmad_bench(a_npu, b_npu, z_npu, param_npu)
        return z_npu, param_npu

    def cann_gemm(self, A: Tensor, B: Tensor) -> Tensor:
        out = A.to('cpu') @ B.to('cpu')
        return out

    def is_correct(self, cann_result: Tensor, deepgemm_result: Tensor) -> (bool, float):
        output = deepgemm_result.cpu().numpy()
        golden = cann_result.cpu().numpy().astype(np.float32)

        output = output.reshape(-1)
        golden = golden.reshape(-1)

        diff_ele_result = np.isclose(output,
                                     golden,
                                     rtol=relative_tol,
                                     atol=absolute_tol,
                                     equal_nan=True)
        diff_ele_idxs = np.where(diff_ele_result == False)[0]

        error_ratio = float(diff_ele_idxs.size) / golden.size
        return error_ratio <= error_tol, error_ratio

    def ms_prof(self) -> float:
        cmd_str = f"msprof op --output={self.msp_dir} --aic-metrics='PipeUtilization' --kernel-name='mmad' --application='python3 {self.msp_bench_path}'"
        result = subprocess.run()

        # 从ms_prof目录下解析最新的耗时
        all_items = os.listdir(self.msp_dir)
        subdirs = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
        subdirs.sort()
        last_subdir_name = subdirs[-1]
        last_subdir_path = os.path.join(self.msp_dir, last_subdir_name)
        PipeUtilization_path = os.path.join(last_subdir_path, "PipeUtilization.csv")
            
        def parse_time(csv_path: str):
            data = pd.read_csv(csv_path, usecols=['aic_time(us)'])
            return data.iloc[0, 0]
        
        time_us = float(parse_time(PipeUtilization_path))
        return time_us

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
    
    def save_params_to_jsonl(self, params: list, is_negative: bool, diff: float, jsonl_file_path="./params.jsonl") -> None:
        param_names = {
            0: "m_sections",
            1: "n_sections",
            2: "m_sec_o_blocks",
            3: "n_sec_o_blocks",
            4: "k_o_iter_blocks",
            5: "db_o_blocks",
            6: "m",
            7: "k",
            8: "n",
            9: "batch",
            10: "k_iters",
            11: "m_blocks",
            12: "n_blocks",
            13: "k_blocks",
            14: "m_sc_blocks",
            15: "n_sc_blocks",
            16: "m_o_fix",
            17: "n_o_fix",
            18: "k_o_fix",
            19: "db_o_num",
            20: "m_parts",
            21: "n_parts",
            22: "r_m_parts",
            23: "r_n_parts",
            24: "r_m_blocks",
            25: "r_n_blocks",
            26: "r_k_blocks",
            27: "r_db_num"
        }
        data_dict = {}
        for i, param in enumerate(params):
            param_name = param_names.get(i, f"param_{i}")
            data_dict[param_name] = param.item() if hasattr(param, 'item') else param
        data_dict["negative"] = is_negative
        data_dict["diff"] = diff

        self.parameter_cache.append(data_dict)

        if len(self.parameter_cache) >= 100:
            Path(jsonl_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with jsonlines.open(jsonl_file_path, mode='a') as writer:
                writer.write_all(self.parameter_cache)
            
            self.parameter_cache = []

    def save_negative_debug_info(self, has_negative:bool, x_npu:Tensor, y_npu:Tensor, z_npu:Tensor):
        if has_negative:
            print(f"exist negative!!")

            negative_indices = torch.where(z_npu < 0)
            num_negatives = len(negative_indices[0])
            
            x_cpu = x_npu.cpu().numpy().tolist()
            y_cpu = y_npu.cpu().numpy().tolist()
            z_cpu = z_npu.cpu().numpy().tolist()

            jsonl_filename = "negative_debug_info.jsonl"
            with open(jsonl_filename, 'a', encoding='utf-8') as f:
                for idx in range(num_negatives):
                    i = negative_indices[0][idx].item()
                    j = negative_indices[1][idx].item()
                    negative_value = z_npu[i, j].item()
                    x_slice = x_npu[i, :]
                    y_slice = y_npu[:, j]

                    debug_info = {
                        "negative_location": {"row":i, "col":j},
                        "negative_value": negative_value,
                        "x_npu_slice_i_row": x_slice.cpu().numpy().tolist(),
                        "y_npu_slice_j_col": y_slice.cpu().numpy().tolist(),
                        "params_npu": params_npu,
                        "x": x_cpu,
                        "y": y_cpu,
                        "z": z_npu
                    }
            
                    json_line = json.dumps(debug_info, ensure_ascii=False)
                    f.write(json_line + '\n')

            print(f'negative info has saved in {jsonl_filename}')

    def run_benchmarks(self) -> None:
        print("=====STARTING GEMM BENCHMARK=====")
        checkpoints = 'checkpoint.json'
        for shape in self.shape_group:
            self.benchmark_shape(shape，checkpoints)
            
    
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
    # todo-1 解析参数
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
